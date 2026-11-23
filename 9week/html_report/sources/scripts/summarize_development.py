"""Verify and summarize the seven WISDM development experiments before holdout."""

import argparse
import csv
import json
import math
import statistics
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score


ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / 'runs'
REPORTS = ROOT / 'reports'
MODELS = ('A0', 'A1', 'A2', 'B1', 'B2', 'B3', 'B4')
ABLATION = 'B3-C'
FOLDS = range(5)
SEEDS = range(3)
EXPECTED = {(fold, seed) for fold in FOLDS for seed in SEEDS}
TEST_USERS = [6, 21, 24, 27, 29, 34]
DEVELOPMENT_WINDOWS = 6430
COMPARISONS = (
    ('Q1', 'A1', 'A0', '일반 합성곱에서 residual 추가'),
    ('Q1', 'A2', 'A0', 'residual 없이 Inception 도입'),
    ('Q1', 'B1', 'A2', 'Inception에서 residual 추가'),
    ('Q1', 'B1', 'A1', 'residual을 유지하며 Inception 도입'),
    ('Q1', 'B1', 'A0', '기본 CNN에서 두 변경을 함께 적용'),
    ('Q2', 'B2', 'B1', 'SE 추가'),
    ('Q2', 'B3', 'B1', 'CBAM 추가'),
    ('Q2', 'B3', 'B2', 'CBAM과 SE 비교'),
    ('Q3', 'B4', 'B1', 'Inception 시간 합성곱을 분리 합성곱으로 교체'),
)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8'))


def index_runs(rows, name):
    indexed = {(int(row['fold']), int(row['seed'])): row for row in rows}
    missing = sorted(EXPECTED - indexed.keys())
    extra = sorted(indexed.keys() - EXPECTED)
    require(len(rows) == len(indexed) and not missing and not extra,
            f'{name}: 15개 고유 실행이 필요합니다. 누락={missing}, 추가={extra}, 기록 수={len(rows)}')
    require(all(row['model'] == name for row in rows), f'{name}: 모델 이름 불일치')
    return indexed


def validate_predictions(name, rows, groups):
    """Check the fixed validation subjects and label order, then reproduce metrics."""
    validation_count = 0
    for fold, seed in sorted(EXPECTED):
        row = rows[fold, seed]
        args = f'{name}_fold{fold}_seed{seed}_predictions.npz'
        reference = RUNS / f'A0_fold{fold}_seed0_predictions.npz'
        with np.load(RUNS / args) as saved, np.load(reference) as baseline:
            y_true, y_pred, users = (saved[key] for key in ('y_true', 'y_pred', 'users'))
            require(set(users) == set(groups[fold]), f'{args}: 고정 개발 사용자 불일치')
            require(np.array_equal(y_true, baseline['y_true'])
                    and np.array_equal(users, baseline['users']), f'{args}: 검증 윈도우 불일치')
            if name != 'RF':
                require(np.allclose(saved['mean'], baseline['mean'], rtol=1e-6, atol=1e-8)
                        and np.allclose(saved['std'], baseline['std'], rtol=1e-6, atol=1e-8),
                        f'{args}: 표준화 값 불일치')
            f1 = f1_score(y_true, y_pred, labels=range(6), average='macro', zero_division=0)
            require(math.isclose(f1, float(row['macro_f1']), rel_tol=0, abs_tol=1e-12)
                    and math.isclose(float(np.mean(y_true == y_pred)), float(row['accuracy']),
                                     rel_tol=0, abs_tol=1e-12), f'{args}: 저장 지표 불일치')
            if seed == 0:
                validation_count += len(y_true)
    require(validation_count == DEVELOPMENT_WINDOWS, f'{name}: 개발 윈도우 수 불일치')


def fold_means(rows, field):
    return [statistics.mean(float(rows[fold, seed][field]) for seed in SEEDS) for fold in FOLDS]


def load_results(include_ablation=False):
    reference = None
    results = {}
    model_names = (*MODELS, ABLATION) if include_ablation else MODELS
    for model in model_names:
        path = RUNS / f'{model}_progress.json'
        progress = read_json(path)
        settings = dict(progress['protocol'])
        require(settings.pop('model') == model, f'{path.name}: 프로토콜 모델 불일치')
        if reference is None:
            reference = settings
            groups = settings['groups']
            require(len(groups) == 5 and all(len(group) == 6 for group in groups), '개발 분할 크기 불일치')
            development_users = [user for group in groups for user in group]
            require(settings['test_users'] == TEST_USERS
                    and sorted(development_users + TEST_USERS) == list(range(1, 37)), '보류 분할 불일치')
        require(reference == settings, f'{path.name}: A0와 학습 프로토콜 불일치')
        rows = index_runs(progress['rows'], model)
        params = {row['params'] for row in rows.values()}
        require(len(params) == 1 and all(p > 0 for p in params), f'{model}: 파라미터 수 불일치')
        validate_predictions(model, rows, groups)
        epoch = int(statistics.median(row['best_epoch'] for row in rows.values()) + 0.5)
        require(1 <= epoch <= settings['max_epochs'], f'{model}: 모델의 고정 epoch 범위 오류')
        f1 = fold_means(rows, 'macro_f1')
        results[model] = {'f1': f1, 'mean': statistics.mean(f1), 'sd': statistics.stdev(f1),
                          'accuracy': statistics.mean(fold_means(rows, 'accuracy')),
                          'seconds': sum(row['train_sec'] for row in rows.values()),
                          'params': params.pop(), 'epoch': epoch}

    rf = read_json(RUNS / 'RF_reproduction.json')
    require(rf['groups'] == groups and rf['test_users'] == TEST_USERS
            and rf['archive_sha256'] == reference['data_sha256'], 'RF 참조선의 데이터 또는 사용자 분할 불일치')
    with (RUNS / 'RF_runs.csv').open(encoding='utf-8', newline='') as source:
        rows = index_runs(list(csv.DictReader(source)), 'RF')
    validate_predictions('RF', rows, groups)
    f1 = fold_means(rows, 'macro_f1')
    results['RF'] = {'f1': f1, 'mean': statistics.mean(f1), 'sd': statistics.stdev(f1),
                     'accuracy': statistics.mean(fold_means(rows, 'accuracy'))}
    return results, reference


def paired_difference(results, model, baseline):
    return [a - b for a, b in zip(results[model]['f1'], results[baseline]['f1'])]


def verdict(delta):
    mean = statistics.mean(delta)
    if mean >= 0.005 and sum(value > 0 for value in delta) >= 4:
        return '개선'
    if mean <= -0.005 and sum(value < 0 for value in delta) >= 4:
        return '악화'
    return '구별 불가'


def pp(value):
    return f'{100 * value:+.2f}'


def table(headers, rows):
    return '\n'.join('|' + '|'.join(f' {value} ' for value in row) + '|'
                     for row in (headers, ['---'] * len(headers), *rows))


def latency_summary(path, results):
    if path is None:
        return '같은 CPU·스레드·batch 1 설정의 별도 측정으로 추론 효율을 확인한다.'
    benchmark = read_json(path)
    require(benchmark['input_shape'] == [1, 128, 3]
            and benchmark['device'] == 'CPU:0'
            and benchmark['intra_op_threads'] == benchmark['inter_op_threads'] == 1
            and benchmark['warmup_calls'] == 50
            and benchmark['trials'] == 5 and benchmark['calls_per_trial'] == 200,
            'CPU 측정 설정이 연구계획서와 다릅니다.')
    measurements = benchmark['models']
    by_model = {entry['model']: entry for entry in measurements}
    model_names = tuple(model for model in results if model != 'RF')
    require(len(measurements) == len(model_names) and set(by_model) == set(model_names), 'CPU 측정의 모델 집합 불일치')
    rows = []
    for model in model_names:
        entry = by_model[model]
        median, q1, q3 = (float(entry[key]) for key in ('median_ms', 'q1_ms', 'q3_ms'))
        require(entry['params'] == results[model]['params'], f'{model}: CPU 측정의 파라미터 수 불일치')
        require(0 < q1 <= median <= q3 and math.isfinite(q3), f'{model}: CPU 지연 시간 범위 오류')
        rows.append([model, f'{median:.3f}', f'{q1:.3f}–{q3:.3f}'])
    b1_median, b4_median = (float(by_model[model]['median_ms']) for model in ('B1', 'B4'))
    caption = table(['모델', 'CPU 중앙값(ms)', '제1–제3사분위수(ms)'], rows)
    source = (' [CPU 측정 기록](../final/cpu_latency.json) · [호출별 시간](../final/cpu_latency_calls.csv)'
              if path.resolve() == ROOT / 'final' / 'cpu_latency.json' else '')
    relative_difference = (b4_median / b1_median - 1) * 100
    return (f"{benchmark['processor']}에서 TensorFlow {benchmark['tensorflow']}로 측정했다. "
            f'스레드 1개와 batch 1에서 모델별 워밍업 50회 후 200회씩 5차례 실행했다. '
            f'아래 수치는 출력 변환을 포함한 호출 1,000회의 중앙값과 사분위수다. '
            f'B4−B1의 지연 중앙값 차이는 {b4_median - b1_median:+.3f}ms({relative_difference:+.1f}%)다.'
            f'{source}\n\n{caption}')


def render(results, protocol, stage_two, stage_two_reason, stage_three, stage_three_reason, latency_path=None):
    def rank(model):
        return (-results[model]['mean'], results[model]['params'], model)

    primary_candidate = min(MODELS, key=rank)
    ablation_complete = ABLATION in results
    cnn_models = (*MODELS, ABLATION) if ablation_complete else MODELS
    candidate = min(cnn_models, key=rank)
    summary_rows = []
    for model in (*cnn_models, 'RF'):
        item = results[model]
        params = f"{item['params']:,}" if model != 'RF' else '—'
        elapsed = f"{item['seconds']:.1f}" if model != 'RF' else '—'
        summary_rows.append([model, f"{item['mean']:.4f} ± {item['sd']:.4f}", f"{item['accuracy']:.4f}",
                             '도달' if item['mean'] >= 0.90 else '미달', params, elapsed])
    comparisons = []
    for question, model, baseline, description in COMPARISONS:
        delta = paired_difference(results, model, baseline)
        comparisons.append([question, f'{model}−{baseline}', description, *map(pp, delta),
                            pp(statistics.mean(delta)), f'{sum(d > 0 for d in delta)}/5', verdict(delta)])

    interaction = [inception - standard for inception, standard in
                   zip(paired_difference(results, 'B1', 'A2'), paired_difference(results, 'A1', 'A0'))]
    b4_loss = statistics.mean(paired_difference(results, 'B1', 'B4'))
    b1_params, b4_params = results['B1']['params'], results['B4']['params']
    parameter_reduction = (b1_params - b4_params) / b1_params
    q3 = '경량화 유지' if b4_loss <= 0.01 and b4_params < b1_params else '유지 실패'
    attention_delta = paired_difference(results, 'B3', 'B2')
    attention_verdict = verdict(attention_delta)
    stage_three_triggered = attention_verdict == '개선'
    require(not ablation_complete or (stage_three_triggered and stage_three == 'run'),
            'B3-C를 포함하려면 사전 개선 조건과 --stage-three run이 필요합니다.')
    ablation_note = ''
    if stage_three_triggered:
        require(stage_three in {'run', 'defer'} and bool(stage_three_reason),
                'B3가 B2 대비 사전 개선 기준을 충족했습니다. --stage-three와 --stage-three-reason을 기록하세요.')
        stage_three_decision = 'B3-C를 실행한다.' if stage_three == 'run' else 'B3-C는 보류한다.'
        if ablation_complete:
            stage_three_decision = 'B3-C 개발 검증 15회를 완료했다.'
            temporal_delta = paired_difference(results, 'B3', ABLATION)
            comparisons.append(['3단계', 'B3 − B3-C', '같은 채널 attention에서 시간 attention 추가',
                                *map(pp, temporal_delta), pp(statistics.mean(temporal_delta)),
                                f'{sum(d > 0 for d in temporal_delta)}/5', verdict(temporal_delta)])
            ablation_note = ('\n\nB3−B3-C는 같은 채널 attention에서 시간 attention을 추가한 방향의 차이다. '
                             f'평균 {pp(statistics.mean(temporal_delta))}%p이고 사전 판정은 '
                             f'**{verdict(temporal_delta)}**다. [B3-C 연구보고서](10_B3-C_연구보고서.md)')
        stage_three_status = f'발동 조건을 충족했다. {stage_three_decision} {stage_three_reason}'
    else:
        require(stage_three in {None, 'condition-unmet'}, 'B3-C는 사전 개선 조건을 충족한 경우에만 검토합니다.')
        stage_three_status = '발동 조건이 충족되지 않아 B3-C를 실행하지 않는다.'
    stage_two_status = '실행한다' if stage_two == 'run' else '보류한다'
    b6_status = 'B6-C는 B6 학습을 마친 뒤 검토한다.' if stage_two == 'run' else 'B6-C는 보류한다.'
    if not stage_three_triggered:
        b6_status = 'B6-C에도 같은 발동 조건을 적용한다.'

    model_summary = table(['구조', 'Macro F1 평균 ± 표준편차', 'Accuracy 평균', 'F1 0.90',
                           '파라미터', '학습 시간 합계(초)'], summary_rows)
    paired_summary = table(['질문', '비교', '변경 사항', *[f'fold {f}' for f in FOLDS],
                            '평균', '상승 fold', '판정'], comparisons)
    interaction_summary = table([*[f'fold {f}' for f in FOLDS], '평균'],
                                [[*map(pp, interaction), pp(statistics.mean(interaction))]])
    fixed_cnn = tuple(dict.fromkeys((candidate, 'A0')))
    locked_summary = table(['모델', '최종 학습 재설정'],
                           [[model, f"{results[model]['epoch']} epoch × seed 0·1·2"] for model in fixed_cnn]
                           + [['RF', '기존 특징과 모델 설정, seed 0·1·2']])
    if stage_two == 'run' or (stage_three == 'run' and not ablation_complete):
        candidate_note = (f'{primary_candidate}는 1단계 기준 상위 구조다. 확장 실험 검토가 끝나면 개발 결과로 '
                          '최종 후보와 학습 epoch를 고정한다. 그 전까지 보류 테스트를 열지 않는다.')
    else:
        candidate_note = (f'완료된 CNN 구조 중 개발 평균 Macro F1, 파라미터 수, 모델명 순서로 후보를 선택했다. '
                          f'{candidate} 구조를 후보로 고정하고 A0와 RF를 참조선에 포함한다. CNN은 개발 30명으로 '
                          '표준화를 다시 계산하고 15개 최적 epoch의 중앙값을 반올림해 고정한다.\n\n'
                          f'{locked_summary}')
    total_training_time = sum(results[model]['seconds'] for model in MODELS)
    rf_delta = paired_difference(results, primary_candidate, 'RF')

    return f"""# WISDM 1단계 개발 검증 종합

필수 CNN 7개 구조의 사용자 독립 교차검증 105회와 RF 참조선 15회를 비교한다. 보류 테스트의 예측값과 지표는 사용하지 않았다. 필수 7개 구조 중 개발 평균 Macro F1 기준 상위 CNN은 **{primary_candidate}**({results[primary_candidate]['mean']:.4f})다.

## 실험과 집계

WISDM v1.1의 개발 사용자 30명, 5개 fold, seed 0·1·2를 사용했다. 각 fold에서 학습 사용자는 24명, 검증 사용자는 6명이다. 개발 검증 윈도우 6,430개의 정답과 사용자 순서, CNN의 학습 사용자 기반 표준화 값, 저장 Accuracy와 Macro F1을 대조했다. 데이터 SHA-256은 `{protocol['data_sha256']}`다.

fold별로 seed 3개의 지표를 먼저 평균했다. 표의 표준편차는 fold 5개의 표본 표준편차다. F1 0.90은 절대 수준 참고선이다.

{model_summary}

{primary_candidate}의 RF 대비 개발 Macro F1 차이는 평균 {pp(statistics.mean(rf_delta))}%p이며 {sum(d > 0 for d in rf_delta)}/5 fold에서 높았다. RF는 같은 사용자·검증 윈도우의 통계 특징 모델이다. CNN에서는 검증 Macro F1로 epoch를 선택했고 RF에는 이 과정이 없었다.

## Q1·Q2·Q3의 짝지은 비교

차이는 같은 fold의 Macro F1 차이(%p)다. 평균 차이 +0.5%p 이상 및 4/5 fold 이상 상승이면 개선, −0.5%p 이하 및 4/5 fold 이상 하락이면 악화로 판정했다. 그 외에는 사전 기준상 구별 불가다. 표시는 소수점 둘째 자리로 반올림했고 판정은 반올림 전 값으로 했다.

{paired_summary}

### Q1의 상호작용

일반 합성곱과 Inception에서 residual을 추가했을 때의 차이를 `(B1−A2)−(A1−A0)`로 비교했다. 양수일수록 이 구현의 Inception에서 residual 추가 이득이 컸다는 관측이다. 아래 값은 %p이며 보조 분석으로 보고한다. Inception을 도입하면 파라미터 수와 수용 영역도 달라진다.

{interaction_summary}

### Q2의 attention 비교

B2−B1 및 B3−B1은 각각 SE와 CBAM을 추가한 구현의 전체 효과다. B3−B2는 평균 {pp(statistics.mean(attention_delta))}%p로 사전 판정은 **{attention_verdict}**다. 두 구조는 채널 attention 계산에서도 차이가 있어 시간 attention의 기여는 B3와 채널 attention만 남긴 B3-C를 직접 비교해야 분리할 수 있다.{ablation_note}

### Q3와 효율

B4의 B1 대비 Macro F1 손실은 {pp(b4_loss)}%p다. 파라미터 수는 B1 {b1_params:,}개, B4 {b4_params:,}개이며 B1 기준 감소율은 {100 * parameter_reduction:.1f}%다. 손실이 1.0%p 이내이고 파라미터도 감소할 때 적용하는 사전 판정은 **{q3}**다.

표의 학습 시간은 `model.fit` 합계이며 총 {total_training_time / 60:.1f}분이다. 런타임과 선택 epoch가 달라질 수 있다. 파라미터 수만으로 CPU 지연이나 전력 사용량을 추정할 수 없다.

{latency_summary(latency_path, results)}

## 테스트 열람 전 범위와 후보 기록

- 2단계(B5·B6·A3·A4·A5)는 {stage_two_status}. {stage_two_reason} 같은 반복 설계에서는 75회 추가 학습이 필요하다.
- 3단계 B3-C의 ‘B3가 B2보다 좋은 경우’는 계획서의 사전 개선 규칙을 적용했다. {stage_three_status} {b6_status}

{candidate_note}

## 해석의 한계

검증 Macro F1로 epoch와 모델 구조를 선택했으므로 개발 점수에는 선택 편향이 있을 수 있다. fold 사이 학습 사용자가 겹치고 seed는 같은 검증 사용자를 반복하므로 fold 표준편차를 신뢰구간으로 사용하거나 15회를 독립 표본으로 취급할 수 없다. 이 비교는 통계적 유의성이나 비열등성 검정을 수행하지 않는다. 데이터 품질과 라벨 분포를 분석할 때 전체 데이터를 살펴봤고, 보류 사용자는 모델 학습과 개발 검증에서 제외했다. 보류 사용자 6명과 통제된 수집 환경은 다른 기기, 휴대 위치, 일상 환경으로의 일반화에 한계를 둔다.

[연구계획서](../연구계획서.md) · [모델별 보고서](README.md) · [종합 계산 코드](../scripts/summarize_development.py)
"""


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage-two', required=True, choices=('run', 'defer'), help='테스트 열람 전 2단계 결정')
    parser.add_argument('--stage-two-reason', required=True, help='2단계 결정 사유')
    parser.add_argument('--stage-three', choices=('run', 'defer', 'condition-unmet'), help='B3-C 실행 여부')
    parser.add_argument('--stage-three-reason', help='발동 조건 충족 시 B3-C 결정 사유')
    parser.add_argument('--include-b3-c', action='store_true', help='완료된 B3-C 검증 15회까지 종합')
    parser.add_argument('--latency-json', type=Path, help='검증과 별도로 측정한 CPU 추론시간 JSON')
    args = parser.parse_args()
    stage_two_reason = args.stage_two_reason.strip()
    stage_three_reason = (args.stage_three_reason or '').strip()
    if not stage_two_reason:
        parser.error('2단계 결정 사유를 입력하세요.')
    try:
        results, protocol = load_results(args.include_b3_c)
        content = render(results, protocol, args.stage_two, stage_two_reason, args.stage_three,
                         stage_three_reason, args.latency_json)
    except (FileNotFoundError, KeyError, ValueError) as error:
        parser.exit(1, f'{error}\n')
    destination = REPORTS / '10_1단계_개발_검증_종합.md'
    destination.write_text(content, encoding='utf-8')
    print(destination)


if __name__ == '__main__':
    main()
