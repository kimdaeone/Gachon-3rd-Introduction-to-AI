"""Verify saved final scores and render the WISDM holdout report without running inference."""

import argparse
from pathlib import Path

from final_wisdm import (
    ACTIVITIES, SEEDS, TEST_USERS, WEEK, check_result, digest, frozen_cnns, read_json, read_selection,
    require, save_csv,
)


def table(headers, rows):
    return "\n".join("| " + " | ".join(map(str, row)) + " |"
                     for row in (headers, ["---"] * len(headers), *rows))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--notebooks", type=Path, default=WEEK)
    parser.add_argument("--output", type=Path, default=WEEK / "final")
    parser.add_argument("--report", type=Path, default=WEEK / "reports/11_최종_테스트_평가.md")
    args = parser.parse_args()
    selection, selection_hash = read_selection(args)

    import numpy as np
    from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

    fit_dir, holdout = args.output / "refits", args.output / "holdout"
    development = read_json(fit_dir / "development.json")
    evaluation = read_json(holdout / "evaluation.json")
    latency = read_json(args.output / "cpu_latency.json")
    require(all(item["selection_sha256"] == selection_hash for item in (development, evaluation, latency)),
            "Fit, test and benchmark must share the same frozen selection")
    raw = args.output / "cpu_latency_calls.csv"
    require(digest(raw) == latency["raw_calls_sha256"], "Benchmark call log hash mismatch")
    cnn = frozen_cnns(selection)
    require([row["model"] for row in latency["models"]] == list(cnn), "Expected all frozen CNN timings")
    runs, activities, users, confusions = [], [], [], []
    baseline = None
    for name in selection["final_models"]:
        for seed in SEEDS:
            prefix = f"{name}_seed{seed}"
            fit = check_result(fit_dir / f"{prefix}.json", fit_dir, selection_hash)
            prediction = check_result(holdout / f"{prefix}.json", holdout, selection_hash)
            require(prediction["checkpoint_sha256"] == fit["artifacts"][fit["checkpoint"]],
                    f"{prefix}: prediction does not match final checkpoint")
            with np.load(holdout / f"{prefix}_predictions.npz") as data:
                y, pred, people, probs = (data[key] for key in ("y_true", "y_pred", "users", "probabilities"))
            labels = range(len(ACTIVITIES))
            require(y.shape == pred.shape == people.shape == (1411,) and
                    probs.shape == (1411, len(ACTIVITIES)) and np.isfinite(probs).all() and
                    np.allclose(probs.sum(axis=1), 1) and np.array_equal(probs.argmax(1), pred) and
                    set(y) == set(labels) and set(people) == set(TEST_USERS),
                    f"{prefix}: invalid holdout file")
            if baseline is None:
                baseline = y, people
            require(np.array_equal(baseline[0], y) and np.array_equal(baseline[1], people),
                    f"{prefix}: test examples differ")
            precision, recall, f1, support = precision_recall_fscore_support(
                y, pred, labels=labels, zero_division=0)
            cm = confusion_matrix(y, pred, labels=labels)
            accuracies = []
            for person in TEST_USERS:
                mask = people == person
                accuracy = float((pred[mask] == y[mask]).mean())
                users.append(dict(model=name, seed=seed, user=person, windows=int(mask.sum()),
                                  accuracy=accuracy))
                accuracies.append(accuracy)
            runs.append(dict(model=name, seed=seed, windows=len(y), accuracy=float((pred == y).mean()),
                             macro_precision=float(precision.mean()), macro_recall=float(recall.mean()),
                             macro_f1=float(f1.mean()), user_mean_accuracy=float(np.mean(accuracies)),
                             user_min_accuracy=min(accuracies), epochs=fit["epochs"],
                             train_seconds=fit["train_seconds"], params=fit["params"]))
            for index, activity in enumerate(ACTIVITIES):
                activities.append(dict(model=name, seed=seed, activity=activity, windows=int(support[index]),
                                       precision=float(precision[index]), recall=float(recall[index]),
                                       f1=float(f1[index])))
                confusions.append(dict(model=name, seed=seed, true_activity=activity,
                                       **dict(zip(ACTIVITIES, cm[index].tolist()))))

    for filename, rows in (("test_runs.csv", runs), ("test_activities.csv", activities),
                           ("test_users.csv", users), ("test_confusion.csv", confusions)):
        save_csv(args.output / filename, rows)

    def aggregate(values):
        return f"{np.mean(values):.4f} ± {np.std(values, ddof=1):.4f}"

    def values(rows, name, metric, **keys):
        return [row[metric] for row in rows
                if row["model"] == name and all(row[key] == value for key, value in keys.items())]

    winner = selection["selection"]
    train_env, test_env = development["environment"], evaluation["environment"]
    gpu = ", ".join(train_env["gpu"]) or train_env["platform"]
    test_device = ", ".join(test_env["gpu"]) or test_env["platform"]
    gate = selection["b3_c_gate"]
    gate_sentence = (
        f"B3는 B2보다 평균 {gate['mean_delta'] * 100:+.2f}%p, "
        f"{gate['improved_folds']}/5 fold에서 개선됐다. "
        + ("사전 기준을 충족해 B3-C를 학습하고 후보 비교에 포함했다."
           if gate["triggered"] else "사전 기준에 따라 B3-C 학습 단계는 생략했다.")
    )
    selection_rows = [[row["model"], f'{row["macro_f1"]:.4f} ± {row["fold_std"]:.4f}',
                       f'{row["params"]:,}', row["refit_epochs"]]
                      for row in selection["cnn_development"]]
    mean_rows = [[name] + [aggregate(values(runs, name, metric)) for metric in
                           ("accuracy", "macro_precision", "macro_recall", "macro_f1", "user_mean_accuracy",
                            "user_min_accuracy")] for name in selection["final_models"]]
    seed_rows = [[row["model"], row["seed"], row["epochs"] if row["epochs"] is not None else "—",
                  f'{row["accuracy"]:.4f}', f'{row["macro_f1"]:.4f}', f'{row["train_seconds"]:.1f}']
                 for row in runs]
    activity_rows = [[activity, next(row["windows"] for row in activities if row["activity"] == activity)] +
                     [aggregate(values(activities, name, "recall", activity=activity))
                      for name in selection["final_models"]] for activity in ACTIVITIES]
    user_rows = [[person, next(row["windows"] for row in users if row["user"] == person)] +
                 [aggregate(values(users, name, "accuracy", user=person))
                  for name in selection["final_models"]] for person in TEST_USERS]
    timings = [[row["model"], f'{row["params"]:,}', f'{row["median_ms"]:.3f}',
                f'{row["q1_ms"]:.3f}–{row["q3_ms"]:.3f}', f'{row["iqr_ms"]:.3f}']
               for row in latency["models"]]
    comparisons = []
    for reference in ("A0", "RF"):
        if reference == winner:
            continue
        deltas = [100 * (a - b) for a, b in zip(values(runs, winner, "macro_f1"),
                                              values(runs, reference, "macro_f1"))]
        comparisons.append(f"{reference} 대비 {winner}의 테스트 Macro F1 차이는 평균 "
                           f"{np.mean(deltas):+.2f}%p였다. "
                           f"seed 0·1·2 차이는 {', '.join(f'{d:+.2f}%p' for d in deltas)}였다.")
    winner_f1 = float(np.mean(values(runs, winner, "macro_f1")))
    reference_line = (f"{winner}의 평균 Macro F1은 {winner_f1:.4f}이며 "
                      + ("0.90 참고선에 도달했다." if winner_f1 >= 0.9 else "0.90 참고선 아래였다."))
    latency_by_name = {row["model"]: row for row in latency["models"]}
    b1, b4 = latency_by_name["B1"], latency_by_name["B4"]
    parameter_change = (b4["params"] / b1["params"] - 1) * 100
    latency_change = (b4["median_ms"] / b1["median_ms"] - 1) * 100
    recovery_path = args.output / "runtime_recovery.json"
    recovery_note = ""
    if recovery_path.exists():
        recovery = read_json(recovery_path)
        require(recovery["selection_sha256"] == selection_hash, "Recovery record must match frozen selection")
        recovery_note = f'\n{recovery["report_note"]} [런타임 복구 기록](../final/runtime_recovery.json)\n'

    report = f"""# 11. WISDM 최종 테스트 평가와 CPU 추론시간

[연구계획서](../연구계획서.md)에 따라 개발 30명에서 **{winner}**를 선택했다. CNN {len(cnn)}개를 동일한 개발 fold 5개 × seed 3개로 비교한 뒤 선택과 학습 epoch를 기록했다. 테스트 사용자 6명에서는 선택 CNN과 A0·RF의 고정된 seed를 평가했다.

## 개발 결과와 최종 설정 고정

동일 모델에서 먼저 seed 3개를 fold별로 평균했다. 표는 fold 5개의 평균과 표본 표준편차를 보여준다. 최고 평균 Macro F1을 기준으로 선택하며, 동점이면 파라미터 수와 모델명 순서를 적용한다. CNN 재학습 epoch는 해당 모델의 개발 best epoch 15개의 중앙값을 반올림해 고정했다.

{table(["모델", "개발 Macro F1", "파라미터", "고정 epoch"], selection_rows)}

같은 개발 fold에서 RF 참조선의 Macro F1은 {selection["rf_development"]["macro_f1"]:.4f} ± {selection["rf_development"]["fold_std"]:.4f}였다. 개발 구조 비교와 사전 판정은 [1단계 개발 검증 종합](10_1단계_개발_검증_종합.md)에 정리했다.

{gate_sentence}

개발 결과와 소스·입력 해시는 {selection["frozen_at"]} (UTC)에 [선택 기록](../final/selection.json)으로 보존했다. 테스트 평가 시작은 {evaluation["started_at"]} (UTC)이다.

## 최종 재학습과 테스트 조건

- 개발 사용자 30명의 정제된 데이터에서 stride 64로 **{development["windows"]:,}개** 학습 윈도우를 만들었다. CNN 표준화는 이 30명의 원본 축별 표본으로 다시 계산했다. Adam 0.001, batch 64, 고정 epoch를 사용했다.
- RF는 stride 64 윈도우의 원본 스케일에서 통계 특징 {development["rf_features"]}개를 계산했고, 고정 설정의 트리 300개로 다시 학습했다.
- 사용자 {", ".join(map(str, TEST_USERS))}의 128 × 3 입력에서 stride 128로 테스트 윈도우 **1,411개**를 구성했다. 각 모델의 seed 0·1·2를 동일한 윈도우에서 평가했다.
- CNN 재학습은 {gpu}, Python {train_env["python"]}, TensorFlow {train_env["tensorflow"]}, Keras {train_env["keras"]}, NumPy {train_env["numpy"]}에서 실행했다. CNN 평가에는 {test_device}, TensorFlow {test_env["tensorflow"]}을 사용했다. RF 재학습과 평가는 같은 런타임의 CPU에서 scikit-learn {test_env["scikit_learn"]}으로 실행했다. 개발 RF 재현 기록의 scikit-learn은 1.9.1이다. [표준화 값 및 실행 환경](../final/refits/development.json)
{recovery_note}
## 테스트 성능

평균과 표본 표준편차는 고정된 테스트 6명에서 seed 3개를 반복한 결과다. 사용자 동일 가중치 Accuracy는 각 seed에서 사용자별 Accuracy를 먼저 계산해 단순 평균했다.

{table(["모델", "Accuracy", "Macro Precision", "Macro Recall", "Macro F1",
        "사용자 동일 가중치 Accuracy", "seed별 최저 사용자 Accuracy"], mean_rows)}

{table(["모델", "seed", "epoch", "Accuracy", "Macro F1", "학습 초"], seed_rows)}

{" ".join(comparisons)}

{reference_line}

### 행동별 Recall

{table(["행동", "윈도우", *selection["final_models"]], activity_rows)}

### 사용자별 Accuracy

{table(["사용자", "윈도우", *selection["final_models"]], user_rows)}

[seed별 전체 지표](../final/test_runs.csv) · [행동별 Precision·Recall·F1](../final/test_activities.csv) · [사용자별 Accuracy](../final/test_users.csv) · [혼동행렬](../final/test_confusion.csv)

## CPU 추론시간

{latency["processor"]} PC에서 TensorFlow {latency["tensorflow"]}, Keras {latency["keras"]}, CPU intra-op·inter-op 각 1스레드, 배치 1, float32 입력 (1, 128, 3)을 사용했다. oneDNN은 모든 모델에서 비활성화했다. CNN {len(cnn)}개에 동일한 합성 입력과 각 모델의 개발 fold 0·seed 0 가중치를 적용했다. 모델마다 50회 워밍업 후 200회씩 5차례 호출했다. 측정은 컴파일된 모델 호출과 NumPy 출력 변환을 포함한다. 가중치 로딩과 전처리는 측정 범위 밖이다. 각 모델의 1,000회 호출에서 중앙값과 사분위수를 계산했다.

{table(["모델", "파라미터", "중앙값 (ms)", "Q1–Q3 (ms)", "IQR (ms)"], timings)}

B1 대비 B4의 파라미터 수 변화는 {parameter_change:+.1f}%, CPU 중앙 지연시간 변화는 {latency_change:+.1f}%였다.

[측정 조건과 요약](../final/cpu_latency.json) · [호출별 시간](../final/cpu_latency_calls.csv)

## 해석 범위

사전 개선·악화·경량화 유지 판정은 개발 fold 비교에 적용한다. 테스트는 사용자 6명의 고정된 사례이므로 더 넓은 모집단의 성능이나 기기·휴대 위치 변화에 대한 일반화에는 한계가 있다. seed 간 표준편차에는 새 사용자 표본의 불확실성이 포함되지 않는다. 연구 초기에는 전체 데이터의 품질과 라벨 분포를 살폈다. CPU 지연시간은 이 장비와 실행 조건에 한정된다.

## 재현

저장소 루트에서 `python 9week/scripts/final_wisdm.py freeze`로 완료된 개발 결과를 확인한다. GPU 런타임에서 같은 선택 기록과 노트북·아카이브를 사용해 `refit`과 `evaluate`를 순서대로 실행한다. PC에서 `python 9week/scripts/benchmark_wisdm.py`로 CNN 지연시간을 측정한다. 저장된 예측과 체크포인트로 `python 9week/scripts/report_final_wisdm.py`를 실행하면 보고서를 다시 생성할 수 있다. 각 스크립트의 `--help`에서 경로 인수를 확인할 수 있다.

[최종 평가 코드](../scripts/final_wisdm.py) · [CPU 벤치마크 코드](../scripts/benchmark_wisdm.py) · [보고서 생성 코드](../scripts/report_final_wisdm.py)
"""
    args.report.write_text(report, encoding="utf-8")
    print(f"Verified all final artifacts and wrote {args.report}", flush=True)


if __name__ == "__main__":
    main()
