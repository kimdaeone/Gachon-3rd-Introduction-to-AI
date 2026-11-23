"""Verify one completed CNN experiment and write its standalone report.

Run from the repository root: python 9week/scripts/report_cnn.py A1
"""
from pathlib import Path
import json, sys
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'reports'
ACT=['Walking','Jogging','Upstairs','Downstairs','Sitting','Standing']
SPEC={
 'A1':('04','A0 + residual',['A0'],'같은 폭과 깊이에서 residual 연결 추가의 효과'),
 'A2':('05','Inception',['A0'],'여러 길이의 시간 합성곱을 사용하는 구조의 효과'),
 'B1':('06','Inception + ResNet',['A0','A1','A2'],'Inception에 residual을 추가한 효과와 결합 효과'),
 'B2':('07','Inception + ResNet + SE',['B1'],'residual 경로의 채널 attention 효과'),
 'B3':('08','Inception + ResNet + CBAM',['B1','B2'],'채널·시간 attention 구현의 효과'),
 'B4':('09','Depthwise Inception + ResNet',['B1'],'시간 합성곱 경량화에 따른 성능과 파라미터 변화'),
 'B3-C':('10','Inception + ResNet + CBAM 채널 attention',['B3','B2'],'같은 CBAM 채널 attention에서 시간 attention을 추가한 효과'),
}

def table(headers,rows):
 return '| '+' | '.join(headers)+' |\n| '+' | '.join(['---']*len(headers))+' |\n'+'\n'.join('| '+' | '.join(map(str,r))+' |' for r in rows)+'\n'

LESSON = {
 'A1': '블록의 두 합성곱이 입력 x를 변환한 결과를 F(x)라고 하자. A0의 ReLU(F(x))가 A1에서는 ReLU(F(x)+x)가 된다. 두 경로 모두 시간 길이와 64채널을 유지해 같은 위치의 값을 더할 수 있다. 덧셈에 새 가중치가 필요하지 않아 두 모델은 모두 84,870개 파라미터를 갖는다. 손실의 기울기가 입력으로 직접 전달되는 경로를 하나 추가한 셈이다. 이 실험에서는 같은 검증 사람끼리 성능과 선택 epoch를 비교해 이 변화의 영향을 읽는다.',
 'A2': '블록 안에서 1×1 필터가 각 시점의 64채널을 16채널로 조합한다. 여기서 kernel 3·7·15의 세 분기가 서로 다른 시간 폭의 특징을 계산한다. 별도 pooling 분기까지 네 결과를 채널 방향으로 이어 붙이면 16×4=64채널이 된다. 짧은 움직임과 긴 움직임을 함께 표현할 수 있다는 가설이다. A2−A0에는 필터 크기, 병목, 분기, 파라미터 수의 변화가 함께 들어간다.',
 'B1': 'A2의 Inception 블록에서 만든 특징 F(x)에 블록 입력 x를 더한 뒤 ReLU를 적용한다. A1−A0는 기본 합성곱에서 residual을 추가한 차이이고 B1−A2는 여러 분기를 가진 합성곱에서 같은 연결을 추가한 차이다. 두 차이를 같은 fold 안에서 다시 빼면 residual 효과가 블록 종류에 따라 어떻게 달라졌는지 읽을 수 있다.',
 'B2': 'SE는 각 채널의 값을 시간 방향으로 평균해 64개 요약값을 만든다. Dense 층이 64→8→64로 변환하고 sigmoid가 각 채널에 적용할 가중치를 0~1로 정한다. 한 채널 안에서는 모든 시점에 같은 값을 곱한다. 이렇게 조절한 특징을 블록 입력에 더한다. B2−B1은 입력 구간에 따라 채널의 상대적인 크기를 조절하는 연산의 효과를 보여준다.',
 'B3': 'CBAM은 시간 평균과 최댓값으로 채널을 요약하고 같은 두 Dense 층을 통과시켜 채널 가중치를 만든다. 이어 각 시점의 채널 평균과 최댓값을 kernel 7 합성곱에 넣어 시간 가중치를 만든다. 첫 단계는 어떤 특징에, 둘째 단계는 어느 시점에 큰 값을 곱할지 결정한다. B3−B1은 두 단계를 합친 효과를 보여준다. B2와는 채널 계산도 달라 시간 가중치만의 기여를 알고 싶으면 B3-C가 필요하다.',
 'B4': 'Inception의 시간 합성곱 세 분기에서 각 채널별로 시간 필터를 적용하고 1×1 필터로 채널을 조합한다. 예컨대 kernel 15, 입력·출력 16채널 분기에서 편향을 제외하면 일반 합성곱은 15×16×16=3,840개, 분리 합성곱은 15×16+16×16=496개 가중치를 쓴다. 전체 모델에는 다른 층도 있으므로 전체 감소율을 따로 계산한다. B1과 같은 입력을 CPU에서 반복 측정하면 연산 구조 변화가 실제 응답시간에 반영되는지도 판단할 수 있다.',
 'B3-C': 'B3의 시간 평균·최댓값 요약, 공유 Dense, 채널 sigmoid 게이트를 유지한다. 채널 attention이 적용된 특징을 residual 경로에 바로 넘겨 시간 방향 게이트를 제거한다. B3−B3-C는 시간 가중치 계산을 추가한 효과다. 추가 연산과 가중치 수도 함께 달라진다.',
}

def main(name):
 number,structure,comparators,question=SPEC[name]
 runs=pd.read_csv(ROOT/f'runs/{name}_runs.csv')
 users=pd.read_csv(ROOT/f'runs/{name}_users.csv')
 progress=json.loads((ROOT/f'runs/{name}_progress.json').read_text(encoding='utf-8'))
 protocol=progress['protocol'];groups=protocol['groups']
 expected={(f,s) for f in range(5) for s in range(3)}
 assert len(runs)==15 and set(zip(runs.fold,runs.seed))==expected
 assert len(users)==90 and not users.duplicated(['fold','seed','user']).any()
 assert len(progress['rows'])==15 and runs.params.nunique()==1
 metrics=[];classes=[];cm=np.zeros((6,6),int);counts=np.zeros(6,int)
 histories={}
 for r in runs.itertuples():
  prefix=ROOT/f'runs/{name}_fold{r.fold}_seed{r.seed}'
  d=np.load(str(prefix)+'_predictions.npz');y,p,u=d['y_true'],d['y_pred'],d['users']
  baseline=np.load(ROOT/f'runs/A0_fold{r.fold}_seed{r.seed}_predictions.npz')
  assert np.array_equal(y,baseline['y_true']) and np.array_equal(u,baseline['users'])
  assert set(u)==set(groups[r.fold]) and not set(u)&set(protocol['test_users'])
  assert np.isfinite(d['mean']).all() and np.isfinite(d['std']).all() and (d['std']>0).all()
  assert np.allclose(d['mean'],baseline['mean']) and np.allclose(d['std'],baseline['std'])
  pr,re,f1,support=precision_recall_fscore_support(y,p,labels=range(6),zero_division=0)
  assert abs(np.mean(y==p)-r.accuracy)<1e-12 and abs(f1.mean()-r.macro_f1)<1e-12
  assert np.allclose(re,[getattr(r,'recall_'+a) for a in ACT],atol=1e-12,rtol=0)
  for user in groups[r.fold]:
   entry=users.query('fold == @r.fold and seed == @r.seed and user == @user')
   assert len(entry)==1 and entry.iloc[0].windows==sum(u==user)
   assert abs(entry.iloc[0].accuracy-np.mean(p[u==user]==y[u==user]))<1e-12
  h=pd.read_csv(str(prefix)+'_history.csv');histories[r.fold,r.seed]=h
  assert len(h)==r.epochs and int(h.val_macro_f1.argmax())+1==r.best_epoch
  assert abs(h.val_macro_f1.max()-r.macro_f1)<1e-10
  assert Path(str(prefix)+'.weights.h5').stat().st_size>0
  metrics.append([r.fold,r.seed,r.accuracy,pr.mean(),re.mean(),f1.mean(),np.average(f1,weights=support)])
  for i,a in enumerate(ACT):classes.append([r.fold,r.seed,a,pr[i],re[i],f1[i],int(support[i])])
  cm+=confusion_matrix(y,p,labels=range(6))
  if r.seed==0:counts+=support
 assert counts.sum()==6430 and cm.sum()==19290
 details=pd.DataFrame(metrics,columns=['fold','seed','Accuracy','Macro Precision','Macro Recall','Macro F1','Weighted F1'])
 classdf=pd.DataFrame(classes,columns=['fold','seed','activity','precision','recall','f1','support'])
 fold=details.groupby('fold').mean().drop(columns='seed');mean=fold.mean();std=fold.std()
 summary=users.groupby('user').agg(accuracy=('accuracy','mean'),seed_std=('accuracy','std'),windows=('windows','first'))
 details.to_csv(OUT/f'{name}_classification_runs.csv',index=False)
 classdf.to_csv(OUT/f'{name}_classification_by_class.csv',index=False)
 summary.to_csv(OUT/f'{name}_users_summary.csv')
 pd.DataFrame(cm,index=ACT,columns=ACT).to_csv(OUT/f'{name}_confusion_counts.csv')
 figdir=OUT/'figures';figdir.mkdir(exist_ok=True)
 norm=cm/cm.sum(axis=1,keepdims=True)*100
 fig,ax=plt.subplots(figsize=(9,7),layout='constrained');im=ax.imshow(norm,vmin=0,vmax=100,cmap='Blues')
 for i in range(6):
  for j in range(6):ax.text(j,i,f'{norm[i,j]:.1f}%\n({cm[i,j]})',ha='center',va='center',color='white' if norm[i,j]>60 else 'black',fontsize=9)
 ax.set(xticks=range(6),yticks=range(6),xticklabels=ACT,yticklabels=ACT,xlabel='Predicted',ylabel='True',title=f'{name}: 3 seeds pooled; 6,430 unique windows / 19,290 predictions')
 plt.setp(ax.get_xticklabels(),rotation=30,ha='right');fig.colorbar(im,ax=ax,label='Row percentage')
 fig.savefig(figdir/f'{name}_confusion.png',dpi=150);plt.close(fig)
 for kind in ['loss','f1']:
  fig,axes=plt.subplots(5,3,figsize=(12,15),layout='constrained')
  for (f,s),h in histories.items():
   ax=axes[f,s];x=np.arange(1,len(h)+1)
   if kind=='loss':ax.plot(x,h.loss,label='Train');ax.plot(x,h.val_loss,label='Validation')
   else:ax.plot(x,h.val_macro_f1,label='Validation Macro F1');ax.set_ylim(0,1)
   best=int(runs.query('fold == @f and seed == @s').iloc[0].best_epoch)
   ax.axvline(best,ls='--',color='gray',label=f'Selected {best}');ax.set(title=f'Fold {f}, seed {s}',xlabel='Epoch',ylabel=kind);ax.legend(fontsize=7);ax.grid(alpha=.2)
  fig.suptitle(f'{name}: all 15 training runs');fig.savefig(figdir/f'{name}_{kind}_curves.png',dpi=130);plt.close(fig)
 comparisons=[];comparison_text=[]
 for other in comparators:
  ref=pd.read_csv(ROOT/f'runs/{other}_runs.csv');assert len(ref)==15 and set(zip(ref.fold,ref.seed))==expected
  base=ref.groupby('fold').macro_f1.mean();delta=fold['Macro F1']-base
  passed=delta.mean()>=.005 and sum(delta>0)>=4
  worse=delta.mean()<=-.005 and sum(delta<0)>=4
  verdict='개선' if passed else ('악화' if worse else '사전 기준으로 구별 불가')
  for f in range(5):comparisons.append([other,f,f'{base[f]:.4f}',f'{fold.loc[f,"Macro F1"]:.4f}',f'{delta[f]*100:+.2f}'])
  refusers=pd.read_csv(ROOT/f'runs/{other}_users.csv').groupby('user').accuracy.mean()
  du=summary.accuracy-refusers
  comparison_text.append(f'{other} 대비 평균 {delta.mean()*100:+.2f}%p, {sum(delta>0)}/5 fold 상승: **{verdict}**. 사용자별 평균 정확도는 {sum(du>1e-12)}명 상승, {sum(du < -1e-12)}명 하락, {sum(abs(du)<=1e-12)}명 동일하다.')
 classmean=classdf.groupby('activity')[['precision','recall','f1']].mean().reindex(ACT)
 lowest=classmean.f1.idxmin();bestclass=classmean.f1.idxmax()
 main_reference={'A1':'A0','A2':'A0','B1':'A2','B2':'B1','B3':'B1','B4':'B1','B3-C':'B3'}[name]
 ref_runs=pd.read_csv(ROOT/f'runs/{main_reference}_runs.csv')
 recall_change=pd.Series({activity:100*(classmean.loc[activity,'recall']-ref_runs['recall_'+activity].mean()) for activity in ACT})
 current_selection=runs.best_epoch.median()
 ref_selection=ref_runs.best_epoch.median()
 curve_notes=[]
 for trial in runs.itertuples():
  selected=histories[trial.fold,trial.seed].iloc[trial.best_epoch-1]
  curve_notes.append([selected.loss,selected.val_loss])
 train_loss,val_loss=np.median(curve_notes,axis=0)
 errors=sorted([(cm[i,j],ACT[i],ACT[j]) for i in range(6) for j in range(6) if i!=j],reverse=True)[:5]
 low=summary.sort_values('accuracy').head(6)
 notebook=next(ROOT.glob(f'{number}_{name}_*.ipynb'))
 filename=f'{number}_{name}_연구보고서.md'
 report=f'''# WISDM {name} 연구보고서 — {structure}

## 1 연구 질문과 결과

{question}를 확인한다. 개발 사용자 독립 검증 Macro F1은 **{mean['Macro F1']:.4f} ± {std['Macro F1']:.4f}**, Accuracy는 **{mean['Accuracy']:.4f}**다. 최종 테스트 성능이 아니다.

'''+ '\n\n'.join(comparison_text)+f'''

## 2 데이터와 평가 설계

WISDM v1.1 raw의 36명, 6개 행동, 약 20 Hz 3축 가속도 데이터다. timestamp 0·완전 중복·연속 동일 사용자/행동/시각의 뒤 레코드를 제외하고, 사용자/행동 변경·파싱 실패·시간 역행·1초 초과 공백에서 구간을 끊는다. 128시점 미만 구간은 제외한다. 정제 후 424개 구간, 비중첩 윈도우 7,841개다.

테스트 사용자 {protocol['test_users']}의 1,411개 윈도우를 보류했다. 개발 30명에서 매 fold 학습 24명·검증 6명을 분리한다. 입력은 128×3, 학습 stride 64·검증 stride 128이다. 표준화 통계는 해당 fold 학습 사용자에게서만 계산한다. 6,430개 고유 검증 윈도우를 seed별로 반복하므로 총 예측 19,290건은 독립 표본 19,290개가 아니다.

'''+table(['fold','검증 사용자','검증 윈도우'],[[f,', '.join(map(str,groups[f])),int(users.query('fold == @f and seed == 0').windows.sum())] for f in range(5)])+f'''
## 3 모델과 학습

구조는 **{structure}**이며 총 파라미터는 **{int(runs.params.iloc[0]):,}개**다. {LESSON[name]} [모델 구현]({"../"+notebook.name})

Adam(0.001), sparse categorical cross entropy, batch 64, Dropout 0.4, 최대 100 epoch, 검증 Macro F1 patience 12로 학습하고 최고 epoch 가중치를 복원했다. TensorFlow {protocol['tensorflow']}, NumPy {protocol['numpy']}, Colab T4 GPU를 사용했다. 5-fold × seed 0·1·2의 독립 학습 15회다.

model.fit 시간 합계 {runs.train_sec.sum():.1f}초, 평균 {runs.train_sec.mean():.1f}초다. 데이터 전송·정제·저장 시간은 제외한다. 선택 epoch 범위는 {int(runs.best_epoch.min())}~{int(runs.best_epoch.max())}, 종료 epoch 범위는 {int(runs.epochs.min())}~{int(runs.epochs.max())}다.

![전체 학습·검증 loss](figures/{name}_loss_curves.png)

![전체 검증 Macro F1](figures/{name}_f1_curves.png)

점선은 검증 Macro F1이 가장 높아 가중치를 보존한 epoch다. 선택 epoch의 중앙값은 {name}에서 {current_selection:.0f}회, 비교 대상 {main_reference}에서 {ref_selection:.0f}회였다. {name}의 선택 시점에서 15회 훈련 loss의 중앙값은 {train_loss:.3f}, 검증 loss의 중앙값은 {val_loss:.3f}였다. Loss는 실제 행동에 얼마나 높은 확률을 줬는지 반영한다. 맞힌 행동의 개수가 같아도 틀린 행동에 확신이 높으면 loss가 클 수 있다.

## 4 전체 분류지표

seed 3개를 fold 안에서 평균하고, fold 5개의 평균과 표본 표준편차를 제시한다. 표준편차는 신뢰구간이 아니다. Precision은 예측 행동의 정밀도, Recall은 실제 행동의 검출률, F1은 두 값의 조화평균이다. Macro는 행동별 동일 가중치, Weighted F1은 실제 표본 수 가중치다.

'''+table(['지표','fold 평균','fold 표준편차'],[[k,f'{mean[k]:.4f}',f'{std[k]:.4f}'] for k in mean.index])+f'''
## 5 행동별 지표와 혼동행렬

Precision·Recall·F1은 15회 지표 평균이며 표본 수는 seed 중복을 제외한 고유 윈도우다. 평균 F1이 가장 낮은 행동은 {lowest}({classmean.loc[lowest,'f1']:.4f}), 가장 높은 행동은 {bestclass}({classmean.loc[bestclass,'f1']:.4f})다.

'''+table(['행동','Precision','Recall','F1','고유 윈도우'],[[a,*[f'{classmean.loc[a,k]:.4f}' for k in ['precision','recall','f1']],counts[i]] for i,a in enumerate(ACT)])+f'''
![혼동행렬](figures/{name}_confusion.png)

행은 실제, 열은 예측이다. 비율은 행 기준이며 괄호는 seed 3개 합산 건수다. 행동별 평균 Recall과 합산 행 비율은 집계 방식이 다르다. 아래 오류 방향의 원인을 센서 방향·걸음걸이 등 특정 요인으로 단정하지 않는다.

'''+table(['실제','예측','오류 건수'],[[a,b,int(n)] for n,a,b in errors])+f'''
## 6 fold·seed·사용자 분석

'''+table(['fold','Accuracy','Macro F1','seed F1 표준편차'],[[f,f'{fold.loc[f,"Accuracy"]:.4f}',f'{fold.loc[f,"Macro F1"]:.4f}',f'{runs.groupby("fold").macro_f1.std()[f]:.4f}'] for f in range(5)])+f'''
사용자 동일 가중치 평균 Accuracy는 {summary.accuracy.mean():.4f}다. 아래는 낮은 6명이며, 전체 30명은 [{name}_users_summary.csv]({name}_users_summary.csv)에 있다. 낮은 점수 때문에 사용자를 제외하지 않는다.

'''+table(['사용자','윈도우','Accuracy','seed 표준편차'],[[u,int(r.windows),f'{r.accuracy:.4f}',f'{r.seed_std:.4f}'] for u,r in low.iterrows()])+'''
## 7 구조 비교와 판정

'''+table(['비교 대상','fold','참조 F1','현재 F1','차이 %p'],comparisons)+'\n\n'+'\n\n'.join(comparison_text)+'''

평균 +0.5%p 및 4/5 fold 이상 상승은 사전 실용 규칙이며 통계적 유의성을 뜻하지 않는다. 구조 비교에서 파라미터 수와 수용 영역이 달라질 수 있다. CBAM과 SE는 채널 계산도 달라 둘의 차이가 시간 attention만의 효과는 아니다.

## 8 한계와 다음 단계

검증 fold를 조기 종료와 구조 비교에 함께 사용하므로 선택 편향이 있을 수 있다. fold의 학습 사용자들이 겹치고 seed도 같은 사용자 반복이어서 15개 독립 표본으로 검정하지 않는다. 전체 데이터 품질·라벨 분포를 살핀 뒤 분할했으나 보류 테스트 성능은 확인하지 않았다. 다른 기기·휴대 위치·실생활 동작 전환으로의 일반화는 검증하지 않았다.

다음 구조에서도 동일 분할과 설정을 유지해 사전 비교를 수행한다. 필수 7개 구조가 완료되면 Q1·Q2·Q3을 종합하고, CPU 추론시간과 경량화 판단을 보완한다. 파라미터 감소만으로 속도·전력 개선을 주장하지 않는다. 테스트를 보기 전에 후보와 최종 학습 epoch를 고정한다.

## 9 전체 실행 기록

'''+table(['fold','seed','Accuracy','Macro F1','선택 epoch','종료 epoch','학습 초'],[[r.fold,r.seed,f'{r.accuracy:.4f}',f'{r.macro_f1:.4f}',r.best_epoch,r.epochs,r.train_sec] for r in runs.itertuples()])+f'''
## 10 재현 자료

- [연구계획서](../연구계획서.md), [실행계획](../실행계획_1단계.md), [데이터 출처 설명](../artifacts/source_descriptions.json)
- [실행별 기록](../runs/{name}_runs.csv), [사용자별 기록](../runs/{name}_users.csv), [프로토콜·데이터 해시](../runs/{name}_progress.json)
- [전체 지표]({name}_classification_runs.csv), [행동별 지표]({name}_classification_by_class.csv), [혼동행렬 건수]({name}_confusion_counts.csv)
- 실행별 예측·표준화 값은 `runs/{name}_fold{{fold}}_seed{{seed}}_predictions.npz`, 학습 곡선은 같은 접두사의 `_history.csv`다. 가중치는 PC와 Drive에 보존한다.
- 보고서 계산 코드는 [report_cnn.py](../scripts/report_cnn.py)다. 저장 지표, 사용자별 정확도, 학습 곡선의 최적 epoch, A0와 동일 검증 표본 및 표준화 값을 검사한 후 작성한다.
'''
 if name=='B4':
  base=pd.read_csv(ROOT/'runs/B1_runs.csv');loss=base.groupby('fold').macro_f1.mean().mean()-mean['Macro F1']
  report+=f'\nB4 경량화 추가 판정: 파라미터 {int(base.params.iloc[0]):,} → {int(runs.params.iloc[0]):,}, 감소율 {(1-runs.params.iloc[0]/base.params.iloc[0])*100:.2f}%. 평균 F1 손실 {loss*100:+.2f}%p로 사전 허용폭 1.0%p를 '+('충족한다.' if loss<=.01 else '초과한다.')+' 실제 추론 속도는 별도 측정 전이다.\n'
 if name=='B3-C':
  time_attention=ref_runs.groupby('fold').macro_f1.mean()-fold['Macro F1']
  gains=int((time_attention>0).sum())
  losses=int((time_attention<0).sum())
  delta=time_attention.mean()
  decision='개선' if delta>=.005 and gains>=4 else ('악화' if delta<=-.005 and losses>=4 else '구별 불가')
  effect=f'시간 attention 추가(B3 − B3-C)의 평균 Macro F1 차이는 {delta*100:+.2f}%p, 상승은 {gains}/5 fold로 사전 판정은 **{decision}**이다.'
  report=report.replace('## 8 한계와 다음 단계',effect+'\n\n## 8 한계와 다음 단계')
  report=report.replace('다음 구조에서도 동일 분할과 설정을 유지해 사전 비교를 수행한다. 필수 7개 구조가 완료되면 Q1·Q2·Q3을 종합하고, CPU 추론시간과 경량화 판단을 보완한다.',
                        '필수 7개 구조의 Q1·Q2·Q3 판정과 함께 이 추가 실험을 보고하고 CPU 추론시간으로 효율을 해석한다.')
  comparison_text.insert(0,effect)
 recall_table=table(['행동',f'{main_reference} Recall',f'{name} Recall','차이 (%p)'],[
   [activity,f'{ref_runs["recall_"+activity].mean():.4f}',f'{classmean.loc[activity,"recall"]:.4f}',f'{recall_change[activity]:+.2f}']
   for activity in ACT])
 report=report.replace('## 6 fold·seed·사용자 분석',f'**구조 변경 뒤 어떤 행동을 더 자주 찾아냈는가?** 아래 값은 같은 검증 사용자에서 얻은 15회 Recall의 평균이다. 차이가 가장 큰 행동은 {recall_change.idxmax()} ({recall_change.max():+.2f}%p), 가장 작은 행동은 {recall_change.idxmin()} ({recall_change.min():+.2f}%p)다.\n\n'+recall_table+'\n## 6 fold·seed·사용자 분석')
 (OUT/filename).write_text(report,encoding='utf-8')
 doc=json.loads(notebook.read_text(encoding='utf-8'))
 doc['cells'][-1]['source']=[f'## 결과 해석\n\nMacro F1 {mean["Macro F1"]:.4f} ± {std["Macro F1"]:.4f}.\n\n'+'\n\n'.join(comparison_text)+f'\n\n[별도 연구보고서](reports/{filename})\n']
 notebook.write_text(json.dumps(doc,ensure_ascii=False,indent=1)+'\n',encoding='utf-8')
 index=OUT/'README.md';text=index.read_text(encoding='utf-8')
 lines=text.splitlines()
 index_line=f'| {name} {structure} | [{name} 연구보고서]({filename}) | 15회 완료 · 지표 검증 완료 |'
 for i,line in enumerate(lines):
  if line.startswith(f'| {name} '):
   lines[i]=index_line
   break
 else:
  if name=='B3-C':
   lines.insert(next(i for i,line in enumerate(lines) if line.startswith('| B4 '))+1,index_line)
 index.write_text('\n'.join(lines)+'\n',encoding='utf-8')
 print(name,'verified 15/15',mean.to_dict(),flush=True)

if __name__=='__main__':main(sys.argv[1])
