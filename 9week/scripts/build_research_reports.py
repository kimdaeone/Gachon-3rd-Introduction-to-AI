from pathlib import Path
import json,io,hashlib,shutil,platform
import numpy as np,pandas as pd
from sklearn.metrics import precision_recall_fscore_support,confusion_matrix,accuracy_score,f1_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

root=Path('9week');out=root/'reports';figdir=out/'figures';figdir.mkdir(parents=True,exist_ok=True)
ACT=['Walking','Jogging','Upstairs','Downstairs','Sitting','Standing']
KO=['걷기','조깅','계단 오르기','계단 내려가기','앉기','서기']
GROUPS=[[1,7,20,22,25,36],[9,16,19,26,28,33],[2,3,4,14,30,31],[8,10,11,12,13,15],[5,17,18,23,32,35]]
TEST={6,21,24,27,29,34}
expected={(f,s) for f in range(5) for s in range(3)}
models={}
for model in ['A0','RF']:
 runs=pd.read_csv(root/f'runs/{model}_runs.csv');users=pd.read_csv(root/f'runs/{model}_users.csv')
 assert len(runs)==15 and set(zip(runs.fold,runs.seed))==expected
 assert len(users)==90 and not users.duplicated(['fold','seed','user']).any()
 detailed=[];classes=[];cm=np.zeros((6,6),int);truth=[];prediction=[];byuser={};counts=np.zeros(6,int)
 for r in runs.itertuples():
  d=np.load(root/f'runs/{model}_fold{r.fold}_seed{r.seed}_predictions.npz');y,p,u=d['y_true'],d['y_pred'],d['users']
  assert set(u)==set(GROUPS[r.fold]) and not set(u)&TEST
  precision,recall,f1,support=precision_recall_fscore_support(y,p,labels=range(6),zero_division=0)
  assert abs(accuracy_score(y,p)-r.accuracy)<1e-12 and abs(f1.mean()-r.macro_f1)<1e-12
  assert np.allclose(recall,[getattr(r,'recall_'+a) for a in ACT],atol=1e-12,rtol=0)
  for k,a in enumerate(ACT):classes.append(dict(fold=r.fold,seed=r.seed,activity=a,precision=precision[k],recall=recall[k],f1=f1[k],support=int(support[k])))
  detailed.append(dict(fold=r.fold,seed=r.seed,accuracy=r.accuracy,macro_precision=precision.mean(),macro_recall=recall.mean(),macro_f1=f1.mean(),weighted_f1=np.average(f1,weights=support)))
  cm+=confusion_matrix(y,p,labels=range(6));truth.extend(y);prediction.extend(p)
  if r.seed==0:counts+=support
  for user in GROUPS[r.fold]:
   mask=u==user
   entry=byuser.setdefault(user,dict(y=[],pred=[]));entry['y'].extend(y[mask]);entry['pred'].extend(p[mask])
 details=pd.DataFrame(detailed);classframe=pd.DataFrame(classes)
 fold=details.groupby('fold').mean(numeric_only=True).drop(columns='seed')
 user_summary=users.groupby('user').agg(accuracy=('accuracy','mean'),seed_std=('accuracy','std'),windows=('windows','first'),fold=('fold','first'))
 data=dict(runs=runs,users=users,details=details,classes=classframe,cm=cm,fold=fold,mean=fold.mean(),std=fold.std(),user=user_summary,byuser=byuser,counts=counts,pooled_accuracy=accuracy_score(truth,prediction),pooled_f1=f1_score(truth,prediction,labels=range(6),average='macro',zero_division=0))
 models[model]=data
 details.to_csv(out/f'{model}_classification_runs.csv',index=False)
 classframe.to_csv(out/f'{model}_classification_by_class.csv',index=False)
 user_summary.to_csv(out/f'{model}_users_summary.csv')
 pd.DataFrame(cm,index=ACT,columns=ACT).to_csv(out/f'{model}_confusion_counts.csv')
 normalized=cm/cm.sum(axis=1,keepdims=True)*100
 fig,ax=plt.subplots(figsize=(8.4,6.6),layout='constrained');im=ax.imshow(normalized,vmin=0,vmax=100,cmap='Blues')
 for i in range(6):
  for j in range(6):ax.text(j,i,f'{normalized[i,j]:.1f}%\n({cm[i,j]:,})',ha='center',va='center',fontsize=9,color='white' if normalized[i,j]>60 else '#172033')
 ax.set(xticks=range(6),yticks=range(6),xticklabels=ACT,yticklabels=ACT,xlabel='Predicted activity',ylabel='True activity',title=f'{model} development confusion matrix\n3 seeds pooled; 6,430 unique windows, 19,290 predictions')
 plt.setp(ax.get_xticklabels(),rotation=30,ha='right');fig.colorbar(im,ax=ax,label='Row percentage')
 fig.savefig(figdir/f'{model}_confusion.png',dpi=170);plt.close(fig)

assert np.array_equal(models['A0']['counts'],models['RF']['counts'])
# Confirm the two models evaluate exactly the same windows in the same order.
for fold,seed in expected:
 a=np.load(root/f'runs/A0_fold{fold}_seed{seed}_predictions.npz');r=np.load(root/f'runs/RF_fold{fold}_seed{seed}_predictions.npz')
 assert np.array_equal(a['y_true'],r['y_true']) and np.array_equal(a['users'],r['users'])

fig,axes=plt.subplots(2,3,figsize=(12.5,7),layout='constrained')
for col,fold in enumerate([0,2,3]):
 h=pd.read_csv(root/f'runs/A0_fold{fold}_seed0_history.csv');epochs=np.arange(1,len(h)+1);run=models['A0']['runs'].query('fold == @fold and seed == 0').iloc[0]
 axes[0,col].plot(epochs,h.loss,label='Train',color='#2563eb');axes[0,col].plot(epochs,h.val_loss,label='Validation',color='#e11d48')
 axes[0,col].set(title=f'Fold {fold}, seed 0',ylabel='Cross-entropy loss');axes[0,col].legend()
 axes[1,col].plot(epochs,h.val_macro_f1,color='#2563eb',label='Validation Macro F1');axes[1,col].axvline(run.best_epoch,color='#b45309',ls='--',label=f'Chosen epoch {int(run.best_epoch)}')
 axes[1,col].set(xlabel='Epoch',ylabel='Macro F1',ylim=(0,1));axes[1,col].legend(fontsize=8)
 for row in range(2):axes[row,col].grid(alpha=.2)
fig.suptitle('A0 training trajectories: selected folds, fixed seed 0',fontsize=14)
fig.savefig(figdir/'A0_learning_curves.png',dpi=170);plt.close(fig)

order=models['A0']['user'].accuracy.sort_values().index
fig,ax=plt.subplots(figsize=(8.4,10),layout='constrained');pos=np.arange(30)
for model,color in [('RF','#64748b'),('A0','#2563eb')]:ax.scatter(models[model]['user'].loc[order,'accuracy'],pos,color=color,label=model,s=28,zorder=3)
for i,u in enumerate(order):ax.plot([models['RF']['user'].loc[u,'accuracy'],models['A0']['user'].loc[u,'accuracy']],[i,i],color='#cbd5e1',linewidth=1.2)
ax.set(yticks=pos,yticklabels=[f'User {u}  (n={int(models["A0"]["user"].loc[u,"windows"])})' for u in order],xlim=(0,1.03),xlabel='Validation accuracy (mean of 3 seeds)',title='All 30 development users\nSorted by A0 accuracy; n is unique validation windows')
ax.invert_yaxis();ax.grid(axis='x',alpha=.2);ax.legend(loc='lower left');fig.savefig(figdir/'users_comparison.png',dpi=170);plt.close(fig)

fig,axes=plt.subplots(1,2,figsize=(12.2,4.6),layout='constrained')
for model,color in [('RF','#64748b'),('A0','#2563eb')]:
 axes[0].plot(range(5),models[model]['fold'].macro_f1,'o-',label=model,color=color)
 y=models[model]['classes'].groupby('activity').recall.mean().reindex(ACT)
 axes[1].bar(np.arange(6)+(-.18 if model=='RF' else .18),y,width=.36,label=model,color=color)
axes[0].axhline(.9,ls='--',color='#b45309',label='0.90 reference');axes[0].set(xticks=range(5),xlabel='Subject fold',ylabel='Macro F1',ylim=(.6,1),title='Macro F1 by fold (3-seed mean)');axes[0].legend()
axes[1].set(xticks=range(6),xticklabels=ACT,ylabel='Recall',ylim=(0,1.05),title='Recall by activity (mean of 15 runs)');plt.setp(axes[1].get_xticklabels(),rotation=30,ha='right');axes[1].legend()
for ax in axes:ax.grid(axis='y',alpha=.2)
fig.savefig(figdir/'models_comparison.png',dpi=170);plt.close(fig)

# Report tables keep aggregation definitions explicit.
def table(headers,rows):
 return '| '+' | '.join(headers)+' |\n| '+' | '.join(['---']*len(headers))+' |\n'+'\n'.join('| '+' | '.join(str(x) for x in row)+' |' for row in rows)+'\n'
def metric_table(model):
 d=models[model];labels={'accuracy':'Accuracy','macro_precision':'Macro Precision','macro_recall':'Macro Recall','macro_f1':'Macro F1','weighted_f1':'Weighted F1'}
 return table(['지표','fold 평균','fold 표준편차'],[[labels[k],f'{d["mean"][k]:.4f}',f'{d["std"][k]:.4f}'] for k in labels])
def class_table(model):
 d=models[model];g=d['classes'].groupby('activity')[['precision','recall','f1']].mean().reindex(ACT)
 return table(['행동','Precision','Recall','F1','개발 윈도우 수'],[[f'{a} {KO[i]}',*[f'{g.loc[a,k]:.4f}' for k in ['precision','recall','f1']],f'{d["counts"][i]:,}'] for i,a in enumerate(ACT)])
def fold_table(model):
 d=models[model];f=d['fold'];sd=d['runs'].groupby('fold').macro_f1.std()
 return table(['fold','검증 사용자','Accuracy','Macro F1','seed 간 F1 표준편차'],[[i,', '.join(map(str,GROUPS[i])),f'{f.loc[i,"accuracy"]:.4f}',f'{f.loc[i,"macro_f1"]:.4f}',f'{sd[i]:.4f}'] for i in range(5)])
def user_table(model):
 d=models[model];low=d['user'].sort_values('accuracy').head(6)
 return table(['사용자','fold','윈도우 수','평균 Accuracy','seed 표준편차'],[[u,int(r.fold),int(r.windows),f'{r.accuracy:.4f}',f'{r.seed_std:.4f}'] for u,r in low.iterrows()])
def errors_table(model):
 cm=models[model]['cm'];total=int(cm.sum()-np.trace(cm));pairs=sorted([(int(cm[i,j]),i,j) for i in range(6) for j in range(6) if i!=j],reverse=True)[:5]
 return table(['실제 행동','예측 행동','오류 수','전체 오류 내 비중'],[[ACT[i],ACT[j],n,f'{100*n/total:.1f}%'] for n,i,j in pairs])
def run_table(model):
 d=models[model];rows=[]
 for r in d['runs'].itertuples():
  values=[r.fold,r.seed,f'{r.accuracy:.4f}',f'{r.macro_f1:.4f}',f'{r.train_sec:.1f}']
  if model=='A0':values.extend([r.best_epoch,r.epochs])
  rows.append(values)
 return table(['fold','seed','Accuracy','Macro F1','학습 초']+(['선택 epoch','종료 epoch'] if model=='A0' else []),rows)

rep=json.loads((root/'runs/RF_reproduction.json').read_text(encoding='utf-8'))
counts_by_fold={r['fold']:r for r in rep['runs'] if r['seed']==0}
split_table=table(['fold','학습 사용자 수','학습 윈도우 수','검증 사용자 수','검증 윈도우 수'],[[f,24,counts_by_fold[f]['train_windows'],6,counts_by_fold[f]['val_windows']] for f in range(5)])
class_counts=table(['행동','전체 정제 윈도우','개발 검증 윈도우'],[[a,n,int(models['A0']['counts'][i])] for i,(a,n) in enumerate(zip(ACT,[3146,2333,867,705,433,357]))])
common_data=f'''## 2 데이터와 평가 설계

WISDM v1.1 raw의 휴대폰 3축 가속도 시계열을 사용했다. 사용자 36명이 수행한 6개 행동을 분류하며, 샘플링 주파수는 약 20 Hz이다. 입력 윈도우는 128개 시점으로 약 6.4초에 해당한다. 실제 사용자의 이름 대신 공개 데이터의 숫자 ID를 사용한다.

정제는 timestamp 0 삭제, 완전히 같은 레코드 삭제, 연속된 동일 사용자·행동·시각의 뒤 레코드 제외 순서로 적용했다. 사용자·행동 변경, 파싱 실패, 시간 역행, 1초 초과 공백에서 연속 구간을 끊고 128개보다 짧은 구간은 제외했다. 현재 확정 코드의 결과는 **424개 구간, 겹치지 않는 윈도우 7,841개**이다. 원본 중복 66,820건과 timestamp 0인 12,845건은 서로 겹칠 수 있어 단순히 합산해 제거량으로 해석하지 않는다.

{class_counts}

테스트 사용자 **6, 21, 24, 27, 29, 34번**의 1,411개 윈도우는 최종 평가용으로 보류했다. 나머지 30명을 6명씩 5개 fold로 나누고, 매회 학습 24명과 검증 6명이 겹치지 않게 했다. 학습 stride는 64, 검증 stride는 128이다. 따라서 학습에는 절반 겹치는 윈도우가 있지만 검증에는 동일 구간 내 겹침이 없다. 서로 다른 fold의 학습 윈도우 수를 합해 고유 표본 수라고 보고하지 않는다.

{split_table}

개발 검증의 고유 윈도우는 총 **6,430개**이다. seed 0, 1, 2로 각각 반복하므로 총 검증 예측은 **19,290건**이지만, 이를 독립적인 데이터 19,290개로 해석하지 않는다. 17명은 일부 행동이 없고 Sitting·Standing 보유자는 각각 23명·24명으로, 사용자별 행동 구성과 표본 수의 차이가 남아 있다.
'''
metric_explanation='''### 지표를 읽는 방법

- Accuracy는 전체 윈도우 중 맞힌 비율이다.
- Precision은 특정 행동으로 예측한 것 중 실제로도 그 행동인 비율, Recall은 실제 그 행동 중 찾아낸 비율이다.
- F1은 Precision과 Recall의 조화평균이다. Macro F1은 6개 행동의 F1을 같은 비중으로 평균하고, Weighted F1은 각 행동의 실제 표본 수를 가중치로 사용한다. Macro F1을 추가 정답 개수나 Accuracy 차이로 환산하지 않는다.
- 주 결과는 각 fold 안에서 seed 3개를 먼저 평균한 뒤 fold 5개의 평균과 표준편차로 보고한다. 표준편차는 fold에 포함된 사람 조합에 따른 변동이며 신뢰구간이나 유의확률이 아니다.
- 행동별 표의 Precision·Recall·F1은 15회 지표의 평균이다. 마지막 열은 seed 반복을 제거한 개발 윈도우 수다. 혼동행렬은 15회 예측을 합산하므로 표와 집계 방식이 다르며, 행 비율이 행동별 표의 평균 Recall과 정확히 같지는 않다.
'''
refs='''## 10 근거 자료와 재현 방법

- [연구계획서](../연구계획서.md)와 [데이터 출처 설명](../artifacts/source_descriptions.json)
- 데이터 출처에서 요청한 인용 문헌은 Jennifer R. Kwapisz, Gary M. Weiss, Samuel A. Moore의 2010년 논문 *Activity Recognition using Cell Phone Accelerometers*이다. 문헌 서지는 데이터셋 README를 기준으로 기록했다.
- 지표 정의는 [scikit-learn classification_report](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.classification_report.html), 혼동행렬의 행과 열 정의는 [confusion_matrix](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix)를 따른다.
- 아래 연결된 원시 결과와 보고서용 상세 표에는 반올림 전 수치가 있다. 보고서의 4자리 표시는 가독성을 위한 반올림이다.
'''
a=models['A0'];rf=models['RF'];diff=(a['fold'].macro_f1-rf['fold'].macro_f1)*100
comparison_table=table(['fold','RF Macro F1','A0 Macro F1','차이 %p'],[[f,f'{rf["fold"].loc[f,"macro_f1"]:.4f}',f'{a["fold"].loc[f,"macro_f1"]:.4f}',f'{diff[f]:+.2f}'] for f in range(5)])

intro_a=f'''# WISDM A0 기본 CNN 연구보고서

작성일 2026년 11월 16일 · 기계학습프로그래밍 · 개발 사용자 독립 검증 결과

## 1 연구 목적과 주요 결론

원시 가속도 시계열을 입력으로 사용하는 기본 CNN을 학습하고, 이후 Residual·Inception·Attention·Depthwise 실험을 비교할 기준점을 확보했다. 정식 5-fold × 3-seed 결과는 **Macro F1 {a['mean'].macro_f1:.4f} ± {a['std'].macro_f1:.4f}, Accuracy {a['mean'].accuracy:.4f}**이다. RF 참조선보다 Macro F1이 평균 **{diff.mean():.2f}%p 높고 5개 fold 모두 개선**됐으나, 주 지표의 평균은 사전 참고선 0.90에 조금 못 미쳤다. 사용자 30·25·9번과 계단 동작에서 개선 여지가 크다. 이 보고서의 수치는 최종 테스트 성능이 아니다.

'''
a0=intro_a+common_data+'''## 3 모델 구성과 데이터 학습

입력은 `(윈도우 수, 128, 3)`이다. 각 fold의 학습 사용자 24명에게서만 축별 평균·표준편차를 계산하고, 그 값으로 학습·검증 데이터를 표준화했다. 검증 사용자나 테스트 사용자로 표준화 통계를 맞추지 않았다.

구조는 `Conv1D 64채널 → 두 개의 CNN 블록 → GlobalAveragePooling → Dropout → 6클래스 Softmax`이다. 각 블록은 kernel 5인 Conv1D 두 개와 BatchNormalization·ReLU를 사용하고 마지막에 MaxPool 2를 적용한다. A0에는 residual 더하기나 attention이 없다. 총 파라미터는 **84,870개**, 그중 학습 가능한 파라미터는 84,230개이다.

| 학습 항목 | 설정 |
|---|---|
| Optimizer | Adam, learning rate 0.001 |
| 손실함수 | Sparse categorical cross entropy |
| Batch size | 64 |
| 최대 epoch | 100 |
| Dropout | 0.4 |
| 조기 종료 | 검증 Macro F1, patience 12 |
| 가중치 선택 | 검증 Macro F1 최고 epoch의 가중치 복원 |
| 반복 | 5 fold × seed 0 1 2 |
| 실행 환경 | Colab T4 GPU, TensorFlow 2.21.0, NumPy 2.4.6 |

15회 모두 조기 종료됐고 종료 epoch는 17~53, 선택 epoch는 5~41이었다. 학습시간은 합계 **707.6초**, 평균 **47.2초**, 최소 27.6초·최대 76.5초다. 이 값은 `model.fit` 구간으로 데이터 읽기·전송·결과 저장 시간을 제외한다.

![A0 학습 곡선](figures/A0_learning_curves.png)

위 그림은 fold 0·2·3의 seed 0을 보여준다. 위쪽은 훈련·검증 loss, 아래쪽은 검증 Macro F1과 선택 epoch다. 훈련 loss가 낮아져도 검증 loss가 함께 감소하지 않는 구간이 나타난다. 이를 일반화의 한계로 보되, 검증 loss 증가와 Macro F1 하락이 반드시 동시에 발생하는 것은 아니다. epoch 선택 기준을 결과를 보고 val loss로 바꾸지 않았다. 전체 15회 곡선의 원자료는 보존했다.

## 4 전체 분류지표

'''+metric_explanation+'\n'+metric_table('A0')+f'''
Macro F1이 0.90 이상인 fold는 **3/5**다. Accuracy만 보고 목표에 도달했다고 결론내리지 않는다. 모든 개발 예측을 합친 보조 지표는 Accuracy {a['pooled_accuracy']:.4f}, Macro F1 {a['pooled_f1']:.4f}이며, 사람·표본 비중이 달라 주 결과인 fold 평균과 구분한다.

## 5 행동별 분류지표와 오류

'''+class_table('A0')+'''
Standing과 Jogging은 Recall이 높고, Upstairs·Downstairs는 상대적으로 낮다. 특히 Upstairs는 Precision도 낮아 다른 행동이 Upstairs로 잘못 들어오는 오류가 함께 존재한다.

![A0 혼동행렬](figures/A0_confusion.png)

행은 실제 행동, 열은 예측 행동이다. 각 칸의 위 숫자는 실제 행동 기준 비율이고 괄호는 seed 3개를 포함한 예측 건수다. 아래 표는 정답 대각선을 제외한 큰 오류 방향이다.

'''+errors_table('A0')+'''
걷기와 계단 동작의 혼동이 주요 오류다. 앉기와 서기의 구분만을 핵심 병목으로 간주했던 UCI HAR 분석을 WISDM에 그대로 적용하지 않는다. 이동 동작의 오류가 줄어드는지 후속 구조에서 확인해야 한다.

## 6 fold와 사용자에 따른 차이

'''+fold_table('A0')+'\n'+user_table('A0')+f'''
30명에게 같은 비중을 준 평균 Accuracy는 **{a['user'].accuracy.mean():.4f}**다. 윈도우 수가 다른 사용자들을 함께 계산한 fold 평균 Accuracy와 의미가 다르다. 30번은 Walking 예측 261건 중 236건을 Jogging으로 분류했다. 사용자 25번도 Walking을 거의 구분하지 못한다. 이는 확인된 오류 패턴이며 휴대폰 방향·걸음걸이·센서 이상 중 어느 요인 때문인지는 추가 분석 없이 단정하지 않는다.

![개발 사용자 전체의 정확도 비교](figures/users_comparison.png)

점은 seed 3개 평균, 괄호의 n은 반복을 제거한 검증 윈도우 수다. 낮은 사람을 제외하지 않고 유지한다. 후속 구조의 개선이 몇 사람에게 집중되는지, 다수 사용자에게 나타나는지 함께 본다.

## 7 RF 참조선과 비교

'''+comparison_table+'''
![모델별 fold 성능과 행동별 Recall](figures/models_comparison.png)

사전 규칙인 평균 +0.5%p 이상과 5개 fold 중 4개 이상 개선을 충족한다. 다만 RF는 47개 통계 특징을 사용하고 A0는 표준화한 원시 시계열을 사용하므로, 이 차이를 순수한 모델 구조 한 요소의 효과라고 해석하지 않는다. RF 특징에도 인접 차분의 평균 절댓값이 있어 시간 변화를 일부 반영하지만 상세 파형·순서를 그대로 보존하지는 않는다. 모든 RF가 CNN보다 낮다는 일반화도 할 수 없다.

## 8 한계와 다음 실험

검증 Macro F1은 조기 종료와 모델 비교에 모두 사용되므로 epoch 선택 편향이 포함될 수 있다. 5-fold는 학습 사용자가 서로 겹치며 seed 반복도 새로운 사람 표본이 아니다. 표준편차와 실용적인 개선 규칙을 통계적 유의성으로 해석하지 않는다. 전체 데이터의 품질과 라벨 보유 현황을 살핀 뒤 분할했으나 테스트 성능은 확인하지 않았다.

실제 CPU 추론시간·메모리·전력은 아직 측정하지 않아 효율 우열은 판단하지 않는다. RF의 PC 학습시간과 A0의 T4 학습시간을 직접 비교해 속도 우열을 주장하지 않는다. 또한 이 데이터셋의 성능만으로 다른 휴대 위치나 실생활 동작 전환에 대한 성능을 보장하지 않는다.

다음은 A1에서 residual 연결만 추가하고 동일한 분할·학습 설정으로 A0와 비교한다. 평균 Macro F1과 fold별 차이뿐 아니라 계단 Recall, 사용자 30·25·9번, 개선·악화된 사용자 수를 확인한다. 최종 후보 선정 후에만 보류한 테스트 6명을 사용한다.

## 9 실행별 상세 기록

'''+run_table('A0')+refs+'''
- [실행 결과가 포함된 A0 노트북](../03_A0_기본CNN_5fold.ipynb)
- [기존 실행별 결과](../runs/A0_runs.csv), [사람별 원자료](../runs/A0_users.csv), [실행 조건과 재개 기록](../runs/A0_progress.json)
- [추가 분류지표 원자료](A0_classification_runs.csv), [행동별 지표 원자료](A0_classification_by_class.csv), [사용자 전체 요약](A0_users_summary.csv), [혼동행렬 건수](A0_confusion_counts.csv)
- 예측·학습 곡선 파일은 `../runs/A0_fold{fold}_seed{seed}_predictions.npz`와 `../runs/A0_fold{fold}_seed{seed}_history.csv`다. 최적 가중치는 PC와 Drive에 보존하고 GitHub에는 업로드하지 않았다.

CLI 연결이 한 차례 끊긴 동안에도 원격 학습은 계속됐다. 재연결 후 완료 기록을 읽어 15회를 중복 학습하지 않고 요약을 실행했다. 15개 예측 파일에서 지표를 다시 계산해 저장된 Accuracy·Macro F1·행동별 Recall과 일치함을 검증했고 테스트 사용자 ID가 없음을 확인했다.
'''

rf_report=f'''# WISDM Random Forest 참조선 연구보고서

작성일 2026년 11월 16일 · 기계학습프로그래밍 · 개발 사용자 독립 검증 결과

## 1 연구 목적과 주요 결론

CNN의 비교 기준으로, 가속도 윈도우에서 통계 특징을 추출한 Random Forest를 평가했다. 동일한 사용자 분할에서 **Macro F1 {rf['mean'].macro_f1:.4f} ± {rf['std'].macro_f1:.4f}, Accuracy {rf['mean'].accuracy:.4f}**를 얻었다. Jogging·Standing은 비교적 잘 구분하지만 계단 두 행동의 Recall은 각각 약 0.52로 낮다. A0보다 모든 fold에서 Macro F1이 낮아, 이후 CNN 구성 요소 실험에서 외부 참조선 역할을 한다. 이 결과는 별도 논문의 RF 점수나 최종 테스트 성능과 직접 동일시하지 않는다.

'''+common_data+'''## 3 특징 추출과 데이터 학습

각 윈도우의 x·y·z축과 합성 가속도 크기로 4개 시계열을 구성한다. 각 시계열에서 평균, 표준편차, 평균 절대편차, 최솟값, 최댓값, 중앙값, 10·25·75·90 백분위수, 인접 차분의 평균 절댓값 등 11개 통계량을 얻는다. 44개 값에 xy·xz·yz의 상관계수 3개를 더해 **47개 특징**을 입력한다.

이 RF 구현은 축별 표준화를 적용하지 않고 원래 값에서 특징을 추출한다. A0와 원본 데이터·정제·윈도우·분할은 같지만 입력 표현과 스케일 처리는 다르다. 인접 차분 통계는 시간 변화의 일부를 반영하므로 RF가 시간 정보를 전혀 사용하지 않는다고 표현하지 않는다.

| 학습 항목 | 설정 |
|---|---|
| 분류기 | scikit-learn RandomForestClassifier |
| 트리 수 | 300 |
| 분할 기준 | Gini |
| max depth | 제한 없음 |
| max features | sqrt |
| bootstrap | 사용 |
| min samples split과 leaf | 2와 1 |
| class weight | 사용하지 않음 |
| 병렬 작업 | n_jobs 4 |
| random state | seed 0 1 2 |
| 실행 | 로컬 PC CPU |

RF는 CNN처럼 epoch·learning rate·역전파 loss로 학습하지 않는다. 고정된 설정으로 학습 표본에 300개의 트리를 적합한 후 검증했다. 별도의 하이퍼파라미터 탐색이나 검증점수 기반 조기 종료는 적용하지 않았다. 최초 정식 실행의 학습시간은 평균 **7.8초**, 합계 **116.8초**였다. 원래 훈련 정확도·트리 수별 학습 곡선은 저장하지 않았으므로 검증 점수만으로 과적합 정도를 확정하지 않는다.

상세 분류지표를 작성하기 위해 2026년 11월 16일 동일 코드·분할·seed·설정으로 15회를 재실행했다. 현재 복원 환경은 Python 3.14.2, scikit-learn 1.9.1이며, **기존 Accuracy·Macro F1·행동별 Recall·사람별 정확도가 모두 일치**했다. 최초 실행 파일은 덮어쓰지 않았다. 아래 Precision·행동별 F1·혼동행렬은 이 일치 검증을 통과한 복원 예측으로 계산했다. 집계 점수의 일치가 원래 개별 예측 하나하나의 동일성을 직접 증명하는 것은 아니다.

## 4 전체 분류지표

'''+metric_explanation+'\n'+metric_table('RF')+f'''
Macro F1이 0.90 이상인 fold는 **0/5**다. 전체 개발 예측을 합친 보조 지표는 Accuracy {rf['pooled_accuracy']:.4f}, Macro F1 {rf['pooled_f1']:.4f}다. 주 결과인 fold 평균과 섞어 비교하지 않는다.

## 5 행동별 분류지표와 오류

'''+class_table('RF')+'''
Jogging과 Standing의 높은 Recall만으로 전체 분류가 잘 된다고 판단할 수 없다. Upstairs와 Downstairs는 절반가량을 놓치며, 행동별 표본 수가 다른 상황에서 Accuracy보다 Macro F1을 함께 보는 이유가 드러난다.

![RF 혼동행렬](figures/RF_confusion.png)

행은 실제 행동, 열은 예측 행동이다. 비율은 행별로 계산했으며 괄호는 seed 3개 합산 예측 건수다. 아래는 가장 큰 오류 방향이다.

'''+errors_table('RF')+'''
오류가 반복되는 방향은 현재 특징 표현이 일부 이동 동작을 충분히 나누지 못함을 보여준다. 그러나 이 자료만으로 단일 원인이 시간 순서 손실이라고 단정하지 않는다. 특징 선택·모델 설정·사용자 간 신호 차이가 함께 영향을 줄 수 있다.

## 6 fold와 사용자에 따른 차이

'''+fold_table('RF')+'\n'+user_table('RF')+f'''
사용자별 같은 비중의 평균 Accuracy는 **{rf['user'].accuracy.mean():.4f}**다. 같은 fold 안에서 seed만 바꾼 편차보다 fold 사이의 Macro F1 표준편차가 더 크다. 점수가 특정 seed보다 검증 사용자 조합의 영향을 많이 받는다는 관측이며, 이것이 평가가 잘못됐다는 뜻은 아니다. 모델 간에는 같은 fold끼리 짝지어 비교한다.

![개발 사용자 전체 정확도 비교](figures/users_comparison.png)

사용자 25번의 어려움은 RF와 A0 모두에 나타난다. 어떤 사람을 제외하면 점수가 좋아진다는 이유로 제외하지 않는다. 원본 신호나 라벨에 명확한 오류가 확인되면 그때 별도의 정제 근거와 영향 분석을 남긴다.

## 7 A0 CNN과 비교

'''+comparison_table+f'''
A0는 RF보다 Macro F1이 평균 **{diff.mean():.2f}%p** 높고 모든 fold에서 개선됐다. 계단 동작의 Recall 개선도 확인된다. 그러나 RF의 특징 입력과 CNN의 원시 시계열 입력이 달라 순수한 알고리즘 하나의 효과로 분리할 수 없으며, RF는 검증을 통한 epoch 선택을 하지 않는다는 차이도 있다.

![모델별 성능 비교](figures/models_comparison.png)

RF는 동일 조건의 비교 참조선으로 유지한다. 현재 결과를 보고 RF 설정만 추가로 조정해 기존 결과와 섞지 않는다. RF 개선 실험을 한다면 별도 실험명과 사전 설정을 부여한다.

## 8 한계와 다음 활용

47개 통계 특징과 300개 트리라는 한 구현의 결과다. 더 풍부한 주파수 특징이나 다른 RF 설정의 상한을 뜻하지 않는다. 훈련 점수·학습 곡선을 저장하지 않아 훈련과 검증 사이의 차이를 직접 평가할 수 없고, 최초 실행의 상세 라이브러리 버전은 별도로 남아 있지 않다. 복원 환경과 최초 실행 환경을 같은 것으로 단정하지 않는다.

외부 논문에서 언급된 RF 0.87과는 WISDM 버전·사람 분할·지표가 같은지 확인되지 않았으므로 직접 우열을 비교하지 않는다. 최종 테스트 6명도 평가하지 않았다. CPU 추론시간과 메모리 사용량은 아직 측정하지 않았고, PC와 T4의 학습시간으로 효율 우열을 판단하지 않는다.

A1 이후 CNN 비교에서는 RF를 고정된 참조선으로 유지한다. 개발 결과로 최종 후보가 결정되면 RF도 개발 30명 전체로 다시 학습한 뒤 동일한 보류 테스트에 평가한다.

## 9 최초 정식 실행별 기록

'''+run_table('RF')+refs+'''
- [RF 학습 노트북](../03R_RandomForest_참조선_5fold.ipynb)
- [최초 실행별 결과](../runs/RF_runs.csv), [최초 사람별 결과](../runs/RF_users.csv), [복원 환경과 일치 검증 기록](../runs/RF_reproduction.json)
- [추가 분류지표 원자료](RF_classification_runs.csv), [행동별 지표 원자료](RF_classification_by_class.csv), [사용자 전체 요약](RF_users_summary.csv), [혼동행렬 건수](RF_confusion_counts.csv)
- 복원 예측은 `../runs/RF_fold{fold}_seed{seed}_predictions.npz`에 저장했다. 복원 학습시간은 위 최초 실행 시간과 분리해 `RF_reproduction.json`에 보존했다.
'''
rf_report=rf_report.replace('Python 3.14.2',f"Python {rep['python']}")
(out/'03_A0_연구보고서.md').write_text(a0,encoding='utf-8')
(out/'03R_RF_연구보고서.md').write_text(rf_report,encoding='utf-8')
index='''# WISDM 모델별 연구보고서

실험마다 학습 노트북과 별도로 연구보고서를 작성한다. 아래 보고서는 정식 실행 결과를 근거로 데이터 처리부터 분류지표와 결과 해석까지 정리한 문서다. 현재 모든 성능은 개발 사용자 독립 검증 결과이며 최종 테스트 결과가 아니다.

| 실험 | 연구보고서 | 상태 |
|---|---|---|
| A0 기본 CNN | [A0 연구보고서](03_A0_연구보고서.md) | 15회 완료 |
| RF 참조선 | [RF 연구보고서](03R_RF_연구보고서.md) | 15회 완료 및 상세 예측 복원 검증 |
| A1 ResNet | 학습 완료 후 작성 | 미실행 |
| A2 Inception | 학습 완료 후 작성 | 미실행 |
| B1 Inception ResNet | 학습 완료 후 작성 | 미실행 |
| B2 SE | 학습 완료 후 작성 | 미실행 |
| B3 CBAM | 학습 완료 후 작성 | 미실행 |
| B4 Depthwise | 학습 완료 후 작성 | 미실행 |

## 각 보고서에 포함할 내용

1. 연구 질문과 핵심 결론
2. 데이터 출처·정제·윈도우·사용자 분할과 표본 수
3. 입력 표현·모델 구조·학습 설정·실행 환경과 시간
4. Accuracy·Macro Precision·Macro Recall·Macro F1·Weighted F1 및 집계 방식
5. 행동별 Precision·Recall·F1·Support와 혼동행렬
6. 학습 곡선 또는 해당 모델에 적합한 학습 기록
7. fold·seed 변동과 사용자별 성능 및 오류 집중
8. 사전에 정한 비교 대상과의 차이 및 개선 판정
9. 한계·다음 실험·재현 가능한 파일 연결
10. 실행별 전체 기록과 상세 지표 원자료

평균과 표준편차, fold 평균과 예측 합산 지표를 구분한다. 혼동행렬에서 seed를 합산하면 3배 반복임을 명시한다. 학습하지 않은 모델의 결과를 예상값으로 채우지 않는다. 관측된 오류 패턴과 원인 가설을 구분하고, 개선이 없더라도 결과를 남긴다. 추론시간·메모리처럼 측정하지 않은 지표는 수치를 만들지 않는다.

## 보고서 작성 시점

각 모델의 15회 학습이 끝나면 예측값으로 저장 지표를 재검증하고 보고서를 작성한다. 노트북·보고서·그림·원자료를 Drive에 동기화한 뒤 모델별로 커밋·푸시한다. 모델 변경이 없는 분석 보강은 학습 결과와 구분해 기록한다.
'''
(out/'README.md').write_text(index,encoding='utf-8')
summary={model:{'mean':d['mean'].to_dict(),'std':d['std'].to_dict(),'class_mean':d['classes'].groupby('activity')[['precision','recall','f1']].mean().to_dict(),'pooled_accuracy':d['pooled_accuracy'],'pooled_f1':d['pooled_f1']} for model,d in models.items()}
(out/'report_metrics.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
print('REPORTS GENERATED')
for model,d in models.items(): print(model,d['mean'].round(4).to_dict(), 'unique',int(d['counts'].sum()),'predictions',int(d['cm'].sum()))
print('Original RF learning seconds:',models['RF']['runs'].train_sec.sum())
