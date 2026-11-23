# 3. Xception / Depthwise Separable Convolution

[FAQ 목차로 돌아가기](../README.md)

## 필터를 분리하면 줄어드는 계산

```mermaid
flowchart LR
 X[입력 C개 채널] --> D[채널별 시간 필터]
 D --> P[같은 시점의 채널 혼합]
 P --> Y[출력 M개 채널]
```

시간 필터 길이가 $k$, 입력 채널이 $C$, 출력 채널이 $M$일 때 편향을 뺀 가중치 수는 일반 합성곱에서 $kCM$, 분리 합성곱에서 $kC+CM$이다(depth multiplier 1). WISDM 분기 하나에 $k=15$, $C=M=16$을 대입하면 3,840개에서 496개로 줄어든다.

### 1. Xception은 어떤 모델을 기반으로 발전한 구조인가?

Inception의 아이디어에서 출발했다. Xception은 공간 패턴을 찾는 일과 채널 사이의 관계를 찾는 일을 얼마나 더 분리할 수 있는지 탐구한 구조이다.

### 2. Xception의 핵심 아이디어는 무엇인가?

공간 상관관계와 채널 간 상관관계를 서로 다른 연산으로 모델링한다. 실제 Xception은 depthwise separable convolution을 쌓고 여러 블록에 residual 연결을 사용한다.

### 3. Depthwise separable convolution이란 무엇인가?

먼저 각 입력 채널에서 독립적으로 공간 필터를 적용하고, 그 결과를 1×1 필터로 조합하는 합성곱이다. 1D 신호에서는 공간 필터가 시간축을 따라 이동한다.

### 4. 일반 convolution과 depthwise separable convolution의 차이는 무엇인가?

일반 합성곱은 각 출력 채널마다 공간 위치와 입력 채널의 조합 전체에 가중치를 둔다. 분리 합성곱은 공간 연산과 채널 조합을 두 단계로 묶어 가중치 구조에 제약을 둔다.

### 5. Depthwise convolution은 어떤 방식으로 수행되는가?

depth multiplier가 1이면 입력 채널마다 전용 필터 하나를 적용한다. RGB에 적용하면 R·G·B가 각자의 필터를 거쳐 세 개의 출력 맵이 된다.

### 6. Depthwise convolution에서는 channel 간 상호작용이 일어나는가?

기본 depthwise 단계 안에서 각 출력은 대응하는 입력 채널 하나에만 의존한다. 서로 다른 채널의 값을 합치는 작업은 뒤의 pointwise 단계에서 맡는다.

### 7. Pointwise convolution이란 무엇인가?

한 공간 위치에 있는 모든 입력 채널을 가중합하는 1×1 합성곱이다. 같은 가중치를 모든 위치에서 공유하고 출력 필터 수만큼 새로운 채널을 만든다.

### 8. (1 × 1) pointwise convolution은 depthwise convolution 이후 어떤 역할을 하는가?

채널별로 추출한 공간 반응을 함께 보면서 새 특징을 만든다. 예를 들어 각 축의 시간 변화가 계산된 뒤에는 세 축의 변화를 조합한 패턴을 표현할 수 있다.

### 9. Depthwise convolution과 pointwise convolution을 분리하면 어떤 장점이 있는가?

공간 필터를 출력 채널마다 새로 만들 필요가 줄어 파라미터와 곱셈을 절약한다. 절약한 예산으로 다른 층의 폭이나 깊이를 늘리는 설계도 가능하다.

### 10. Depthwise separable convolution이 연산량을 줄이는 이유는 무엇인가?

편향 제외, depth multiplier 1에서 k×k 일반 합성곱은 위치당 $k^2 C_{in} C_{out}$번 곱한다. 분리하면 $k^2 C_{in}+C_{in} C_{out}$번이다. k=3, 입출력 32채널이면 9,216번이 1,312번이 된다.

### 11. Xception은 Inception 구조를 어떻게 극단적으로 해석한 모델인가?

Xception 논문의 사고실험은 먼저 1×1 합성곱으로 채널을 조합한 뒤, 채널 그룹마다 공간 필터를 적용하는 단순화된 Inception에서 출발한다. 그룹을 채널 하나까지 좁히면 채널별 공간 필터가 된다. 실제 Xception의 분리 합성곱은 공간 필터를 먼저 적용하고 1×1 합성곱으로 채널을 섞는다.

### 12. Xception에서 중간 ReLU 비선형성을 제거한 이유는 무엇인가?

두 연산 사이에 ReLU를 두면 첫 단계의 음수 반응이 사라져 두 번째 단계에서 조합할 정보가 줄 수 있다. Xception 논문의 해당 비교에서 중간 활성화를 생략한 쪽이 더 좋았다. 다른 블록 위치의 ReLU까지 모두 제거한 것은 아니다.

### 13. Depthwise separable convolution이 모바일/경량 모델에서 자주 사용되는 이유는 무엇인가?

작은 저장공간과 제한된 연산 예산으로 공간 특징을 뽑을 수 있기 때문이다. 지연시간은 메모리 접근과 실행 라이브러리에도 좌우되므로 실제 배포 장치에서 확인해야 한다.

### 14. Depthwise separable convolution의 단점은 무엇일 수 있는가?

가중치를 분해하면 일반 합성곱 하나가 표현할 수 있는 모든 필터를 한 블록에서 자유롭게 표현하기 어렵다. 우리 B4는 B1보다 파라미터가 56.9% 적었지만 PC의 CPU 추론 중앙값은 0.643ms에서 0.635ms로 소폭 줄었다.

## 복습할 때 확인할 점

분기 하나에서 3,840개가 496개로 줄어도 모델 전체의 파라미터가 같은 비율로 줄지 않는 이유는 무엇일까?

[WISDM 실험의 구조별 비교](../../9week/reports/10_1단계_개발_검증_종합.md) · [구조 제안 논문](https://arxiv.org/abs/1610.02357)
