# 지표를 위한 함수

1. loss

	모델 최적화(error를 최소로 해주는 parameter 변화)를 위한 값을 도출하는 함수

	정확하게 맞출 확률의 하한여부(Evidence Lower Bound)?

	지금까지 사용한 함수
	- 회귀: MSE, Mean Squared Error
	- 분류: CE, [Cross Entropy](./entropy-review.md)

2. metric

	모델의 성능(performance 점수)을 평가하기 위한 값을 도출하는 함수

# Metric 종류

대부분의 지표는 거의 기존의 통계학에서 비롯된 것이 대부분

특수한 목적으로 만든 지표들도 많이 있으니 다양한 저널검색이 도움이 됌

키워드
- 회귀분석
- 분산분석

## 회귀
E(error)(D:Deviation로 쓰기도 함), 편차를 이용한 주요 지표

앞에 R(root)이 붙으면 전체에 제곱근 => 보통 예측값과 실값의 차가 클 때 사용

앞에 N(normalized)이 붙으면 => 값을 normalizing한 경우로 $`\bar{y}=y_{max}-y_{min}`$로 나누어주어 전체 데이터셋의 scale을 무시하기 위해 사용

보통 Mean을 이용하지만 경우에 따라서는 Median을 이용하기도 함

- Residual Sum of Square(RSS, SSR)

	1. 절대적인 오차의 크기만 지표화 => 직관적 해석
	2. 입력 크기에 의존적
	- $`\sum_{i}^{N}(y_i-\hat{y_i})^2`$

- Mean Absolute Error(MAE)

	1. outlier에 둔감robust
	2. 결과값과 동일한 단위
	3. 절댓값 => Under/Over-Estimates 인지 파악하기 어려움
	4. 입력 크기에 의존적
	- $`\frac{1}{N}\sum_{i}^{N}\left|y_i-\hat{y_i}\right|`$

- Mean Squared Error(MSE)

	1. outlier에 민감(하나의 이상치가 전체 값에 크게 반영됌)
	2. 오차가 클수록 큰 패널티 부여(잔차제곱이 1을 기준으로 값 변형이 다르게 나타남)
	3. 잔차(Residual)제곱 => Under/Over-Estimates 인지 파악하기 어려움
	4. 입력 크기에 의존적
	5. 실제값과 유사한 단위로 사용하기 위해 RMSE로 적용할 수 있음
	- $`\frac{1}{N}\sum_{i}^{N}(y_i-\hat{y_i})^2`$

- Mean Squared Logarithmic Error(MSLE)

	1. outlier에 둔감robust
	2. 상대적 Error 측정
	3. Under-Estimation에 큰 패널티 부여
	- $`\frac{1}{N}\sum_{i}^{N}(log(y_i+1)-log(\hat{y_i}+1))^2`$

- Mean Absolute Percentage Error(MAPE)

	1. 입력 크기에 의존적이지 않음
	2. outlier에 둔감robust
	- $`\frac{1}{N}\sum_{i}^{N}\left|\frac{y_i-\hat{y_i}}{y_i}\right|\cdot100(\%)`$

- Mean Percentage Error(MPE)

	1. Under/Over-Estimates 파악하기 쉬움
	- $`\frac{1}{N}\sum_{i}^{N}(y_i-\hat{y_i})\cdot100(\%)`$

- 결정계수, R-squaured, R2, Coefficient of determination

	1. 모형이 주어진 자료에 적합한 정도를 나타냄(상관분석보다는 회귀분석에 해당, 변수간의 상관도보다 모델의 변동이 고려된 상관도이기에), 독립변수(입력변수)가 종속변수(출력변수)를 얼마나 잘 설명하는지 보여주는 척도 => 불필요한 입력 데이터 판별에 활용
	2. 종속변수의 분산 중에서 모형이 설명가능한 부분의 비율의 척도를 의미
	3. `1`에 가까울 수록 모델이 좋은 성능을 가짐, 경우에 따라서는 `0`보다 작은 값이 나올 수 있는데(일괄 평균으로 예측할 때보다 성능이 떨어지는 경우) 이는 모델과 전혀 상관없는 임의의 데이터를 사용하는 경우라 할 수 있음
	4. ANOVA(Analysis of Variance, 분산분석)

	- $`R^2 = \frac{SSE}{SST} = 1 - \frac{SSR}{SST}`$

	~~용어가 상당히 혼용되어 있는데 수식보고 이해할 것~~

	($`\bar{y}=\frac{1}{N}\sum_{i}^{N}y_i`$)
	- Sum of Squared Total(SST, TSS): 총제곱합, **전체의 변동**(편차)을 나타냄: $`\sum_{i}^{N}(y_i-\bar{y_i})^2`$
	- Sum of Squared Residual(SSR, RSS)(Sum of Squared Error(SSE)): 잔차 제곱합, 모형에 의해 **설명이 되지 않는 변동**을 나타냄: $`\sum_{i}^{N}(y_i-\hat{y_i})^2`$
	- Explained Sum of Squares(SSE)(Sum of Squared Regression(SSR)): 회귀 제곱합, 회귀에 대한 변동성, 즉 모형에 의해 **설명된 변동**을 나타냄: $`\sum_{i}^{N}(\hat{y_i}-\bar{y_i})^2`$

- Adjusted R2

	결정계수는 학습 데이터의 크기가 증가할 수록 모형의 성능과 상관없이 값이 커지는 경향, 독립 변수 개수의 증가에도 동일한 경향이 존재하기에 이를 수정한 방법
	1. Adjusted R2가 R2보다 값이 작다면 이는 불필요한 독립변수가 존재한다는 지표로도 해석할 수 있음.(출처:EasyFlow회귀분석, 한나래, pp. 48 ~ 50, 140~142)

	- $`R^2_{adj} = 1-(1-R^2)\left(\frac{N-1}{N-k}\right)=1-\left(\frac{\frac{SSR}{N-k}}{\frac{SST}{N-1}}\right),\,N:데이터크기,\,k:독립변수개수`$

## 분류
- **Confusion Matrix**

# Confusion Matrix

(one-class)binary classification에서 우선 정의한 후, multi classification으로 확장하여 적용할 수 있다.

![Confusion Matrix in binary class](./src/Confusion%20Matrix.png)

보통 accuracy, precision, Recall, F-score, AUC를 많이 사용

### 정확도 Accuracy
= True / 전체

### 정밀도 Precision
= 해당 결과의 측정에서, True / 전체
Positive Predictive Value(PPV) <-> Negative Predictive Value(NPV)

### 재현도 Recall
= 해당 결과의 실제에서, True / 전체

### 민감도 Sensitivity
= 재현도

### 특이도 Specificity
민감도와 반비례
True Positive Rate(TPR) <-> False Positive Rate(FPR)

#### Trade-off, Error

### F-score, F-Measure

General $`F_\beta-score`$, 여기서 $`\beta`$값은 R(recall)이 P(precision)의 $`\beta`$배 만큼 중요하다고 고려될 때 나타내는 값이다. ($`\beta`$는 양수만 고려됌)

Van Rijsbergen's effectiveness measure에 기반한 아이디어

$$F-score = \frac{1}{\alpha\frac{1}{P}+(1-\alpha)\frac{1}{R}} = \frac{(\beta^2+1)PR}{\beta^2P+R}$$
$$\alpha=\frac{1}{1+\beta^2}$$

$`\alpha=1/2, \beta^2=1`$일 때, P와 R은 같은 가중치를 가지는 F-score이므로

일반적으로 F1-score는 P와 R의 조화평균을 나타내는 지표로도 사용

#### G-score, G-Measure

반대로 기하평균을 사용하는 경우

$$G-score = \sqrt{PR}$$

### ROC(Receiver Operating Characteristic) curve
fall-out : FPR = 1 - Specificity

### AUC(Area Under Curve)

### BLEU

### Rouge Score(Recall-Oriented Understudy for Gisting Evaluation)


## 확장

분류에서의 확장은 크게 두가지가 있다.
1. multi-class classification
2. multi-label classification

두 확장 binary 에서의 metric을 구하는 방식을 확장하여 사용할 수 있다.
이런 확장된 metric(Aggregate Metrics)은 크게 2가지가 존재한다.
1. Macro Averaging: 각 class의 metric을 구한 뒤, 이들을 평균낸다
2. Micro Averaging: 모든 class를 하나의 confusion matrix로 합친 뒤, metric을 구한다.

### multi-label classification로의 확장

각 class끼리 mutually exclusive함이 보장되지 않으므로 공통부분을 활용해 각 metric를  집합의 관계로 재해석할 필요가 있다.
또는 완전히 동일한 경우만 판단하는 방법도 존재한다.

- **Exact Match Ratio**
- **0/1 Loss**
- **Hamming Loss**

# REF

1. https://mmuratarat.github.io/2020-01-25/multilabel_classification_metrics
2. https://www.semanticscholar.org/paper/A-Literature-Survey-on-Algorithms-for-Multi-label-Sorower/6b5691db1e3a79af5e3c136d2dd322016a687a0b?p2df
3. https://scikit-learn.org/stable/modules/model_evaluation.html
4. EasyFlow회귀분석, 한나래, pp. 48 ~ 50, 140~142
5. https://en.wikipedia.org/wiki/Coefficient_of_determination
6. https://docs.microsoft.com/ko-kr/azure/machine-learning/how-to-understand-automated-ml
7. https://en.wikipedia.org/wiki/Confusion_matrix
