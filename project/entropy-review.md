# Entropy Review

Shannon의 <통신의 수학적 이론>에 나온 entropy 개념의 기반이 됌 => 논문의 내용은 띄어쓰기가 실제 문장의 불확실성을 감소시킨다는 것을 통계적으로 분석함
엔트로피 = 불확실성 => 높은 엔트로피 = 많은 정보와 낮은 확률

$$H(p)=-\sum_{i=1}^{n}p(x_i)log(p(x_i)), p(x): x의 확률$$

모든 사건의 정보량의 기댓값, 
깊게 들어가 보려면 ***정보이론***의 information gain에 대해 찾아보는 것이 좋다.

## Cross Entropy

왜 크로스 엔트로피를 머신러닝에 쓸까?

그러나 여기서 미지의 $p(x)$에 대한 상세한 분포를 구하는 것(모방하는 것)이 목표이므로 임의의 $q(x)$를 만들어 두 분포간의 차를 줄이는 방식으로 최적화하자! 라는 아이디어로서 활용가능

두 분포의 차이를 측정하는 방법 => KL-Divergence(Kullback-Leibler Divergence)
$$D_{KL}(p||q)=H_q(p)-H(p)=\sum_{x}p(x)log(\frac{p(x)}{q(x)})=E_{x\sim p}\left[log(\frac{p(x)}{q(x)})\right]$$
p와 q의 분포가 같으면 값이 0이 된다.

추가적으로 분포가 같을 확률의 하한을 알아낼 수 있다는 특징도 있음

## 추가 키워드

- Jensen-Shannon divergence
- log likelihood

## reference

- [Daniel Godoy, Understanding binary cross-entropy / log loss: a visual explanation, medium blog, 2018.11.22](https://towardsdatascience.com/understanding-binary-cross-entropy-log-loss-a-visual-explanation-a3ac6025181a)
- [제이미, 왜 크로스 엔트로피를 머신러닝에 쓸까?, postype blog, 2020.03.14](https://theeluwin.postype.com/post/6080524)