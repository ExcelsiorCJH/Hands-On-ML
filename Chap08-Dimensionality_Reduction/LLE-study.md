# 차원 축소 - Locally Linear Embedding (LLE)



> 이번 포스팅은  [Nonlinear Dimensionality Reduction by Locally Linear Ebedding](http://www.robots.ox.ac.uk/~az/lectures/ml/lle.pdf) (Roweis et.al) 논문과  [핸즈온 머신러닝](http://www.yes24.com/24/goods/59878826?scode=032&OzSrank=1) 교재를 가지고 공부한 것을 정리한 것입니다.



## 1. LLE - Locally Linear Embedding 란?

**LLE**(Locally Liner Embedding, 지역 선형 임베딩)는 [Nonlinear Dimensionality Reduction by Locally Linear Ebedding](http://www.robots.ox.ac.uk/~az/lectures/ml/lle.pdf) (Roweis et.al) 논문에서 제안된 알고리즘이다. LLE는 **비선형 차원 축소**(NonLinear Dimensionality Reduction, NLDR) 기법으로 '[차원 축소 - PCA, 주성분분석 (1)](http://excelsior-cjh.tistory.com/167)' 포스팅에서 살펴본 PCA와 달리 투영(projection)이 아닌 **매니폴드 학습**(manifold learning) 이다.



![](./images/manifold02.png)



LLE는 머신러닝 알고리즘 중 Unsuperviesed Learning에 해당하며, 서로 인접한 데이터들을 보존(neighborhood-preserving)하면서 고차원인 데이터셋을 저차원으로 축소하는 방법이다. 즉, LLE는 입력 데이터셋을 낮은 차원의 단일 글로벌 좌표계(single global coordinate system)으로 매핑하는 알고리즘이다.



## 2. LLE 알고리즘

LLE의 과정은 아래 논문의 그림과 같이 크게 세 단계로 나눌 수 있다.

1. **Step 1**: Select neighbors 
2. **Step 2**: Reconstruct with linear weights
3. **Step 3**: Map to embedded coordinates

![](./images/LLE.png)



LLE 알고리즘을 각 단계별로 자세히 살펴보도록 하자.



### Step 1: Select Neighbors

먼저, 각 데이터 포인트 $X_i​$에 대해, $X_i​$와 가장 가까운 $k​$-개의 이웃점($k​$-nearest neighbors) $X_j​$, $(j=1, \dots, k)​$ 들을 선택한다. 여기서 $k​$는 하이퍼파라미터(hyper-parmeter)로써 사람이 직접 적절한 개수를 정해준다.



### Step 2: Reconstruct with linear weights

LLE는 'Step 1'에서 선택한 각 데이터 포인트 $X_i$와 그리고 $X_i$에 가까운 $k$-개의 이웃점들 $X_j$($j=1,\dots,k$)는 매니폴드의 **locally linear patch** 상에 있거나 가까이 있을 것이라 가정한다. 

![](./images/llp.png)



이러한 가정을 바탕으로 해당 데이터 포인트 $X_i$와 가장 가까운 $k$-개의 이웃들 $X_j$로 부터 $X_i$를 가장 잘 **재구성(reconstruction)**하는 가중치 $W_{ij}$를 구한다. 즉, 이웃점 $X_j$에 대해 **linear transformation**(선형 변환)을 의미하는 $W_{ij}$와의 행렬 곱을 통해 $W_{ij}X_j \approx X_i$를 만족하는 $W_{ij}$를 구하는 것이다. $X_i$와 $W_{ij} X_j$ 간의 오차(error)를 Reconstruction error라 하고 다음과 같이 식으로 나타낼 수 있다.
$$
\varepsilon (W) = \left\| X_i - \sum_{j=1}^{k}{W_{ij}X_{j}} \right\|^{2}
$$

따라서, 위의 식 $\varepsilon(W)$를 **최소화(minimize)** 하는 문제이며, 이때의 제약식은 $\sum_{j=1}^{k}{W_{ij}} = 1$이다. 이를 식으로 나타내면 다음과 같다. 아래의 식에서 $\frac{1}{2}$는 미분을 했을 때, 편하게 계산하기 위해 넣어주는 일종의 트릭(?)이므로 최소화하는 문제에 영향을 미치지 않는다.
$$
\text{min} \quad \frac{1}{2} \left\| X_i - \sum_{j=1}^{k}{W_{ij}X_{j}} \right\|^{2}
$$

$$
\text{s.t.} \quad \sum_{j=1}^{k}{W_{ij}} = 1
$$



위의 식을 ['서포트벡터머신, SVM'](http://excelsior-cjh.tistory.com/165)에서 살펴본 라그랑제 승수법을 이용하여 계산할 수 있다. 위의 식을 라그랑지안 함수 $L$로 나타내면 다음과 같다 ($\lambda$는 라그랑제 승수이다).
$$
L(W, \lambda) = \frac{1}{2} \left\| X_i - \sum_{j=1}^{k}{W_{ij}X_{j}} \right\|^{2} - \lambda \left( \sum_{j=1}^{k}{W_{ij}} - 1 \right)
$$


이렇게 라그랑제 함수로 나타낸 함수 $L$ 을 $W$에 대한 편미분 $\frac{\partial L}{\partial W} = 0$을 통해 $W_{ij}$를 구할 수 있으며, 자세한 계산 과정은 생략한다.



