# Chap13 - Convolutional Neural Networks

## 13.1 시각 피질의 구조

David H. Hubel과 Torsten Wiesel은 1958년과 1959년에 시각 피질의 구조에 대한 결정적인 통찰을 제공한 고양이 실험을 수행했다. 이들은 시각 피질 안의 많은 뉴런이 작은 **local receptive field**(국부 수용영역)을 가진다는 것을 보였으며, 이것은 뉴런들이 시야의 일부 범위 안에 있는 시각 자극에만 반응을 한다는 의미이다. 뉴런의 수용영역(receptive field)들은 서로 겹칠수 있으며, 이렇게 겹쳐진 수용영역들이 전체 시야를 이루게 된다. 추가적으로 어떤 뉴런은 수직선의 이미지에만 반응하고, 다른 뉴런은 다른 각도의 선에 반응하는 뉴런이 있을 뿐만아니라, 어떤 뉴런은 큰 수용영역을 가져 저수준의 패턴(edge, blob 등)이 조합되어 복잡한 패턴(texture, object)에 반응하다는 것을 알게 되었다.  이러한 관찰을 통해 고수준의 뉴런이 이웃한 저수준의 뉴런의 출력에 기반한다는 아이디어를 생각해 냈다. (아래 그림출처 : [brainconnection](https://brainconnection.brainhq.com/2004/03/06/overview-of-receptive-fields/))



![](./images/receptive.jpg)



이러한 아이디어가 바로 **합성곱 신경망(CNN, Convolutional Neural Network)**으로 점차 진화되어 왔으며, 1998년 Yann Lecn et al.의 논문에서 손글씨 숫자를 인식하는데 사용한 LeNet-5가 소개 되면서 CNN이 등장하게 되었다.

CNN의 구조는 아래의 그림과 같이 완전연결(fully connected)계층과는 달리 CNN은 **합성곱층(covolutional layer)**과 **풀링층(pooling layer)**으로 구성되어 있다.

![](./images/cnn-vs-fcn.png)





## 13.2 합성곱층 (Convolutional layer)

### 13.2.1 완전연결 계층의 문제점

완전연결 계층(fully connected layer)을 이용해 MNIST 데이터셋을 분류하는 모델을 만들 때,  3차원(세로, 가로, 채널)인 MNIST 데이터(28, 28, 1)를 입력층(input layer)에 넣어주기 위해서 아래의 그림(출처: [cntk.ai](https://cntk.ai/pythondocs/CNTK_103A_MNIST_DataLoader.html))처럼, 3차원 → 1차원의 평평한(flat) 데이터로 펼쳐줘야 했다.  즉, (28, 28, 1)의 3차원 데이터를 $28 \times 28 \times 1 = 784$의 1차원 데이터로 바꾼다음 입력층에 넣어줬다.



![](./images/mnist-fc.png)



이러한 완전연결 계층의 문제점은 바로 **'데이터의 형상이 무시'**된다는 것이다. 이미지 데이터의 경우 3차원(세로, 가로, 채널)의 형상을 가지며, 이 형상에는 **공간적 구조(spatial structure)**를 가진다. 예를 들어 공간적으로 가까운 픽셀은 값이 비슷하거나, RGB의 각 채널은 서로 밀접하게 관련되어 있거나, 거리가 먼 픽셀끼리는 관련이 없는 등, 이미지 데이터는 3차원 공간에서 이러한 정보들이 내포 되어있다. 하지만, 완전연결 계층에서 1차원의 데이터로 펼치게 되면 이러한 정보들이 사라지게 된다.



### 13.2.2 합성곱층

합성곱층은 CNN에서 가장 중요한 구성요소이며, 13.2.1의 완전연결 계층과는 달리 **합성곱층(convolutional layer)**은 아래의 그림과 같이 입력 데이터의 형상을 유지한다. 3차원의 이미지 그대로 입력층에 입력받으며, 출력 또한 3차원 데이터로 출력하여 다음 계층(layer)으로 전달하기 때문에 CNN에서는 이미지 데이터처럼 형상을 가지는 데이터를 제대로 학습할 가능성이 높다고 할 수 있다.

   

![](./images/cnn.png)



합성곱층의 뉴런은 아래의 그림처럼(출처: [towardsdatascience.com](https://www.google.co.kr/url?sa=i&source=images&cd=&cad=rja&uact=8&ved=2ahUKEwiisMajvYzeAhWBzbwKHQwADpsQjhx6BAgBEAM&url=https%3A%2F%2Ftowardsdatascience.com%2Fintuitively-understanding-convolutions-for-deep-learning-1f6f42faee1&psig=AOvVaw2rBeiGhqGeRHABcckWUyi1&ust=1539831412136958)) 입력 이미지의 모든 픽셀에 연결되는 것이 아니라 합성곱층 뉴런의 **수용영역(receptive field)안에 있는 픽셀에만 연결**이 되기 때문에, 앞의 합성곱층에서는 저수준 특성에 집중하고, 그 다음 합성곱층에서는 고수준 특성으로 조합해 나가도록 해준다. 

<img src="./images/conv-layer.gif" height="50%" width="50%" />

![](./images/cnn-network.png)





### 13.2.3 필터 (Filter)

위에서 설명한 수용영역(receptive field)을 합성곱층에서 **필터(filter)** 또는 커널(kernel)이라고 한다. 아래의 그림처럼, 이 필터가 바로 합성곱층에서의 가중치 파라미터($\mathbf{W}$)에 해당하며, 학습단계에서 적절한 필터를 찾도록 학습되며,  합성곱 층에서 입력데이터에 필터를 적용하여 필터와 유사한 이미지의 영역을 강조하는 **특성맵(feature map)**을 출력하여 다음 층(layer)으로 전달한다.

![](./images/filter.png)



그렇다면, 입력 데이터와 필터에 어떠한 연산을 통해 특성맵을 출력하는지에 대해 알아보도록 하자.



### 13.2.4 합성곱 (Convolution) vs. 교차 상관 (Cross-Correlation)

합성곱은 *'하나의 함수와 또 다른 함수를 **반전** 이동한 값을 곱한 다음, 구간에 대해 적분하여 새로운 함수를 구하는 연산자이다'* 라고 [wikipedia](https://ko.wikipedia.org/wiki/%ED%95%A9%EC%84%B1%EA%B3%B1)에서 정의하고 있다. 합성곱 연산은 푸리에 변환(Fourier transform)과 라플라스 변환(Laplace transform)에 밀접한 관계가 있으며 신호 처리 분야에서 많이 사용된다. 


$$
\left( f * g \right)(t) = \int_{-\infty}^{\infty}{f(\tau)g(t-\tau)}d \tau
$$


이미지의 경우 2차원의 평면(높이 $h$, 너비 $w$)이며, 픽셀로 구성되어 있어 아래와 같이 ($\Sigma$)를 이용해 나타낼 수 있으며, 한 함수가 다른 함수 위를 이동하면서 원소별(element-wise) 곱셈의 합을 계산하는 연산이다.


$$
\left( f*g \right)(i, j) = \sum_{x=0}^{h-1}{\sum_{y=0}^{w-1}{f(x,y)g\left(i-x, j-y\right)}}
$$


합성곱과 매우 유사한 연산을 하는 **교차 상관(cross-correlation)**이 있는데, 교차상관의 식은 다음과 같다.


$$
\left( f * g \right)(t) = \int_{-\infty}^{\infty}{f(\tau)g(t+\tau)}d \tau
$$

$$
\left( f*g \right)(i, j) = \sum_{x=0}^{h-1}{\sum_{y=0}^{w-1}{f(x,y)g\left(i+x, j+y\right)}}
$$



합성곱과 교차상관의 차이는 한 함수(위에서 $g$)를 반전($-$)하는 것만 빼고는 동일한 함수이다. 

![convolution-vs-correlation](./images/conv-cross.PNG)



CNN의 합성곱층(convolutional layer)에서는 합성곱이 아닌, 교차상관(cross-correlation)을 사용하는데, 그 이유는 합성곱 연산을 하려면, 필터(filter/kernel)를 뒤집은(반전) 다음 적용해야 한다. 그런데, CNN에서는 필터의 값을 학습하는 것이 목적이기 때문에, 합성곱을 적용하는 것이나 교차상관을 적용하는 것이나 동일하다. 다만, 학습단계와 추론(inference) 단계에서 필터만 일정하면 된다. 이러한 이유로 [텐서플로](https://www.tensorflow.org/api_docs/python/tf/layers/conv2d)나 다른 딥러닝 프레임워크들은 합성곱이 아닌 교차상관으로 합성곱층이 구현되어 있다(참고: [tensorflow.blog](https://tensorflow.blog/2017/12/21/convolution-vs-cross-correlation/)). 



### 13.2.5 합성곱층 연산

그럼, 합성곱 계층에서 연산이 어떻게 이루어지는지 알아보도록 하자. 데이터와 필터(또는 커널)의 모양을 (높이, 너비)로 나타내고, 윈도우(Window)라고 부른다. 여기서 입력 데이터는 (4, 4), 필터는 (3, 3)이고, 필터가 바로 **Conv Layer의 가중치에 해당**한다. 

합성곱 연산은 필터의 윈도우를 일정한 간격으로 이동해가며 계산한다. 아래의 그림처럼, 합성곱 연산은 입력데이터와 필터간에 서로 대응하는 원소끼리 곱한 후 총합을 구하게 되며, 이것을 Fused Multiply-Add(FMA)라고한다. 마지막으로 편향(bias)은 필터를 적용한 후에 더해주게 된다.



![convolution](./images/conv-op.png)



### 13.2.6 패딩 (padding)

패딩(Padding)은 합성곱 연산을 수행하기 전, 입력데이터 주변을 특정값으로 채워 늘리는 것을 말한다. 패딩(Padding)은 주로 출력데이터의 공간적(Spatial)크기를 조절하기 위해 사용한다. 패딩을 할 때 채울 값은 hyper-parameter로 어떤 값을 채울지 결정할 수 있지만, 주로 **zero-padding**을 사용한다. 

패딩을 사용하는 이유는 패딩을 사용하지 않을 경우, 데이터의 Spatial 크기는 Conv Layer를 지날 때 마다 작아지게 되므로, 가장자리의 정보들이 사라지는 문제가 발생하기 때문에 패딩을 사용하며, 주로 합성곱 계층의 출력이 입력 데이터의 공간적 크기와 동일하게 맞춰주기 위해 사용한다.

![padding](./images/padding.png)



### 13.2.7 스트라이드(Stride)

스트라이드는 입력데이터에 필터를 적용할 때 이동할 간격을 조절하는 것, 즉 **필터가 이동할 간격을 말한다**. 스트라이드 또한 출력 데이터의 크기를 조절하기 위해 사용한다. 스트라이드(Stride)는 보통 1과 같이 작은 값이 더 잘 작동하며, Stride가 1일 경우 입력 데이터의 spatial 크기는 pooling 계층에서만 조절하게 할 수 있다. 아래의 그림은 1폭 짜리 zero-padding과 Stride값을 1로 적용한 뒤 합성곱 연산을 수행하는 예제이다.

![convolution](./images/conv-layer2.gif)



### 13.2.8 출력 크기 계산

패딩과 스트라이드를 적용하고, 입력데이터와 필터의 크기가 주어졌을 때 출력 데이터의 크기를 구하는 식은 아래와 같다.


$$
\left(\text{OH, OW} \right) = \left( \frac{\text{H} + 2\text{P} - \text{FH}}{\text{S}} +1, \frac{\text{W} + 2\text{P} - \text{FW}}{\text{S}} + 1 \right)
$$


- $(\text{H, W})$ : 입력 크기 (input size)
- $(\text{FH, FW})$ : 필터 크기 (filter/kernel size)
- $\text{S}$ : 스트라이드 (stride)
- $\text{P}$ : 패딩 (padding)
- $(\text{OH, OW})$ : 출력 크기 (output size)



아래의 그림은 패딩(padding) 1,  스트라이드(stride) 1 일때의 출력데이터 크기를  구한 예제다.  출력크기가 정수가 아닌 경우에는 에러가 발생할 수 있는데, 보통 딥러닝 프레임워크에서는 반올림을 통해 에러없이 작동한다.

![](./images/conv-output.PNG)
$$
\left(\text{OH, OW} \right) = \left( \frac{4 + 2 \times 1 - 3}{1} + 1, \frac{4 + 2 \times 1 - 3}{1} + 1 \right) = (4, 4)
$$




### 13.2.9 3차원 데이터의 합성곱

지금까지는 이미지 데이터에서 채널(channel)을 제외한 2차원(높이, 너비)의 형상에 대해 합성곱층에서의 연산에 대해 알아보았다. 이번에는 채널을 고려한 3차원 데이터에 대해 합성곱 연산을 알아보도록 하자. 예를 들어 아래의 그림(출처: [밑바닥부터 시작하는 딥러닝](https://github.com/WegraLee/deep-learning-from-scratch))처럼, 3개의 채널을 가지는 이미지의 다음과 같이 합성곱 연산을 수행할 수 있는데, 여기서 주의해야할 점은 합성곱 연산을 수행할 때, **입력 데이터의 채널 수와 필터의 채널수가 같아야 한다**.

![](./images/3d-conv.PNG)



#### 블록으로 생각하기

3차원의 합성곱 연산은 입력 데이터와 필터를 아래의 그림처럼 직육면체의 블록으로 생각하면 쉽다. 3차원 데이터의 모양은 `(높이, 너비, 채널) = (Height, Width, Channel)` 순으로 표현한다. 



![](./images/3d-conv02.PNG)



위의 그림에서 볼 수 있듯이, 3차원 입력 데이터에 하나의 필터를 이용해 합성곱 연산을 하게 되면 출력으로는 하나의 채널을 가지는 특성맵이다. 출력 데이터 또한 여러개의 채널을 가지는 특성맵을 내보내기 위해서는 여러개의 필터를 사용하면 된다. 아래의 그림은 `FN`개의 필터를 적용해 `FN`개의 채널을 가지는 특성맵을 출력으로 내보내는 예시이며, 합성곱 연산에서도 편향(bias)이 쓰이기 때문에 편향을 더해주기 위해 `(1, 1, FN)` 모양의 편향을 더해준 것이다.



![](./images/3d-conv03.PNG)



