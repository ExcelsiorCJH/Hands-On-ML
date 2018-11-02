>  이번 포스팅은 [핸즈온 머신러닝](http://www.yes24.com/24/goods/59878826?scode=032&OzSrank=1) 교재를 가지고 공부한 것을 정리한 포스팅입니다. 

ㅇㄹ

# 08. 오토인코더 - Autoencoder



저번 포스팅 [07. 순환 신경망, RNN](http://excelsior-cjh.tistory.com/183)에서는 자연어, 음성신호, 주식과 같은 연속적인 데이터에 적합한 모델인 RNN, LSTM, GRU에 대해 알아보았다. 이번 포스팅에서는 딥러닝에서의 비지도 학습(unsupervised learning)이라고 할 수 있는 **오코인코더**(autoencoder)에 대해 알아보도록 하자.



## 1. 오토인코더 란?

오토인코더(Autoencoder)는 아래의 그림과 같이 단순히 입력을 출력으로 복사하는 신경망이다. 어떻게 보면 간단한 신경망처럼 보이지만 네트워크에 여러가지 방법으로 제약을 줌으로써 어려운 신경망으로 만든다. 예를들어 아래 그림처럼 hidden layer의 뉴런 수를 input layer(입력층) 보다 작게해서 데이터를 압축(차원을 축소)한다거나, 입력 데이터에 노이즈(noise)를 추가한 후 원본 입력을 복원할 수 있도록 네트워크를 학습시키는 등 다양한 오토인코더가 있다. 이러한 제약들은 오토인코더가 단순히 입력을 바로 출력으로 복사하지 못하도록 방지하며, 데이터를 효율적으로 표현(representation)하는 방법을 학습하도록 제어한다.



![](./images/ae01.png)



## 2. Uncomplete 오토인코더

오토인코더는 위의 그림에서 볼 수 있듯이 항상 인코더(encoder)와 디코더(decoder), 두 부분으로 구성되어 있다.

- **인코더(encoder)** : 인지 네트워크(recognition network)라고도 하며, 입력을 내부 표현으로 변환한다.
- **디코더(decoder)** : 생성 네트워크(generative nework)라고도 하며, 내부 표현을 출력으로 변환한다.

오토인코더는 위의 그림에서 처럼, 입력과 출력층의 뉴런 수가 동일하다는 것만 제외하면 일반적인 MLP(Multi-Layer Perceptron)과 동일한 구조이다. 오토인코더는 입력을 재구성하기 때문에 출력을 **재구성(reconstruction)**이라고도 하며, 손실함수는 입력과 재구성(출력)의 차이를 가지고 계산한다. 

위 그림의 오토인토더는 히든 레이어의 뉴런(노드, 유닛)이 입력층보다 작으므로 입력이 저차원으로 표현되는데, 이러한 오토인코더를 **Undercomplete Autoencoder**라고 한다. undercomplete 오토인코더는 저차원을 가지는 히든 레이어에 의해 입력을 그대로 출력으로 복사할 수 없기 때문에, 출력이 입력과 같은 것을 출력하기 위해 학습해야 한다. 이러한 학습을 통해 undercomplete 오토인코더는 입력 데이터에서 가장 중요한 특성(feature)을 학습하도록 만든다.



### 2.1 Undercomplete Linear 오토인코더로 PCA 구현하기

위에서 살펴본 Undercomplete 오토인코더에서 활성화 함수를 sigmoid, ReLU같은 비선형(non-linear)함수가 아니라 선형(linear) 함수를 사용하고, 손실함수로 MSE(Mean Squared Error)를 사용할 경우에는 [PCA](http://excelsior-cjh.tistory.com/167?category=918734)라고 볼 수 있다.

아래의 예제코드는 가상의 3차원 데이터셋을 undercomplete 오토인코더를 사용해 2차원으로 축소하는 PCA를 수행한 코드이다. 전체 코드는 [ExcelsiorCJH's GitHub](https://github.com/ExcelsiorCJH/Hands-On-ML/blob/master/Chap15-Autoencoders/Chap15-Autoencoders.ipynb)에서 확인할 수 있다. 

```python
import tensorflow as tf

################
# layer params #
################
n_inputs = 3
n_hidden = 2  # coding units
n_outputs = n_inputs

# autoencoder
X = tf.placeholder(tf.float32, shape=[None, n_inputs])
hidden = tf.layers.dense(X, n_hidden)
outputs = tf.layers.dense(hidden, n_outputs)

################
# Train params #
################
learning_rate = 0.01
n_iterations = 1000
pca = hidden

# loss
reconstruction_loss = tf.reduce_mean(tf.square(outputs - X))  # MSE
# optimizer
train_op = tf.train.AdamOptimizer(learning_rate).minimize(reconstruction_loss)

with tf.Session() as sess:
    tf.global_variables_initializer().run()
    for iteration in range(n_iterations):
        train_op.run(feed_dict={X: X_train})
    pca_val = pca.eval(feed_dict={X: X_test})
```

![](./images/unc-ae.PNG)



위의 코드에서 입력의 개수(`n_inputs`)와 출력의 개수(`n_outputs`)가 동일한 것을 알 수 있으며, PCA를 위해 `tf.layers.dense()`에서 따로 활성화 함수를 지정해주지 않아 모든 뉴런이 선형인 것을 알 수 있다. 



## 3. Stacked 오토인코더

**Stacked** 오토인코더 또는 **deep** 오토인코더는 여러개의 히든 레이어를 가지는 오토인코더이며, 레이어를 추가할수록 오토인코더가 더 복잡한 코딩(부호화)을 학습할 수 있다. stacked 오토인코더의 구조는 아래의 그림과 같이 가운데 히든레이어(코딩층)을 기준으로 대칭인 구조를 가진다.

![stacked autoencoder](./images/stacked-ae.PNG)



### 3.1 텐서플로로 stacked 오토인코더 구현

Stacked 오토인코더는 기본적인 Deep MLP와 비슷하게 구현할 수 있다. 아래의 예제는 [He](http://excelsior-cjh.tistory.com/177?category=940400) 초기화, [ELU](http://excelsior-cjh.tistory.com/178?category=940400) 활성화 함수, $l_2$ 규제(regularization)을 사용해 MNIST 데이터셋에 대한 stacked 오토인코더를 구현한 코드이다. 오토인코더는 레이블이 없는 즉, 레이블(?, 정답)이 입력 자기자신이기 때문에 MNIST 데이터셋의 레이블은 사용되지 않는다. 아래 코드의 전체코드는 ['ExcelsiorCJH's GitHub](https://github.com/ExcelsiorCJH/Hands-On-ML/blob/master/Chap15-Autoencoders/Chap15-Autoencoders.ipynb)에서 확인할 수 있다.

```python
import numpy as np
import tensorflow as tf
from functools import partial

################
# layer params #
################
n_inputs = 28 * 28
n_hidden1 = 300  # encoder
n_hidden2 = 150  # coding units
n_hidden3 = n_hidden1  # decoder
n_outputs = n_inputs  # reconstruction

################
# train params #
################
learning_rate = 0.01
l2_reg = 0.0001
n_epochs = 5
batch_size = 150
n_batches = len(train_x) // batch_size

# set the layers using partial
he_init = tf.keras.initializers.he_normal()  # He 초기화
l2_regularizer = tf.contrib.layers.l2_regularizer(scale=l2_reg)  # L2 규제
dense_layer = partial(tf.layers.dense,
                      activation=tf.nn.elu,
                      kernel_initializer=he_init,
                      kernel_regularizer=l2_regularizer)

# stacked autoencoder
inputs = tf.placeholder(tf.float32, shape=[None, n_inputs])

hidden1 = dense_layer(inputs, n_hidden1)
hidden2 = dense_layer(hidden1, n_hidden2)
hidden3 = dense_layer(hidden2, n_hidden3)
outputs = dense_layer(hidden3, n_outputs, activation=None)

# loss
reconstruction_loss = tf.reduce_mean(tf.square(outputs - inputs))
reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
loss = tf.add_n([reconstruction_loss] + reg_losses)

# optimizer
train_op = tf.train.AdamOptimizer(learning_rate).minimize(loss)

# Train
with tf.Session() as sess:
    tf.global_variables_initializer().run()
    for epoch in range(n_epochs):
        for iteration in range(n_batches):
            batch_x, batch_y = next(shuffle_batch(train_x, train_y, batch_size))
            sess.run(train_op, feed_dict={inputs: batch_x})
        loss_train = reconstruction_loss.eval(feed_dict={inputs: batch_x})
        print('epoch : {}, Train MSE : {:.5f}'.format(epoch, loss_train))
        
'''
epoch : 0, Train MSE : 0.02418
epoch : 1, Train MSE : 0.01289
epoch : 2, Train MSE : 0.01061
epoch : 3, Train MSE : 0.01056
epoch : 4, Train MSE : 0.01062
'''
```



위의 코드를 학습 시킨 후에 테스트셋의 일부를 재구성하였을 때, 아래의 그림과 같은 결과가 나온다.

![](./images/stacked-ae02.PNG)





### 3.2 가중치 묶기

위(3.1)에서 구현한 stacked 오토인코더처럼, 오토인코더가 완전히 대칭일 때에는 일반적으로 인코더(encoder)의 가중치와 디코더(decoder)의 가중치를 묶어준다. 이렇게 가중치를 묶어주게 되면, 네트워크의 가중치 수가 절반으로 줄어들기 때문에 학습 속도를 높이고 오버피팅의 위험을 줄여준다.

![](./images/stacked-ae03.PNG)



위의 그림을 수식으로 나타내면, 예를들어 오토인코더가 $N$개의 층을 가지고 있고 $\mathbf{W}_L$ 이 $L$ 번째 층의 가중치를 나타낸다고 할 때, 디코더 층의 가중치는 $\mathbf{W}_{N-L+1} = \mathbf{W}_{L}^{T}$로 정의할 수 있다. 

텐서플로에서는 `tf.layers.dense()`를 이용해서 가중치를 묶는 것이 복잡하기 때문에 아래의 코드처럼 직접 층을 구현해주는 것이 좋다. 아래의 예제는 3.1에서 구현한 stacked 오토인코더를 가중치를 묶어서 작성한 코드이다.

```python
import numpy as np
import tensorflow as tf

################
# layer params #
################
n_inputs = 28 * 28
n_hidden1 = 300  # encoder
n_hidden2 = 150  # coding units
n_hidden3 = n_hidden1  # decoder
n_outputs = n_inputs  # reconstruction

################
# train params #
################
learning_rate = 0.01
l2_reg = 0.0005
n_epochs = 5
batch_size = 150
n_batches = len(train_x) // batch_size

# set the layers using partial
activation = tf.nn.elu
weight_initializer = tf.keras.initializers.he_normal()  # He 초기화
l2_regularizer = tf.contrib.layers.l2_regularizer(scale=l2_reg)  # L2 규제

inputs = tf.placeholder(tf.float32, shape=[None, n_inputs])

W1_init = weight_initializer([n_inputs, n_hidden1])
W2_init = weight_initializer([n_hidden1, n_hidden2])

# Encoder weights
W1 = tf.Variable(W1_init, dtype=tf.float32, name='W1')
W2 = tf.Variable(W2_init, dtype=tf.float32, name='W2')
# Decoder weights
W3 = tf.transpose(W2, name='W3')  # 가중치 묶기
W4 = tf.transpose(W1, name='W4')  # 가중치 묶기

# bias
b1 = tf.Variable(tf.zeros(n_hidden1), name='b1')
b2 = tf.Variable(tf.zeros(n_hidden2), name='b2')
b3 = tf.Variable(tf.zeros(n_hidden3), name='b3')
b4 = tf.Variable(tf.zeros(n_outputs), name='b4')

hidden1 = activation(tf.matmul(inputs, W1) + b1)
hidden2 = activation(tf.matmul(hidden1, W2) + b2)
hidden3 = activation(tf.matmul(hidden2, W3) + b3)
outputs = tf.matmul(hidden3, W4) + b4

# loss
reconstruction_loss = tf.reduce_mean(tf.square(outputs - inputs))
reg_loss = l2_regularizer(W1) + l2_regularizer(W2)
loss = reconstruction_loss + reg_loss

# optimizer
train_op = tf.train.AdamOptimizer(learning_rate).minimize(loss)

# Train
with tf.Session() as sess:
    tf.global_variables_initializer().run()
    for epoch in range(n_epochs):
        for iteration in range(n_batches):
            batch_x, batch_y = next(shuffle_batch(train_x, train_y, batch_size))
            sess.run(train_op, feed_dict={inputs: batch_x})
        loss_train = reconstruction_loss.eval(feed_dict={inputs: batch_x})
        print('epoch : {}, Train MSE : {:.5f}'.format(epoch, loss_train))
        
'''
epoch : 0, Train MSE : 0.01495
epoch : 1, Train MSE : 0.01514
epoch : 2, Train MSE : 0.01716
epoch : 3, Train MSE : 0.01761
epoch : 4, Train MSE : 0.01720
'''
```



### 3.3 한 번에 한 층씩 학습하기

3.1과 3.2에서 처럼 한 번에 전체 오토인코더를 학습시키는 것보다 아래의 그림처럼 한 번에 오토인코더 하나를 학습하고, 이를 쌓아올려서 한 개의 stacked-오토인코더를 만드는 것이 훨씬 빠르며 이러한 방식은 아주 깊은 오토인코더일 경우에 유용하다.

![](./images/stacked-ae04.PNG)



- [단계 1]에서 첫 번째 오토인코더는 입력을 재구성하도록 학습된다.
- [단계 2]에서는 두 번째 오토인코더가 첫 번째 히든 레이어(`Hidden 1`)의 출력을 재구성하도록 학습된다.

- [단계 3]에서는 단계1 ~ 2의 오토인코더를 합쳐 최종적으로 하나의 stacked-오토인코더를 구현한다.



텐서플로에서 이렇게 여러 단계의 오토인코더를 학습시키는 방법으로는 다음과 같이 두 가지 방법이 있다.

-  각 단계마다 다른 텐서플로 그래프(graph)를 사용하는 방법
- 하나의 그래프에 각 단계의 학습을 수행하는 방법

위의 두 가지 방법에 대한 코드는 [ExcelsiorCJH's GitHub](https://github.com/ExcelsiorCJH/Hands-On-ML/blob/master/Chap15-Autoencoders/Chap15-Autoencoders.ipynb) 에서 확인할 수 있다.



## 4. Stacked-오토인코더를 이용한 비지도 사전학습

대부분이 레이블되어 있지 않는 데이터셋이 있을 때, 먼저 전체 데이터를 사용해 stacked-오토인코더를 학습시킨다. 그런 다음 오토인코더의 하위 레이어를 재사용해 분류와 같은 실제 문제를 해결하기 위한 신경망을 만들고 레이블된 데이터를 사용해 학습시킬 수 있다.



![](./images/unsupervised.PNG)



위와 같은 방법을 텐서플로에서 구현할 때는 [Transfer Learning](http://excelsior-cjh.tistory.com/179?category=940399)포스팅에서 살펴본 방법과 같이 구현하면 된다. 이러한 비지도 사전학습 방법에 대한 소스코드는 [여기](https://github.com/rickiepark/handson-ml/blob/master/15_autoencoders.ipynb)에서 확인할 수 있다.



## 5. Denoising 오토인코더

오토인코더가 의미있는 특성(feature)을 학습하도록 제약을 주는 다른 방법은 입력에 노이즈(noise, 잡음)를 추가하고, 노이즈가 없는 원본 입력을 재구성하도록 학습시키는 것이다. 노이즈는 아래의 그림처럼 입력에 [가우시안(Gaussian) 노이즈](https://en.wikipedia.org/wiki/Gaussian_noise)를 추가하거나, 드롭아웃(dropout)처럼 랜덤하게 입력 유닛(노드)를 꺼서 발생 시킬 수 있다.



![denoising-autoencoder](./images/denoising.PNG)



### 5.1 텐서플로로 구현하기

이번에는 텐서플로를 이용해 가우시안 노이즈와 드롭아웃을 이용한 denoising-오토인코더를 구현해보도록 하자. 오토인코더 학습에 사용한 데이터셋은 위에서 부터 다뤘던 MNIST 데이터셋이다.  아래의 코드에 대한 전체 코드는 [ExcelsiorCJH's GitHub](https://github.com/ExcelsiorCJH/Hands-On-ML/blob/master/Chap15-Autoencoders/Chap15-Autoencoders.ipynb)에서 확인할 수 있다.

#### 5.1.1 Gaussian noise

```python
import sys
import numpy as np
import tensorflow as tf

################
# layer params #
################
noise_level = 1.0
n_inputs = 28 * 28
n_hidden1 = 300
n_hidden2 = 150  # coding units
n_hidden3 = n_hidden1
n_outputs = n_inputs

################
# train params #
################
learning_rate = 0.01
n_epochs = 5
batch_size = 150

# denoising autoencoder
inputs = tf.placeholder(tf.float32, shape=[None, n_inputs])
# add gaussian noise
inputs_noisy = inputs + noise_level * tf.random_normal(tf.shape(inputs))

hidden1 = tf.layers.dense(inputs_noisy, n_hidden1, activation=tf.nn.relu, name='hidden1')
hidden2 = tf.layers.dense(hidden1, n_hidden2, activation=tf.nn.relu, name='hidden2')
hidden3 = tf.layers.dense(hidden2, n_hidden3, activation=tf.nn.relu, name='hidden3')
outputs = tf.layers.dense(hidden3, n_outputs, name='outputs')

# loss 
reconstruction_loss = tf.losses.mean_squared_error(labels=inputs, predictions=outputs)
# optimizer
train_op = tf.train.AdamOptimizer(learning_rate).minimize(reconstruction_loss)

# saver
saver = tf.train.Saver()

# Train
with tf.Session() as sess:
    tf.global_variables_initializer().run()
    n_batches = len(train_x) // batch_size
    for epoch in range(n_epochs):
        for iteration in range(n_batches):
            print("\r{}%".format(100 * iteration // n_batches), end="")
            sys.stdout.flush()
            batch_x, batch_y = next(shuffle_batch(train_x, train_y, batch_size))
            sess.run(train_op, feed_dict={inputs: batch_x})
        loss_train = reconstruction_loss.eval(feed_dict={inputs: batch_x})
        print('\repoch : {}, Train MSE : {:.5f}'.format(epoch, loss_train))
    saver.save(sess, './model/my_model_stacked_denoising_gaussian.ckpt')
    
'''
epoch : 0, Train MSE : 0.04450
epoch : 1, Train MSE : 0.04073
epoch : 2, Train MSE : 0.04273
epoch : 3, Train MSE : 0.04194
epoch : 4, Train MSE : 0.04084
'''
```



위의 가우시안 노이즈를 추가한 denoising-오토인코더의 MNIST 재구성 결과는 다음과 같다.

![](./images/denoising02.PNG)



#### 5.1.2 Dropout

```python
import sys
import numpy as np
import tensorflow as tf

################
# layer params #
################
noise_level = 1.0
n_inputs = 28 * 28
n_hidden1 = 300
n_hidden2 = 150  # coding units
n_hidden3 = n_hidden1
n_outputs = n_inputs

################
# train params #
################
dropout_rate = 0.3
learning_rate = 0.01
n_epochs = 5
batch_size = 150

training = tf.placeholder_with_default(False, shape=(), name='training')

# denoising autoencoder
inputs = tf.placeholder(tf.float32, shape=[None, n_inputs])
# add dropout
inputs_drop = tf.layers.dropout(inputs, dropout_rate, training=training)

hidden1 = tf.layers.dense(inputs_drop, n_hidden1, activation=tf.nn.relu, name='hidden1')
hidden2 = tf.layers.dense(hidden1, n_hidden2, activation=tf.nn.relu, name='hidden2')
hidden3 = tf.layers.dense(hidden2, n_hidden3, activation=tf.nn.relu, name='hidden3')
outputs = tf.layers.dense(hidden3, n_outputs, name='outputs')

# loss 
reconstruction_loss = tf.losses.mean_squared_error(labels=inputs, predictions=outputs)
# optimizer
train_op = tf.train.AdamOptimizer(learning_rate).minimize(reconstruction_loss)

# saver
saver = tf.train.Saver()

# Train
with tf.Session() as sess:
    tf.global_variables_initializer().run()
    n_batches = len(train_x) // batch_size
    for epoch in range(n_epochs):
        for iteration in range(n_batches):
            print("\r{}%".format(100 * iteration // n_batches), end="")
            sys.stdout.flush()
            batch_x, batch_y = next(shuffle_batch(train_x, train_y, batch_size))
            sess.run(train_op, feed_dict={inputs: batch_x})
        loss_train = reconstruction_loss.eval(feed_dict={inputs: batch_x})
        print('\repoch : {}, Train MSE : {:.5f}'.format(epoch, loss_train))
    saver.save(sess, './model/my_model_stacked_denoising_dropout.ckpt')
```

![](./images/denoising03.PNG)





## 6. Sparse 오토인코더

오토인코더가 좋은 특성을 추출하도록 만드는 다른 제약 방법은 **희소성**(sparsity)를 이용하는 것인데, 이러한 오토인코더를 Sparse Autoencoder라고 한다. 이 방법은 손실함수에 적절한 항을 추가하여 오토인코더가 코딩층(coding layer, 가운데 층)에서 활성화되는 뉴런 수를 감소시키는 것이다. 예를들어 코딩층에서 평균적으로 5% 뉴런만 홀성화되도록 만들어 주게 되면, 오토인코더는 5%의 뉴런을 조합하여 입력을 재구성해야하기 때문에 유용한 특성을 표현하게 된다.

이러한 Sparse-오토인코더를 만들기 위해서는 먼저 학습 단계에서 코딩층의 실제 sparse(희소) 정도를 측정해야 하는데, 전체 학습 배치(batch)에 대해 코딩층의 평균적인 활성화를 계산한다. 배치의 크기는 너무 작지 않게 설정 해준다. 

위에서 각 뉴런에 대한 평균 활성화 정도를 계산하여 구하고, 손실함수에 **희소 손실**(sparsity loss)를 추가하여 뉴런이 크게 활성화 되지 않도록 규제할 수 있다.  예를들어 한 뉴런의 평균 활성화가 `0.3`이고 목표 희소 정도가 `0.1`이라면, 이 뉴런은 **덜** 활성화 되도록 해야한다. 희소 손실을 구하는 간단한 방법으로는 제곱 오차 $(0.3 - 0.1)^{2}$를 추가하는 방법이 있다. 하지만, Sparse-오토인코더에서는 아래의 그래프 처럼 MSE보다 더 경사가 급한 쿨백 라이블러 발산(KL-divergense, Kullback-Leibler divergense)을 사용한다.



![](./images/kl.PNG)



### 6.1 쿨백 라이블러 발산

쿨백-라이블러 발산(Kullback-Leibler divergence, KLD)은 두 확률분포의 차이를 계산하는 데 사용하는 함수이다. 예를들어 딥러닝 모델을 만들 때, 학습 데이터셋의 분포 $P(x)$와 모델이 추정한 데이터의 분포 $Q(x)$ 간에 차이를 KLD를 활용해 구할 수 있다([ratsgo's blog](https://ratsgo.github.io/statistics/2017/09/22/information/)).



$$
{ D }_{ KL }\left( P||Q \right) ={ E }_{ X\sim P }\left[ \log { \frac { P\left( x \right)  }{ Q(x) }  }  \right] ={ E }_{ X\sim P }\left[ \log { P(x) } -\log { Q(x) }  \right]
$$



Sparse-오토인코더에서는 코딩층에서 뉴런이 활성화될 목표 확률 $p$와 실제확률 $q$(학습 배치에 대한 평균 활성화) 사이의 발산을 측정하며, 식은 다음과 같다.


$$
D_{KL}\left( p||q \right) = p \log{\frac{p}{q}} + \left( 1- p \right) \log{\frac{1-p}{1-q}}
$$


위의 식을 이용해 코딩층의 각 뉴런에 대해 희소 손실을 구하고 이 손실을 모두 합한 뒤 희소 가중치 하이퍼파라미터를 곱하여 손실함수의 결과에 더해준다.
$$
Loss = \text{MSE} + \text{sparsity_weight} \times \text{sparsity_loss}
$$


### 6.2 텐서플로 구현

이번에는 텐서플로를 이용해 Sparse-오토인코더를 구현해보도록 하자. 

```python
import sys
import numpy as np
import tensorflow as tf

################
# layer params #
################
noise_level = 1.0
n_inputs = 28 * 28
n_hidden1 = 1000 # sparsity coding units
n_outputs = n_inputs

################
# train params #
################
sparsity_target = 0.1  # p
sparsity_weight = 0.2
learning_rate = 0.01
n_epochs = 20
batch_size = 1000

def kl_divergence(p, q):
    # 쿨백 라이블러 발산
    return p * tf.log(p / q) + (1 - p) * tf.log((1 - p) / (1 - q))

inputs = tf.placeholder(tf.float32, shape=[None, n_inputs])

hidden1 = tf.layers.dense(inputs, n_hidden1, activation=tf.nn.sigmoid)
outputs = tf.layers.dense(hidden1, n_outputs)

# loss
hidden1_mean = tf.reduce_mean(hidden1, axis=0)  # 배치 평균  == q
sparsity_loss = tf.reduce_sum(kl_divergence(sparsity_target, hidden1_mean))
reconstruction_loss = tf.losses.mean_squared_error(labels=inputs, predictions=outputs)
loss = reconstruction_loss + sparsity_weight * sparsity_loss

# optimizer
train_op = tf.train.AdamOptimizer(learning_rate).minimize(loss)

# saver
saver = tf.train.Saver()

# Train
with tf.Session() as sess:
    tf.global_variables_initializer().run()
    n_batches = len(train_x) // batch_size
    for epoch in range(n_epochs):
        for iteration in range(n_batches):
            print("\r{}%".format(100 * iteration // n_batches), end="")
            sys.stdout.flush()
            batch_x, batch_y = next(shuffle_batch(train_x, train_y, batch_size))
            sess.run(train_op, feed_dict={inputs: batch_x})
        recon_loss_val, sparsity_loss_val, loss_val = sess.run([reconstruction_loss, 
                                                                sparsity_loss,
                                                                loss], feed_dict={inputs: batch_x})
        print('\repoch : {}, Train MSE : {:.5f}, \
                sparsity_loss : {:.5f}, total_loss : {:.5f}'.format(epoch, recon_loss_val,
                                                                    sparsity_loss_val, loss_val))
    saver.save(sess, './model/my_model_sparse.ckpt')
```

![](./images/sparse.PNG)



## 7. Variational AutoEncoder (VAE)

**VAE**(Variational AutoEncoder)는 2014년 D.Kingma와 M.Welling이 [Auto-Encoding Variational Bayes](https://arxiv.org/pdf/1312.6114v10.pdf) 논문에서 제안한 오토인코더의 한 종류이다. VAE는 위에서 살펴본 오터인코더와는 다음과 같은 다른점이 있다.

- VAE는 **확률적 오토인코더**(probabilistic autoencoder)다. 즉, 학습이 끝난 후에도 출력이 부분적으로 우연에 의해 결정된다.
- VAE는 **생성 오토인코더**(generatie autoencoder)이며, 학습 데이터셋에서 샘플링된 것과 같은 새로운 샘플을 생성할 수 있다.

VAE의 구조는 아래의 그림과 같다.



![](./images/vae.PNG)



VAE의 코딩층은 다른 오토인코더와는 다른 부분이 있는데 주어진 입력에 대해 바로 코딩을 만드는 것이 아니라, 인코더(encoder)는 **평균 코딩** $\mu$와 **표준편차 코딩** $\sigma$ 을 만든다. 실제 코딩은 평균이 $\mu$이고 표준편차가 $\sigma$인 가우시안 분포(gaussian distribution)에서 랜덤하게 샘플링되며, 이렇게 샘플링된 코딩을 디코더(decoder)가 원본 입력으로 재구성하게 된다.

VAE는 마치 가우시안 분포에서 샘플링된 것처럼 보이는 코딩을 만드는 경향이 있는데, 학습하는 동안 손실함수가 코딩(coding)을 가우시안 샘플들의 집합처럼 보이는 형태를 가진 코딩 공간(coding space) 또는 **잠재 변수 공간**(latent space)로 이동시키기 때문이다. 

이러한 이유로 VAE는 학습이 끝난 후에 새로운 샘플을 가우시안 분포로 부터 랜덤한 코딩을 샘플링해 디코딩해서 생성할 수 있다.



### 7.1 VAE의 손실함수

VAE의 손실함수는 두 부분으로 구성되어 있다. 첫 번째는 오토인코더가 입력을 재구성하도록 만드는 일반적인 재구성 손실(reconstruction loss)이고, 두 번째는 가우시안 분포에서 샘플된 것 샅은 코딩을 가지도록 오토인코더를 제어하는 **latent loss**이다. 이 손실함수의 식에 대해서는 [ratsgo](https://ratsgo.github.io/generative%20model/2018/01/27/VAE/)님의 블로그를 참고하면 자세히 설명되어 있다. *(나도 언젠가 이해해서 포스팅할 날이 오기를...)*



### 7.2 텐서플로 구현

```python
import sys
import numpy as np
import tensorflow as tf
from functools import partial

reset_graph()

################
# layer params #
################
n_inputs = 28 * 28
n_hidden1 = 500
n_hidden2 = 500
n_hidden3 = 20  # 코딩 유닛
n_hidden4 = n_hidden2
n_hidden5 = n_hidden1
n_outputs = n_inputs

################
# train params #
################
learning_rate = 0.001
n_digits = 60
n_epochs = 50
batch_size = 150

initializer = tf.variance_scaling_initializer()
dense_layer = partial(
    tf.layers.dense,
    activation=tf.nn.elu, 
    kernel_initializer=initializer)


# VAE
inputs = tf.placeholder(tf.float32, [None, n_inputs])
hidden1 = dense_layer(inputs, n_hidden1)
hidden2 = dense_layer(hidden1, n_hidden2)
hidden3_mean = dense_layer(hidden2, n_hidden3, activation=None) # mean coding
hidden3_sigma = dense_layer(hidden2, n_hidden3, activation=None)  # sigma coding
noise = tf.random_normal(tf.shape(hidden3_sigma), dtype=tf.float32)  # gaussian noise
hidden3 = hidden3_mean + hidden3_sigma * noise
hidden4 = dense_layer(hidden3, n_hidden4)
hidden5 = dense_layer(hidden4, n_hidden5)
logits = dense_layer(hidden5, n_outputs, activation=None)
outputs = tf.sigmoid(logits)

# loss
eps = 1e-10  # NaN을 반환하는 log(0)을 피하기 위함
xentropy = tf.nn.sigmoid_cross_entropy_with_logits(labels=inputs, logits=logits)
reconstruction_loss = tf.reduce_mean(xentropy)
latent_loss = 0.5 * tf.reduce_sum(
    tf.square(hidden3_sigma) + tf.square(hidden3_mean)
    - 1 - tf.log(eps + tf.square(hidden3_sigma)))

loss = reconstruction_loss + latent_loss

# optimizer
train_op = tf.train.AdamOptimizer(learning_rate).minimize(loss)

# saver
saver = tf.train.Saver()

# train
with tf.Session() as sess:
    tf.global_variables_initializer().run()
    saver.restore(sess, './model/my_model_variational.ckpt')
    n_batches = len(train_x) // batch_size
    for epoch in range(n_epochs):
        for iteration in range(n_batches):
            print("\r{}%".format(100 * iteration // n_batches), end="")
            sys.stdout.flush()
            batch_x, batch_y = next(shuffle_batch(train_x, train_y, batch_size))
            sess.run(train_op, feed_dict={inputs: batch_x})
        recon_loss_val, latent_loss_val, loss_val = sess.run([reconstruction_loss, 
                                                              latent_loss,
                                                              loss], feed_dict={inputs: batch_x})
        print('\repoch : {}, Train MSE : {:.5f},'.format(epoch, recon_loss_val),
               'latent_loss : {:.5f}, total_loss : {:.5f}'.format(latent_loss_val, loss_val))
    saver.save(sess, './model/my_model_variational.ckpt')
```





## 8. 마무리

이번 포스팅에서는 자기지도학습(self-supervised learning)인 오토인코더에 대해 개념과 uncomplete, stacked, denoising, sparse, VAE 오토인코더에 대해 알아보았다. 위의 코드에 대한 전체 코드는 https://github.com/ExcelsiorCJH/Hands-On-ML/blob/master/Chap15-Autoencoders/Chap15-Autoencoders.ipynb 에서 확인할 수 있다.

