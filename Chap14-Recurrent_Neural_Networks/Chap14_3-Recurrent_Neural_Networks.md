>  이번 포스팅은 [핸즈온 머신러닝](http://www.yes24.com/24/goods/59878826?scode=032&OzSrank=1) 교재, cs231n 강의를 가지고 공부한 것을 정리한 포스팅입니다. RNN에 대해 좀더 간략하게 알고 싶으신 분들은 아래의 링크를 참고하면 됩니다. 
>
> - **텐서플로 실습 위주** : [ [러닝 텐서플로]Chap06 - 텍스트2: word2vec, Bidirectional RNN, GRU, 임베딩 시각화](http://excelsior-cjh.tistory.com/156?category=940399)
> - **LSTM & GRU의 간략한 설명** : [RNN - LSTM(Long Short Term Memory networks)](http://excelsior-cjh.tistory.com/89?category=1013831)



# 07-3. 순환 신경망 (RNN, Recurrent Neural Network) - (3)

저번 포스팅인 [07-2. 순환 신경망(RNN) - (2)](http://excelsior-cjh.tistory.com/184)에서는 RNN을 학습시키는 방법인 BPTT와 텐서플로를 이용해 MNIST 분류기와 시계열 데이터를 예측하는 RNN 모델을 구현해 보았다. 그리고 심층 RNN을 구현하는 방법과 RNN에 드롭아웃을 적용하는 방법에 대해 알아보았다. 

이번 포스팅에서는 RNN의 변형이라고 할 수 있는 LSTM과 GRU에 대해 알아보도록 하자.



## 1. RNN Cell의 문제점

### 1.1 BPTT의 문제점

[저번 포스팅](http://excelsior-cjh.tistory.com/184)에서 살펴본 BPTT는 RNN에서의 역전파 방법인 BPTT(BackPropagation Through Time)은 아래의 그림과 같이 모든 타임스텝마다 처음부터 끝까지 역전파한다.



![bptt](./images/bptt01.png)



그렇기 때문에 타임 스텝이 클 경우, 위의 그림과 같이 RNN을 펼치게(unfold)되면 매우 깊은 네트워크가 될것이며, 이러한 네트워크는 [05-1. 심층 신경망 학습](http://excelsior-cjh.tistory.com/177?category=940400)에서 살펴본 **그래디언트 소실 및 폭주**(vanishing & exploding gradient) 문제가 발생할 가능성이 크다. 그리고, 계산량 또한 많기 때문에 한번 학습하는데 아주 오랜 시간이 걸리는 문제가 있다. 



#### Truncated BPTT

BPTT의 이러한 문제를 해결하기 위해 아래의 그림과 같이 타임 스텝을 일정 구간(보통 5-steps)으로 나누어 역전파(backprop)를 계산하여, 전체 역전파로 근사시키는 방법인 **Truncated BPTT**를  대안으로 사용할 수 있다. 



![truncated-bptt](./images/bptt02.png)



하지만 truncated-BPTT의 문제는 만약 학습 데이터가 장기간에 걸쳐 패턴이 발생한다고 하면, 이러한 장기간(Long-Term)의 패턴을 학습할 수 없는 문제가 있다. 



### 1.2 장기 의존성(Long-Term Dependency) 문제

[저번 포스팅](http://excelsior-cjh.tistory.com/183)에서 살펴 보았듯이 RNN은 타임 스텝 $t$에서 이전 타임 스텝($t-1$)의 상태(state, $h_{t-1}$)를 입력으로 받는 구조이기 때문에,  이전의 정보가 현재의 타임 스텝 $t$에 영향을 줄 수 있다. 따라서, RNN의 순환 뉴런(Reccurent Neurons)의 출력은 이전 타임 스텝의 모든 입력에 대한 함수이므로, 이를 **메모리 셀(memory cell)**이라고 한다. 

이렇듯, RNN은 이론적으로 모든 이전 타임 스텝이 영향을 주지만 앞쪽의 타임 스텝(예를 들어 $t=0, t=1$)은 타임 스텝이 길어질 수록 영향을 주지 못하는 문제가 발생하는데 이를 **장기 의존성(Long-Term Dependency) 문제**라고 한다. 이러한 문제가 발생하는 이유는 입력 데이터가 RNN Cell을 거치면서 특정 연산을 통해 데이터가 변환되기 때문에 일부 정보는 타임 스텝마다 사라지기 때문이다.



![](./images/rnn08.png)



이러한 문제를 해결하기 위해 장기간의 메모리를 가질 수 있는 여러 종류의 셀이 만들어졌는데, 그 중에서 대표적인 셀들이 LSTM과 GRU 셀이다. 먼저, LSTM 셀에 대해 알아보도록 하자.



## 2. LSTM Cell

