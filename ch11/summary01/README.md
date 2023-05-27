# 11 Vision transformer

- Transformer 원리와 전개 과정이 잘 정리된 서베이 논문

    > [A survey of Transformers](https://arxiv.org/abs/2106.04554)

    > [Efficient Transformers: A Survey](https://arxiv.org/abs/2009.06732)

- Transformer에 CV를 적용한 서베이 논문

    > [Transformers in Vision: A Survey](https://arxiv.org/abs/2101.01169)

    > [A Survey on Vision Transformer](https://arxiv.org/abs/2012.12556)

---

- 사람은 영상을 인식할 때 intention(의도)에 따라 특정 포인트를 **attention**(주목)한다.

- **VQA**(Visual Question Answering): 영상과 관련된 질문에 답을 하는 Computer Vision

  - 이를 위해서는 사람의 attention과 같은 능력을 갖춰야 한다.

    > 최근 attention mechanism만으로 구성된 transformer model이 CNN 성능을 뛰어넘는다.

---

### 11.1.1 고전 알고리즘: feature selection

> [image recognization 문제에서의 주요 해석 방법](https://www.cognex.com/ko-kr/blogs/deep-learning/research/overview-interpretable-machine-learning-2-interpreting-deep-learning-models-image-recognition)

attention을 제대로 알아보기 전에 먼저 **feature selection**(특징 선택)을 먼저 살펴보자. attention의 일종은 아니지만 마찬가지로 도움이 되는 특징만 골라서 사용한다는 점은 같다.

- 쓸모가 많은 feature(특징)은 남긴다.

- 나머지는 제거한다.

4가지 feature( $x_1, x_2, x_3, x_4$ )가 있는 예시를 살펴보자.

![feature selection ex](images/feature_selection_ex.png)

> 빨간색: 불량 / 파란색: 정상

가령 위 예시에서 feature 2개만 선택한다면(: 1), 분별력이 강한 $x_{2}, x_{3}$ 을 취할 것이다.(선택하지 않음: 0)

---

### 11.1.2 고전 알고리즘: saliency map

**saliency map**(돌출 맵)은 attention해야 하는 정도를 실수로 표현하는 방법이다. pixel 값의 변화가 급격한 부분을 모아서 mapping한 뒤 배경과 분리한다.

![saliency map](images/saliency_map.jpg)

---

### 11.1.3 딥러닝의 attention

![CV attention milestone](images/CV_attention_milestone.png)

- RAM(Recurrent Attention Model): 2014년 발표. RNN을 이용해 딥러닝에서 최초로 attention을 적용한 모델

- STN(Spatial Transformer Network): 2015년 발표. feature map에 affine(translation, scaling, rotation) transform을 적용해서 attention할 곳을 정하는 모델

  - 변환 행렬을 학습으로 알아낸다.

- SENet(Squeeze-and-Excite Network): 2017년 발표. feature map의 어느 channel에 attention할지 알아낸다.

    ![SEnet](images/SENet.png)

  > [SENet 정리: #7.2.8 SENet: Squeeze-and-Excitation block](https://github.com/erectbranch/TinyML_and_Efficient_DLC/tree/master/lec07/summary01)

---

### 11.1.4 딥러닝의 attention: self-attention

이전까지는 '중요한 부분에 더 큰 가중치를 줘서 성능을 개선'했다.

- 중요한 곳은 큰 가중치를 주고, 중요하지 않은 곳은 작은 가중치를 준다.

하지만 **self-attention**(자기 주목)은 영상을 구성하는 요소 상호 간의 관계를 찾아낸다.

![attention vs self-attention](images/attention_vs_self_attention.png)

self-attention은 프리스비가 다른 위치에 attention하는 정도를 선의 굵기로 표현한다.

- 프리스비를 잡으려는 개에 많이 attention한다.

- 하늘에는 별로 attention하지 않는다.

> 참고로 2018년 [Non-local Neural Network](https://arxiv.org/abs/1711.07971)에서 CV에 처음 self-attention이 적용되었다. 아래 식을 통해 self-attention을 계산한다.

$$ y_{i} = {{1} \over {C(\mathbf{x})}}{\sum}_{j}{a(\mathbf{x_{i}}, \mathbf{x_{j}})g(\mathbf{x_{j}})} $$

---

## 11.2 RNN

**RNN**(Recurrent Neural Network, 순환 신경망)은 hidden node끼리 edge를 이어서 순환 구조를 만든다.

![RNN](images/RNN.png)

- weight 집합: $\{U^{1}, U^{2}, U^{3}\}$ 

  - $U^{1}$ : input layer ~ **hidden layer**

  - $U^{2}$ : **hidden layer** ~ output layer

  - $U^{3}$ : **hidden layer** ~ **hidden layer**

이제 문장을 표현한 벡터 $\mathbf{o=(x^{1}, x^{2}, x^{3} , \cdots , x^{T})}$ 를 입력으로 받는다고 하자. 길이는 $T$ 이며, $i$ time의 단어 $x^{i}$ 는 벡터다.

> 예를 들어 '저게 / 저절로 / 붉어질 / 리 / 없다'는 문장은 $T=5$ 이다. embedding되어 벡터로 입력이 들어간다.

위 구조 그림에서 '(c) 시간축으로 펼침' 부분을 보자.

- $\mathbf{x}^{3}, \mathbf{h}^{2}$ 가 입력되어 $\mathbf{h}^{3}$ 이 생성된다. 출력으로 $\mathbf{o}^{3}$ 이 나온다.

- 일반화: $\mathbf{x}^{i}, \mathbf{h}^{i-1}$ 이 입력되어 $\mathbf{h}^{i}$ 가 생성된다. 출력으로 $\mathbf{o}^{i}$ 가 나온다.

일반화는 time $i$ d에서 출력이 나오기까지의 동작을 정의한 것이다. 좀 더 자세히 살펴보자.

- 입력 $\mathbf{x}^i$ 에 weight $\mathbf{U}^1$ 을 곱한 결과와, 입력 $\mathbf{h}^{i-1}$ 에 $\mathbf{U}^3$ 을 곱한 결과를 더해서, time $i$ 의 hidden vector $\mathbf{h}^{i}$ 를 만든다.

$$ \mathbf{h}^i = {\tau}_{1}(\mathbf{U}^3 \mathbf{h}^{i-1} + \mathbf{U}^1 \mathbf{x}^i) $$

- $\mathbf{h}^{i}$ 에 $\mathbf{U}^2$ 를 곱하고 activation function ${\tau}_{2}$ 를 곱해서, output $\mathbf{o}^i$ 를 출력한다.

$$ \mathbf{o}^i = {\tau}_{2}(\mathbf{U}^2 \mathbf{h}^i) $$

이처럼 순차적으로 연산이 진행되고, 지난 hidden state로 다음 hidden state를 만들기 때문에 계속 이전 정보와 상호작용하게 된다. 예를 들어 세 번째 단어라면 첫 번째 단어와 두 번째 단어에 대한 정보와 상호작용한다.

---

### 11.2.1 LSTM

기존 RNN은 $1, 2, ..., i$ 로 가며 오래된 단어의 정보가 희미해지는데, 종종 앞쪽 단어와 멀리 뒤에 있는 단어가 밀접하게 상호작용해야 하는 경우가 있다.(**long-range dependency**)

**LSTM**(Long Short-Term Memory, 장단기 메모리)는 RNN의 한 종류로, 곳곳에 input, output을 열거나 막을 수 있는 gate를 두어서 선별적으로 기억하는 기능을 확보한 모델이다.

![LSTM](images/LSTM.png)

- 여닫는 정도는 학습으로 알아낸 weight에 따라 결정된다.

---