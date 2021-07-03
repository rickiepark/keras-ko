# 엔지니어에게 맞는 케라스 소개

**Author:** [fchollet](https://twitter.com/fchollet)<br>
**Date created:** 2020/04/01<br>
**Last modified:** 2020/04/28<br>
**Description:** 케라스로 실전 머신러닝 솔루션을 만들기 위해 알아야 할 모든 것.


<img class="k-inline-icon" src="https://colab.research.google.com/img/colab_favicon.ico"/> [**코랩에서 보기**](https://colab.research.google.com/github/adsidelab/keras-ko/blob/master/guides/ipynb/intro_to_keras_for_engineers.ipynb)  <span class="k-dot">•</span><img class="k-inline-icon" src="https://github.com/favicon.ico"/> [**깃허브 소스**](https://github.com/adsidelab/keras-ko/blob/master/guides/intro_to_keras_for_engineers.py)



---
## 설정


```python
import numpy as np
import tensorflow as tf
from tensorflow import keras
```

---
## 소개

케라스로 제품에 딥러닝을 적용하고 싶은 머신러닝 엔지니어인가요? 이 가이드에서 케라스 API의 핵심 부분을 소개하겠습니다.

이 가이드에서 다음 방법을 배울 수 있습니다:

- 모델을 훈련하기 전에 데이터를 준비합니다(넘파이 배열이나 `tf.data.Dataset` 객체로 변환합니다).
- 데이터를 전처리합니다. 예를 들면 특성 정규화나 어휘 사전 구축합니다.
- 케라스 함수형 API로 데이터에서 예측을 만드는 모델을 구축합니다.
- 케라스의 기본 `fit()` 메서드로 체크포인팅(checkpointing), 성능 지표 모니터링, 내결함성(fault tolerance)을 고려한 모델을 훈련합니다.
- 테스트 데이터에서 모델 평가하고 새로운 데이터에서 모델을 사용해 추론하는 방법.
- GAN과 같은 모델을 만들기 위해 `fit()` 메서드를 커스터마이징합니다.
- 여러 개의 GPU를 사용해 훈련 속도를 높입니다.
- 하이퍼파라미터를 튜닝하여 모델의 성능을 높입니다.

이 문서 끝에 다음 주제에 대한 엔드-투-엔드 예제 링크를 소개하겠습니다:

- 이미지 분류
- 텍스트 분류
- 신용 카드 부정 거래 감지


---
## 데이터 적재와 전처리

신경망은 텍스트 파일, JPEG 이미지 파일, CSV 파일 같은 원시 데이터를 그대로 처리하지 않습니다.
신경망은 **벡터화**되거나 **표준화**된 표현을 처리합니다.

- 텍스트 파일을 문자열 텐서로 읽어 단어로 분리합니다. 마지막에 단어를 정수 텐서로 인덱싱하고 변환합니다.
- 이미지를 읽어 정수 텐서로 디코딩합니다. 그다음 부동 소수로 변환하고 (보통 0에서 1사이) 작은 값으로 정규화합니다.
- CSV 데이터를 파싱하여 정수 특성은 부동 소수 텐서로 변환하고, 범주형 특성은 정수 텐서로 인덱싱하고 변환합니다.
그다음 일반적으로 각 특성을 평균 0, 단위 분산으로 정규화합니다.

먼저 데이터를 적재해 보죠.

---
## 데이터 적재

케라스 모델은 세 종류의 입력을 받습니다:

- **넘파이(NumPy) 배열**. 사이킷런(Scikit-Learn)이나 다른 파이썬 라이브러리와 비슷합니다.
데이터 크기가 메모리에 맞을 때 좋습니다.
- **[텐서플로 `Dataset` 객체](https://www.tensorflow.org/guide/data)**.
데이터가 메모리보다 커서 디스크나 분산 파일 시스템에서 스트림으로 읽어야할 때 적합한 고성능 방식입니다.
- **파이썬 제너레이터(generator)**. 배치 데이터를 만듭니다(`keras.utils.Sequence` 클래스의
사용자 정의 서브클래스와 비슷합니다).

모델을 훈련하기 전에 이런 포맷 중에 하나로 데이터를 준비해야 합니다.
데이터셋이 크고 GPU에서 훈련한다면 `Dataset` 객체를 사용하는 것이 좋습니다.
다음 같이 성능에 아주 중요한 기능을 제공하기 때문입니다:

- GPU가 바쁠 때 CPU에서 데이터를 비동기적으로 전처리하고 큐에 버퍼링합니다.
- GPU 메모리에 데이터를 프리페치(prefetch)하여 GPU에서 이전 배치에 대한 처리가 끝나는대로 즉시 사용할 수 있습니다.
이를 통해 GPU를 최대로 활용할 수 있습니다.

케라스는 디스크에 있는 원시 데이터를 `Dataset`으로 변환해 주는
여러 유틸리티를 제공합니다(**옮긴이_** 아래 함수는 아직 tf-nightly 패키지에서만 제공합니다):

- `tf.keras.preprocessing.image_dataset_from_directory`는 클래스별로 폴더에 나뉘어 있는 이미지 파일을
레이블된 이미지 텐서 데이터셋으로 변환합니다.
- `tf.keras.preprocessing.text_dataset_from_directory`는 텍스트 파일에 대해 동일한 작업을 수행합니다.

또한 텐서플로의 `tf.data`는 CSV 파일에서 정형화된 데이터를 로드하는
`tf.data.experimental.make_csv_dataset`와 같은 유틸리티를 제공합니다.

**예제: 디스크에 있는 이미지 파일에서 레이블된 데이터셋 만들기**

다음처럼 클래스별로 각기 다른 폴더에 이미지 파일이 들어 있다고 가정해 보죠:

```
main_directory/
...class_a/
......a_image_1.jpg
......a_image_2.jpg
...class_b/
......b_image_1.jpg
......b_image_2.jpg
```

그럼 다음처럼 쓸 수 있습니다:

```python
# 데이터셋을 만듭니다.
dataset = keras.preprocessing.image_dataset_from_directory(
  'path/to/main_directory', batch_size=64, image_size=(200, 200))

# 예시를 위해 데이터셋의 배치를 순회합니다.
for data, labels in dataset:
   print(data.shape)  # (64, 200, 200, 3)
   print(data.dtype)  # float32
   print(labels.shape)  # (64,)
   print(labels.dtype)  # int32
```

샘플의 레이블은 폴더의 알파벳 순서대로 매겨집니다.
매개변수를 사용해 명시적으로 지정할 수도 있습니다.
예를 들어 `class_names=['class_a', 'class_b']`라고 쓸 경우 다
클래스 레이블 `0`은 `class_a`가 되고 `1`은 `class_b`가 됩니.

**예제: 디스크에 있는 텍스트 파일에서 레이블된 데이터셋 만들기**

텍스트도 비슷합니다. 클래스별로 다른 폴더에 `.txt` 파일이 있다면 다음과 같이 쓸 수 있습니다:

```python
dataset = keras.preprocessing.text_dataset_from_directory(
  'path/to/main_directory', batch_size=64)

# 예시를 위해 데이터셋의 배치를 순회합니다.
for data, labels in dataset:
   print(data.shape)  # (64,)
   print(data.dtype)  # string
   print(labels.shape)  # (64,)
   print(labels.dtype)  # int32
```



---
## 케라스를 사용한 데이터 전처리

데이터가 문자열/정수/실수 넘파이 배열이거나
문자열/정수/실수 텐서의 배치를 반환하는 `Dataset` 객체(또는 파이썬 제너레이터)로 준비되었다면
이제 데이터를 **전처리**할 차례입니다. 이 과정은 다음과 같은 작업을 의미합니다:

- 문자열 데이터를 토큰으로 나누고 인덱싱합니다.
- 특성을 정규화합니다.
- 데이터를 작은 값으로 스케일을 조정합니다(일반적으로 신경망의 입력은 0에 가까워야 합니다.
평균이 0이고 분산이 1이거나 `[0, 1]` 범위의 데이터를 기대합니다).

### 이상적인 머신러닝 모델은 엔드-투-엔드 모델입니다

일반적으로 가능하면 데이터 전처리를 별도의 파이프라인으로 만들지 않고 **모델의 일부**가 되도록 해야 합니다.
별도의 데이터 전처리 파이프라인은 모델을 제품에 투입할 때 이식하기 어렵게 만들기 때문입니다.
텍스트 처리를 하는 모델을 생각해 보죠.
이 모델은 특별한 토큰화 알고리즘과 어휘 사전 인덱스를 사용합니다.
이 모델을 모바일 앱이나 자바스크립트 앱에 적용할 때 해당 언어로 동일한 전처리 과정을 다시 구현해야 합니다.
이는 매우 위험한 작업입니다.
원래 파이프라인과 다시 만든 파이프라인 사이에 작은 차이가 모델을 완전히 망가뜨리거나
성능을 크게 낮출 수 있기 때문입니다.

전처리를 포함한 엔드-투-엔드(end-to-end) 모델로 만들 수 있다면 훨씬 간단합니다.
**이상적인 모델은 가능한 원시 데이터에 가까운 입력을 기대해야 합니다.
이미지 모델은 `[0, 255]` 사이의 RGB 픽셀 값을 기대합니다.
텍스트 모델은 `utf-8` 문자열을 기대합니다.**
따라서 이 모델을 사용하는 애플리케이션은 전처리 파이프라인에 대해 신경쓸 필요가 없습니다.

### 케라스 전처리 층 사용하기

케라스에서는 **전처리 층**으로 모델에서 데이터 전처리를 수행합니다.
다음과 같은 기능을 제공합니다:

- `TextVectorization` 층으로 텍스트 원시 문자열을 벡터화합니다.
- `Normalization` 층으로 특성을 정규화합니다.
- 이미지 스케일 조정, 자르기, 데이터 증식을 수행합니다.

케라스 전처리 층을 사용할 때 가장 큰 장점은 훈련하는 중간이나 훈련이 끝난 후에
이 층을 **모델에 직접 포함하여** 모델의 이식성을 높일 수 있다는 점입니다.

일부 전처리 층은 상태를 가집니다:

- `TextVectorization`는 정수 인덱스로 매핑된 단어나 토큰을 저장합니다.
- `Normalization`는 특성의 평균과 분산을 저장합니다.

훈련 데이터의 샘플(또는 전체 데이터)을 사용해 `layer.adapt(data)`를 호출하면 전처리 층의 상태가 반환됩니다.


**예제: 문자열을 정수 단어 인덱스의 시퀀스로 변환하기**



```python
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization

# dtype이 `string`인 예제 훈련 데이터.
training_data = np.array([["This is the 1st sample."], ["And here's the 2nd sample."]])

# TextVectorization 층 객체를 만듭니다.
# 정수 토큰 인덱스 또는 토큰의 밀집 표현(예를 들어 멀티-핫(multi-hot)이나 TF-IDF)을 반환할 수 있습니다.
# 텍스트 표준화와 텍스트 분할 알고리즘을 완전히 커스터마이징할 수 있습니다.
vectorizer = TextVectorization(output_mode="int")

# 배열이나 데이터셋에 대해 `adapt` 메서드를 호출하면 어휘 인덱스를 생성합니다.
# 이 어휘 인덱스는 새로운 데이터를 처리할 때 재사용됩니다.
vectorizer.adapt(training_data)

# `adapt`를 호출하고 나면 이 메서드가 데이터에서 보았던 n-그램(n-gram)을 인코딩할 수 있습니다.
# 본적 없는 n-그램은 OOB(out-of-vocabulary) 토큰으로 인코딩됩니다.
integer_data = vectorizer(training_data)
print(integer_data)
```

<div class="k-default-codeblock">
```
tf.Tensor(
[[4 5 2 9 3]
 [7 6 2 8 3]], shape=(2, 5), dtype=int64)

```
</div>
**예제: 문자열을 원-핫 인코딩된 바이그램(bigram) 시퀀스로 변환하기**


```python
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization

# dtype이 `string`인 예제 훈련 데이터.
training_data = np.array([["This is the 1st sample."], ["And here's the 2nd sample."]])

# TextVectorization 층 객체를 만듭니다.
# 정수 토큰 인덱스 또는 토큰의 밀집 표현(예를 들어 멀티-핫(multi-hot)이나 TF-IDF)을 반환할 수 있습니다.
# 텍스트 표준화와 텍스트 분할 알고리즘을 완전히 커스터마이징할 수 있습니다.
vectorizer = TextVectorization(output_mode="binary", ngrams=2)

# 배열이나 데이터셋에 대해 `adapt` 메서드를 호출하면 어휘 인덱스를 생성합니다.
# 이 어휘 인덱스는 새로운 데이터를 처리할 때 재사용됩니다.
vectorizer.adapt(training_data)

# `adapt`를 호출하고 나면 이 메서드가 데이터에서 보았던 n-그램(n-gram)을 인코딩할 수 있습니다.
# 본적 없는 n-그램은 OOB(out-of-vocabulary) 토큰으로 인코딩됩니다.
integer_data = vectorizer(training_data)
print(integer_data)
```

<div class="k-default-codeblock">
```
tf.Tensor(
[[0. 1. 1. 1. 1. 0. 1. 1. 1. 0. 0. 0. 0. 0. 0. 1. 1.]
 [0. 1. 1. 0. 0. 1. 0. 0. 0. 1. 1. 1. 1. 1. 1. 0. 0.]], shape=(2, 17), dtype=float32)

```
</div>
**예제: 특성 정규화**


```python
from tensorflow.keras.layers.experimental.preprocessing import Normalization

# [0, 255] 사이의 값을 가진 예제 이미지 데이터
training_data = np.random.randint(0, 256, size=(64, 200, 200, 3)).astype("float32")

normalizer = Normalization(axis=-1)
normalizer.adapt(training_data)

normalized_data = normalizer(training_data)
print("분산: %.4f" % np.var(normalized_data))
print("평균: %.4f" % np.mean(normalized_data))
```

<div class="k-default-codeblock">
```
분산: 1.0000
평균: 0.0000

```
</div>
**예제: 이미지 스케일 조정과 자르기**

`Rescaling` 층과 `CenterCrop` 층은 상태가 없습니다.
따라서 `adapt()` 메서드를 호출할 필요가 없습니다.


```python
from tensorflow.keras.layers.experimental.preprocessing import CenterCrop
from tensorflow.keras.layers.experimental.preprocessing import Rescaling

# [0, 255] 사이의 값을 가진 예제 이미지 데이터
training_data = np.random.randint(0, 256, size=(64, 200, 200, 3)).astype("float32")

cropper = CenterCrop(height=150, width=150)
scaler = Rescaling(scale=1.0 / 255)

output_data = scaler(cropper(training_data))
print("크기:", output_data.shape)
print("최소:", np.min(output_data))
print("최대:", np.max(output_data))
```

<div class="k-default-codeblock">
```
크기: (64, 150, 150, 3)
최소: 0.0
최대: 1.0

```
</div>
---
## 케라스 함수형 API로 모델을 만들기

"층"은 (위의 스케일 조정이나 자르기처럼) 단순한 입력-출력 변환입니다.
예를 들어 다음은 입력을 16차원 특성 공간으로 매핑하는 선형 변환 층입니다:

```python
dense = keras.layers.Dense(units=16)
```

"모델"은 층의 유향 비순환 그래프(directed acyclic graph)입니다.
모델을 여러 하위 층을 감싸고 있고 데이터에 노출되어 훈련할 수 있는 "큰 층"으로 생각할 수 있습니다.

케라스 모델을 만들 때 가장 강력하고 널리 사용하는 방법은 함수형 API(Functional API)입니다.
함수형 API로 모델을 만들려면 먼저 입력의 크기(그리고 선택적으로 dtype)를 지정해야 합니다.
입력 차원이 변경될 수 있으면 `None`으로 지정합니다.
예를 들어 200x200 RGB 이미지의 입력 크기는 `(200, 200, 3)`로 지정하고
임의의 크기를 가진 RGB 이미지의 입력 크기는 `(None, None, 3)`으로 지정합니다.


```python
# 임의의 크기를 가진 RGB 이미지 입력을 사용한다고 가정해 보죠.
inputs = keras.Input(shape=(None, None, 3))
```

입력을 정의한 후 이 입력에서 최종 출력까지 층 변환을 연결합니다:


```python
from tensorflow.keras import layers

# 150x150 중앙 부분을 오려냅니다.
x = CenterCrop(height=150, width=150)(inputs)
# [0, 1] 사이로 이미지 스케일을 조정합니다.
x = Rescaling(scale=1.0 / 255)(x)

# 합성곱과 풀링 층을 적용합니다.
x = layers.Conv2D(filters=32, kernel_size=(3, 3), activation="relu")(x)
x = layers.MaxPooling2D(pool_size=(3, 3))(x)
x = layers.Conv2D(filters=32, kernel_size=(3, 3), activation="relu")(x)
x = layers.MaxPooling2D(pool_size=(3, 3))(x)
x = layers.Conv2D(filters=32, kernel_size=(3, 3), activation="relu")(x)

# 전역 풀링 층을 적용하여 일렬로 펼친 특성 벡터를 얻습니다.
x = layers.GlobalAveragePooling2D()(x)

# 그 다음에 분류를 위해 밀집 층을 추가합니다.
num_classes = 10
outputs = layers.Dense(num_classes, activation="softmax")(x)
```

입력을 출력으로 바꾸는 층의 유향 비순환 그래프를 정의하고 나서 `Model` 객체를 만듭니다:


```python
model = keras.Model(inputs=inputs, outputs=outputs)
```

이 모델은 기본적으로 큰 층처럼 동작합니다. 다음처럼 배치 데이터에서 모델을 호출할 수 있습니다:


```python
data = np.random.randint(0, 256, size=(64, 200, 200, 3)).astype("float32")
processed_data = model(data)
print(processed_data.shape)
```

<div class="k-default-codeblock">
```
(64, 10)

```
</div>
모델의 각 단계에서 데이터가 어떻게 변환되는지 요약 정보를 출력하면 디버깅에 도움이 됩니다.

각 층에 표시되는 출력 크기는 **배치 크기**를 포함합니다.
배치 크기가 `None`이면 이 모델이 어떤 크기의 배치도 처리할 수 있다는 의미입니다.


```python
model.summary()
```

<div class="k-default-codeblock">
```
Model: "functional_1"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
input_1 (InputLayer)         [(None, None, None, 3)]   0         
_________________________________________________________________
center_crop_1 (CenterCrop)   (None, 150, 150, 3)       0         
_________________________________________________________________
rescaling_1 (Rescaling)      (None, 150, 150, 3)       0         
_________________________________________________________________
conv2d (Conv2D)              (None, 148, 148, 32)      896       
_________________________________________________________________
max_pooling2d (MaxPooling2D) (None, 49, 49, 32)        0         
_________________________________________________________________
conv2d_1 (Conv2D)            (None, 47, 47, 32)        9248      
_________________________________________________________________
max_pooling2d_1 (MaxPooling2 (None, 15, 15, 32)        0         
_________________________________________________________________
conv2d_2 (Conv2D)            (None, 13, 13, 32)        9248      
_________________________________________________________________
global_average_pooling2d (Gl (None, 32)                0         
_________________________________________________________________
dense (Dense)                (None, 10)                330       
=================================================================
Total params: 19,722
Trainable params: 19,722
Non-trainable params: 0
_________________________________________________________________

```
</div>
함수형 API는 여러 개의 입력(예를 들어 이미지와 메타데이터)이나
여러 개의 출력(예를 들어 이미지 클래스와 클릭 확률을 예측)을 사용하는 모델도 쉽게 만들 수 있습니다.
이에 대해 더 자세한 정보는 [함수형 API 가이드](/guides/functional_api/)를 참고하세요.

---
## `fit()`으로 모델 훈련하기

지금까지 다음 내용을 배웠습니다:

- 데이터를 준비하는 방법(예를 들어 넘파이 배열이나 `tf.data.Dataset` 객체)
- 데이터를 처리할 모델을 만드는 방법

다음 단계는 데이터에서 모델을 훈련하는 것입니다.
`Model` 클래스는 `fit()` 메서드에서 훈련을 반복합니다.
이 메서드는 `Dataset` 객체, 배치 데이터를 반환하는 파이썬 제너레이터, 넘파이 배열을 받습니다.

`fit()` 메서드를 호출하기 전에 옵티마이저와 손실 함수를 지정해야 합니다(여러분이
이런 개념을 이미 알고 있다고 가정하겠습니다). 이것이 `compile()` 단계입니다:

```python
model.compile(optimizer=keras.optimizers.RMSprop(learning_rate=1e-3),
              loss=keras.losses.CategoricalCrossentropy())
```

손실 함수와 옵티마이저는 문자열로 지정할 수 있습니다(기본 생성자 매개변수가 사용됩니다):


```python
model.compile(optimizer='rmsprop', loss='categorical_crossentropy')
```

모델이 컴파일되면 데이터에서 "훈련"을 시작할 수 있습니다.
다음은 넘파이 데이터를 사용해 모델을 훈련하는 예입니다:

```python
model.fit(numpy_array_of_samples, numpy_array_of_labels,
          batch_size=32, epochs=10)
```

데이터 외에도 두 개의 핵심 매개변수를 지정해야 합니다.
`batch_size`와 에포크 횟수(데이터를 반복할 횟수)입니다.
여기에서는 데이터를 32개 샘플씩 배치로 나누어 사용하고 훈련하는 동안 전체 데이터에 대해 10번 반복합니다.

다음은 데이터셋을 사용해 모델을 훈련하는 예입니다:

```python
model.fit(dataset_of_samples_and_labels, epochs=10)
```

데이터셋은 배치 데이터를 반환하기 때문에 배치 크기를 지정할 필요가 없습니다.

MNIST 숫자를 분류하는 작은 예제 모델을 만들어 보겠습니다:


```python
# 넘파이 배열로 데이터를 가져옵니다.
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# 간단한 모델을 만듭니다.
inputs = keras.Input(shape=(28, 28))
x = layers.experimental.preprocessing.Rescaling(1.0 / 255)(inputs)
x = layers.Flatten()(x)
x = layers.Dense(128, activation="relu")(x)
x = layers.Dense(128, activation="relu")(x)
outputs = layers.Dense(10, activation="softmax")(x)
model = keras.Model(inputs, outputs)
model.summary()

# 모델을 컴파일합니다.
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy")

# 넘파이 데이터에서 1 에포크 동안 모델을 훈련합니다.
batch_size = 64
print("넘파이 데이터에서 훈련하기")
history = model.fit(x_train, y_train, batch_size=batch_size, epochs=1)

# 데이터셋을 사용해 1 에포크 동안 모델을 훈련합니다.
dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(batch_size)
print("데이터셋에서 훈련하기")
history = model.fit(dataset, epochs=1)
```

<div class="k-default-codeblock">
```
Model: "functional_3"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
input_2 (InputLayer)         [(None, 28, 28)]          0         
_________________________________________________________________
rescaling_2 (Rescaling)      (None, 28, 28)            0         
_________________________________________________________________
flatten (Flatten)            (None, 784)               0         
_________________________________________________________________
dense_1 (Dense)              (None, 128)               100480    
_________________________________________________________________
dense_2 (Dense)              (None, 128)               16512     
_________________________________________________________________
dense_3 (Dense)              (None, 10)                1290      
=================================================================
Total params: 118,282
Trainable params: 118,282
Non-trainable params: 0
_________________________________________________________________
넘파이 데이터에서 훈련하기
938/938 [==============================] - 1s 838us/step - loss: 0.2675
데이터셋에서 훈련하기
938/938 [==============================] - 1s 875us/step - loss: 0.1197

```
</div>
`fit()` 메서드는 훈련 동안 발생하는 정보를 기록한 "history" 객체를 반환합니다.
`history.history` 딕셔너리는 에포크 순서대로 측정 값을 담고 있습니다(여기에서는
손실 하나와 에포크 횟수가 1이므로 하나의 스칼라 값만 담고 있습니다):


```python
print(history.history)
```

<div class="k-default-codeblock">
```
{'loss': [0.11972887814044952]}

```
</div>
`fit()` 메서드를 사용하는 자세한 방법은
[케라스 내장 메서드를 사용한 훈련과 평가 가이드](/guides/training_with_built_in_methods/)를
참고하세요.

### 성능 지표 기록하기

모델을 훈련하면서 분류 정확도, 정밀도, 재현율, AUC와 같은 지표를 기록할 필요가 있습니다.
이외에도 훈련 데이터에 대한 지표뿐만 아니라 검증 세트에 대한 모니터링도 필요합니다.

**성능 지표 모니터링**

다음처럼 `compile()` 메서드에 측정 지표의 객체 리스트를 전달할 수 있습니다:



```python
model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=[keras.metrics.SparseCategoricalAccuracy(name="acc")],
)
history = model.fit(dataset, epochs=1)
```

<div class="k-default-codeblock">
```
938/938 [==============================] - 1s 898us/step - loss: 0.0831 - acc: 0.9747

```
</div>
**`fit()` 메서드에 검증 데이터 전달하기**

`fit()` 메서드에 검증 데이터를 전달하여 검증 손실과 성능 지표를 모니터링할 수 있습니다.
측정 값은 매 에포크 끝에서 출력됩니다.


```python
val_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(batch_size)
history = model.fit(dataset, epochs=1, validation_data=val_dataset)
```

<div class="k-default-codeblock">
```
938/938 [==============================] - 1s 1ms/step - loss: 0.0566 - acc: 0.9829 - val_loss: 0.1092 - val_acc: 0.9671

```
</div>
### 콜백을 사용해 체크포인트와 다른 여러 작업 수행하기

훈련이 몇 분 이상 지속되면 훈련하는 동안 일정 간격으로 모델을 저장하는 것이 좋습니다.
그러면 훈련 과정에 문제가 생겼을 때 저장된 모델을 사용해 훈련을 다시 시작할 수
있습니다(다중 워커(multi-worker) 분산 훈련일 경우
여러 워커 중 하나가 어느 순간 장애를 일으킬 수 있기 때문에 이 설정이 중요합니다).

케라스의 중요한 기능 중 하나는 `fit()` 메서드에 설정할 수 있는 **콜백**(callback)입니다.
콜백은 훈련하는 동안 각기 다른 지점에서 모델이 호출하는 객체입니다. 예를 들면 다음과 같습니다:

- 각 배치의 시작과 끝에서
- 각 에포크의 시작과 끝에서

콜백을 사용하면 모델의 훈련을 완전하게 제어할 수 있습니다.

콜백을 사용해 일정한 간격으로 모델을 저장할 수 있습니다. 다음이 간단한 예입니다.
에포크가 종료될 때마다 모델을 저장하도록 `ModelCheckpoint` 콜백을 설정하고
파일 이름에 현재 에포크를 포함시켰습니다.

```python
callbacks = [
    keras.callbacks.ModelCheckpoint(
        filepath='path/to/my/model_{epoch}',
        save_freq='epoch')
]
model.fit(dataset, epochs=2, callbacks=callbacks)
```

또 콜백을 사용해 일정한 간격으로 옵티마이저의 학습률을 바꾸거나,
슬랙 봇에 측정 값을 보내거나, 훈련이 완료됐을 때 이메일 알림을 보낼 수 있습니다.

사용할 수 있는 콜백과 사용자 정의 콜백을 작성하는 자세한 방법은
[콜백 API 문서](/api/callbacks/)와
[사용자 정의 콜백 가이드](/guides/writing_your_own_callbacks/)를 참고하세요.

### 텐서보드로 훈련 과정 모니터링하기

케라스 진행 표시줄(progress bar)은 손실과 측정 지표가 시간에 따라
어떻게 변하는지 모니터링하기 편리한 도구는 아닙니다.
더 나은 방법은 실시간 측정 값을 그래프로 (그리고 다른 여러 정보를) 보여주는 웹 애플리케이션인
[텐서보드](https://www.tensorflow.org/tensorboard)(TensorBoard)입니다.

`fit()` 메서드에 텐서보드를 사용하려면 간단하게 텐서보드 로그 저장
디렉토리를 설정한 `keras.callbacks.TensorBoard` 콜백을 전달하면 됩니다:


```python
callbacks = [
    keras.callbacks.TensorBoard(log_dir='./logs')
]
model.fit(dataset, epochs=2, callbacks=callbacks)
```

그다음 텐서보드 프로그램을 실행하여 브라우저에서 저장된 로그를 모니터링할 수 있습니다:

```
tensorboard --logdir=./logs
```

또한 주피터 노트북이나 코랩 노트북에서 모델을 훈련할 때 인라인으로 텐서보드 탭을 실행할 수 있습니다.
[자세한 정보는 여기를 참고하세요](https://www.tensorflow.org/tensorboard/tensorboard_in_notebooks).

### `fit()` 메서드 실행 후 테스트 성능을 평가하고 새로운 데이터에 대해 예측 만들기

모델 훈련을 마치면 `evaluate()` 메서드로 새로운 데이터에 대한 손실과 측정 지표를 평가할 수 있습니다:


```python
loss, acc = model.evaluate(val_dataset)  # 손실과 측정 값을 반환합니다.
print("손실: %.2f" % loss)
print("정확도: %.2f" % acc)
```

<div class="k-default-codeblock">
```
157/157 [==============================] - 0s 741us/step - loss: 0.1092 - acc: 0.9671
손실: 0.11
정확도: 0.97

```
</div>
`predict()` 메서드로 넘파이 배열로 예측(모델에 있는 출력층의 활성화 값)을 만들 수도 있습니다:


```python
predictions = model.predict(val_dataset)
print(predictions.shape)
```

<div class="k-default-codeblock">
```
(10000, 10)

```
</div>
---
## `fit()` 메서드로 사용자 정의 훈련 단계 구현하기

기본적으로 `fit()`은 **지도 학습**을 지원합니다.
다른 종류의 훈련 반복(예를 들면 GAN 훈련 반복)이 필요하면
`Model.train_step()` 메서드를 구현하면 됩니다.
이 메서드는 `fit()` 메서드가 실행되는 동안 반복적으로 호출됩니다.

측정 지표, 콜백 등은 동일하게 작동합니다.

다음은 `fit()` 메서드의 기능을 간단하게 다시 구현한 예입니다.

```python
class CustomModel(keras.Model):
  def train_step(self, data):
    # 데이터를 받아 옵니다.
    # 데이터 구조는 `fit()` 메서드에 전달한 방식과 모델에 따라 다릅니다.
    x, y = data
    with tf.GradientTape() as tape:
      y_pred = self(x, training=True)  # 정방향 계산
      # 손실을 계산합니다.
      # (손실 함수는 `compile()` 메서드에서 설정합니다)
      loss = self.compiled_loss(y, y_pred,
                                regularization_losses=self.losses)
    # 그레이디언트를 계산합니다.
    trainable_vars = self.trainable_variables
    gradients = tape.gradient(loss, trainable_vars)
    # 가중치를 업데이트합니다.
    self.optimizer.apply_gradients(zip(gradients, trainable_vars))
    # 측정 값을 업데이트합니다(손실 지표를 포함합니다).
    self.compiled_metrics.update_state(y, y_pred)
    # 측정 지표 이름과 현재 값을 매핑한 딕셔너리를 반환합니다.
    return {m.name: m.result() for m in self.metrics}

# CustomModel 객체를 만들고 컴파일합니다.
inputs = keras.Input(shape=(32,))
outputs = keras.layers.Dense(1)(inputs)
model = CustomModel(inputs, outputs)
model.compile(optimizer='adam', loss='mse', metrics=[...])

# 보통처럼 `fit()` 메서드를 사용합니다.
model.fit(dataset, epochs=3, callbacks=...)
```

사용자 정의 훈련과 평가에 대한 자세한 내용은 다음 가이드를 참고하세요:
["`fit()` 메서드를 커스터마이징하기"](/guides/customizing_what_happens_in_fit/).

---
## 즉시 실행으로 모델 디버깅하기

사용자 정의 훈련 단계나 층을 만들면 디버깅을 할 필요가 있습니다.
디버깅은 프레임워크에 통합하기 위한 부분입니다.
케라스의 디버깅 작업 흐름은 사용자에게 초점을 맞추어 설계되엇습니다.

기본적으로 케라스 모델은 빠르게 실행하기 위해 매우 최적화된 계산 그래프로 컴파일됩니다.
다시 말해 (`train_step()` 메서드에) 사용자가 작성한 코드와 실제로 실행되는 코드가 다릅니다.
이는 디버깅을 어렵게 만듭니다.

디버깅은 단계별로 수행하는 것이 좋습니다.
코드 여기저기에 `print()` 문을 넣고 연산이 실행된 후에 데이터가 어떻게 변하는지 보고 싶어 합니다.
또는 `pdb`를 사용하고 싶을 것입니다.
**모델을 즉시 실행(eager execution) 모드로 사용하면** 이렇게 할 수 있습니다.
즉시 실행에서는 사용자가 작성한 파이썬 코드가 실행되는 코드가 됩니다.

간단하게 `compile()` 메서드에 `run_eagerly=True`를 전달하면 됩니다:

```python
model.compile(optimizer='adam', loss='mse', run_eagerly=True)
```

물론 모델이 크게 느려진다는 단점이 있습니다.
디버깅이 끝나면 컴파일된 계산 그래프의 장점을 다시 활용하도록 바꾸는 것을 잊지 마세요!

일반적으로 `fit()` 메서드를 디버깅하고 싶을 때 `run_eagerly=True`를 사용합니다.

---
## 다중 GPU로 훈련 속도 높이기

케라스는 `tf.distribute` API를 통해 다중 GPU 훈련과 분산 다중 워커(multi-worker) 훈련을 기본으로 지원합니다.

다중 GPU를 가지고 있다면 모델 훈련에 모두 활용할 수 있습니다:

- `tf.distribute.MirroredStrategy` 객체를 만듭니다.
- 이 분산 정책 스코프(scope) 안에서 모델을 만들고 컴파일합니다.
- 보통과 같이 데이터셋으로 `fit()` 메서드와 `evaluate()` 메서드를 호출합니다.

```python
# MirroredStrategy 객체를 만듭니다.
strategy = tf.distribute.MirroredStrategy()

# 분산 정책 스코프를 시작합니다.
with strategy.scope():
  # 변수를 만드는 작업은 분산 정책 스코프 안에서 수행해야 합니다.
  # 일반적으로 모델 생성과 `compile()` 메서드에 해당합니다.
  model = Model(...)
  model.compile(...)

# 가용한 모든 장치로 모델을 훈련합니다.
train_dataset, val_dataset, test_dataset = get_dataset()
model.fit(train_dataset, epochs=2, validation_data=val_dataset)

# 가용한 모든 장치로 모델을 테스트합니다.
model.evaluate(test_dataset)
```

다중 GPU와 분산 훈련에 관한 자세한 내용은 이 [가이드](/guides/distributed_training/)를 참고하세요.

---
## 온-디바이스 동기 전처리 vs CPU 비동기 전처리

앞서 전처리에 대해 소개했습니다. 이미지 전처리 층(`CenterCrop`과 `Rescaling`)을 모델 안에 직접 넣는 예제도 보았습니다.

GPU 가속을 사용한 특성 정규화나 이미지 증식 같은 온-디바이스(on-device) 전처리가 필요하다면
훈련하는 동안 모델의 일부로 전처리가 수행되는 것이 좋습니다.
하지만 이런 환경에 어울리지 않는 전처리 종류가 있습니다.
예를 들면 `TextVectorization` 층을 사용한 텍스트 전처리입니다.
순차 데이터라는 특징 때문에 CPU에서만 실행할 수 있습니다.
이런 경우 **비동기 전처리**를 사용하는 것이 좋습니다.

비동기 전처리에서는 전처리 연산이 CPU에서 실행되고
GPU가 이전 배치 데이터를 처리하는 동안 전처리된 샘플은 큐(queue)에 저장됩니다.
다음 전처리된 샘플 배치는 GPU 작업이 끝나기 직전에 큐에서 GPU 메모리로 이동합니다(프리페칭(prefetching)).
이런 방식을 통해 전처리에 병목 현상을 방지하고 GPU를 최대로 활용할 수 있습니다.

비동기 전처리를 수행하려면 `dataset.map`을 사용해 전처리 연산을 데이터 파이프라인에 추가하면 됩니다:


```python
# dtype이 `string`인 훈련 데이터.
samples = np.array([["This is the 1st sample."], ["And here's the 2nd sample."]])
labels = [[0], [1]]

# TextVectorization 층 준비.
vectorizer = TextVectorization(output_mode="int")
vectorizer.adapt(samples)

# 비동기 전처리: TextVectorization을 tf.data 파이프라인에 넣습니다.
# 먼저 데이터셋을 만듭니다.
dataset = tf.data.Dataset.from_tensor_slices((samples, labels)).batch(2)
# 샘플에 TextVectorization를 적용합니다.
dataset = dataset.map(lambda x, y: (vectorizer(x), y))
# 2 배치 크기의 버퍼로 프리페치합니다.
dataset = dataset.prefetch(2)

# 이 모델은 정수 시퀀스를 입력으로 기대합니다.
inputs = keras.Input(shape=(None,), dtype="int64")
x = layers.Embedding(input_dim=10, output_dim=32)(inputs)
outputs = layers.Dense(1)(x)
model = keras.Model(inputs, outputs)

model.compile(optimizer="adam", loss="mse", run_eagerly=True)
model.fit(dataset)
```

<div class="k-default-codeblock">
```
1/1 [==============================] - 0s 913us/step - loss: 0.5295

<tensorflow.python.keras.callbacks.History at 0x7fd92008c6d8>

```
</div>
이 모델과 TextVectorization을 모델의 일부로 수행하는 것과 비교해 보죠:


```python
# 이 데이터셋은 문자열 샘플을 반환합니다.
dataset = tf.data.Dataset.from_tensor_slices((samples, labels)).batch(2)

# 이 모델은 문자열을 입력으로 기대합니다.
inputs = keras.Input(shape=(1,), dtype="string")
x = vectorizer(inputs)
x = layers.Embedding(input_dim=10, output_dim=32)(x)
outputs = layers.Dense(1)(x)
model = keras.Model(inputs, outputs)

model.compile(optimizer="adam", loss="mse", run_eagerly=True)
model.fit(dataset)
```

<div class="k-default-codeblock">
```
1/1 [==============================] - 0s 701us/step - loss: 0.5017

<tensorflow.python.keras.callbacks.History at 0x7fd920052b70>

```
</div>
텍스트 모델을 CPU에서 훈련할 때 일반적으로 두 설정 사이에 성능 차이가 없습니다.
하지만 GPU에서 훈련할 때 CPU에서 비동기 버퍼 전처리하고 GPU에서 모델을 수행하는 것이 크게 속도를 높입니다.

훈련이 끝난 후 전처리 층을 포함한 엔드-투-엔드 모델을 내보낼 수 있습니다.
`TextVectorization`이 층이기 때문에 아주 간단합니다:

```python
inputs = keras.Input(shape=(1,), dtype='string')
x = vectorizer(inputs)
outputs = trained_model(x)
end_to_end_model = keras.Model(inputs, outputs)
```

---
## 하이퍼파라미터 튜닝으로 최상의 모델 찾기

작동하는 모델을 만들고 나면 구조, 층 크기 등의 모델 설정을 최적화합니다.
사람의 직관은 더 이상 유효하지 않기 때문에 체계적인 하이퍼파라미터(hyperparameter) 탐색 방법을 사용해야 합니다.

[케라스 튜너(Tuner)](https://keras-team.github.io/keras-tuner/documentation/tuners/)를 사용해
케라스 모델을 위한 최상의 하이퍼파라미터를 찾을 수 있습니다.
간단하게 `fit()` 메서드를 호출하는 것이 전부입니다.

사용하는 방법은 다음과 같습니다.

먼저 모델 정의를 함수로 구현합니다. 이 함수는 `hp` 매개변수 하나를 가집니다.
이 함수 안에서 튜닝하고 싶은 값을 `hp.Int()`나 `hp.Choice()`와 같은 하이퍼파라미터 샘플링 메서드로 바꿉니다:

```python
def build_model(hp):
    inputs = keras.Input(shape=(784,))
    x = layers.Dense(
        units=hp.Int('units', min_value=32, max_value=512, step=32),
        activation='relu'))(inputs)
    outputs = layers.Dense(10, activation='softmax')(x)
    model = keras.Model(inputs, outputs)
    model.compile(
        optimizer=keras.optimizers.Adam(
            hp.Choice('learning_rate',
                      values=[1e-2, 1e-3, 1e-4])),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy'])
    return model
```

이 함수는 컴파일된 모델을 반환해야 합니다.

그다음 최적화 대상과 탐색할 파라미터를 지정하여 튜너 객체를 만듭니다:


```python
import keras_tuner

tuner = keras_tuner.tuners.Hyperband(
  build_model,
  objective='val_loss',
  max_epochs=100,
  max_trials=200,
  executions_per_trial=2,
  directory='my_dir')
```

마지막으로 `Model.fit()`과 같은 매개변수를 받는 `search()` 메서드로 탐색을 시작합니다:

```python
tuner.search(dataset, validation_data=val_dataset)
```

탐색이 끝나면 최상의 모델을 얻을 수 있습니다:

```python
models = tuner.get_best_models(num_models=2)
```

또는 결과를 출력할 수 있습니다:

```python
tuner.results_summary()
```

---
## 엔드-투-엔드 예제

이 가이드에 소개된 개념을 잘 이해하려면 다음의 엔드-투-엔드 예제를 참고하세요:

- [텍스트 분류](/examples/nlp/text_classification_from_scratch/)
- [이미지 분류](/examples/vision/image_classification_from_scratch/)
- [신용 카드 부정 거래 감지](/examples/structured_data/imbalanced_classification/)

---
## 다음에 배울 것들

- [함수형 API](/guides/functional_api/)
- [`fit()` 메서드와 `evaluate()` 메서드의 기능](/guides/training_with_built_in_methods/)
- [콜백](/guides/writing_your_own_callbacks/)
- [사용자 정의 훈련 단계 만들기](/guides/customizing_what_happens_in_fit/)
- [다중 GPU와 분산 훈련](/guides/distributed_training/)
- [전이 학습](/guides/transfer_learning/)
