# 연구자에게 맞는 케라스 소개

**Author:** [fchollet](https://twitter.com/fchollet)<br>
**Date created:** 2020/04/01<br>
**Last modified:** 2020/04/28<br>
**Description:** 케라스와 TF 2.0으로 딥러닝 연구를 하기 위해 알아야 할 모든 것.


<img class="k-inline-icon" src="https://colab.research.google.com/img/colab_favicon.ico"/> [**코랩에서 보기**](https://colab.research.google.com/github/adsidelab/keras-ko/blob/master/guides/ipynb/intro_to_keras_for_researchers.ipynb)  <span class="k-dot">•</span><img class="k-inline-icon" src="https://github.com/favicon.ico"/> [**깃허브 소스**](https://github.com/adsidelab/keras-ko/blob/master/guides/intro_to_keras_for_researchers.py)



---
## 설정


```python
import tensorflow as tf
from tensorflow import keras
```

---
## 소개

머신러닝 연구자인가요?
NeurIPS에 논문을 제출하고 컴퓨터 비전이나 자연어 처리 분야에서 최고의 성능을 달성하려고 하나요?
이 가이드에서 케라스 API의 핵심 개념을 소개하겠습니다.

이 가이드에서 다음 내용을 배울 수 있습니다:

- `Layer` 클래스를 상속하여 층을 만듭니다.
- `GradientTape`으로 그레이디언트(gradient)를 계산하고 저수준 훈련 반복문을 만듭니다.
- `add_loss()` 메서드로 층에서 만든 손실을 기록합니다.
- 저수준 훈련 반복문에서 측정 지표를 기록합니다.
- `tf.function`으로 컴파일하여 실행 속도를 높입니다.
- 훈련 모드나 추론 모드로 층을 실행합니다.
- 케라스 함수형 API

변이형 오토인코더(Variational Autoencoder)와 하이퍼네트워크(Hypernetwork)
두 개의 엔드-투-엔드 연구 예제 통해 실제로 케라스 API를 사용해 보겠습니다.

---
## `Layer` 클래스

`Layer`는 케라스의 기초 추상 클래스입니다.
`Layer`는 상태(가중치)와 (`call` 메서드에서 정의한) 일부 계산을 담고 있습니다.

간단한 층의 예는 다음과 같습니다:


```python

class Linear(keras.layers.Layer):
    """y = w.x + b"""

    def __init__(self, units=32, input_dim=32):
        super(Linear, self).__init__()
        w_init = tf.random_normal_initializer()
        self.w = tf.Variable(
            initial_value=w_init(shape=(input_dim, units), dtype="float32"),
            trainable=True,
        )
        b_init = tf.zeros_initializer()
        self.b = tf.Variable(
            initial_value=b_init(shape=(units,), dtype="float32"), trainable=True
        )

    def call(self, inputs):
        return tf.matmul(inputs, self.w) + self.b

```

`Layer` 클래스 인스턴스를 파이썬 함수처럼 사용할 수 있습니다:


```python
# 층의 객체를 만듭니다.
linear_layer = Linear(units=4, input_dim=2)

# 함수처럼 사용햘 수 있습니다.
# `call` 메서드에 필요한 데이터를 전달하면서 호출합니다.
y = linear_layer(tf.ones((2, 2)))
assert y.shape == (2, 4)
```

(`__init__` 메서드에서 생성한) 가중치 변수는 자동으로 `weights` 속성에 기록됩니다:


```python
assert linear_layer.weights == [linear_layer.w, linear_layer.b]
```

기본으로 내장된 층이 많습니다.
`Dense` 층, `Conv2D` 층, `LSTM` 층이 있고
`Conv3DTranspose`이나 `ConvLSTM2D`와 같은 화려한 층도 있습니다.
가능하면 내장된 기능을 사용하는 것이 좋습니다.

---
## 가중치 생성

`add_weight` 메서드를 사용하면 손쉽게 가중치를 만들 수 있습니다:


```python

class Linear(keras.layers.Layer):
    """y = w.x + b"""

    def __init__(self, units=32):
        super(Linear, self).__init__()
        self.units = units

    def build(self, input_shape):
        self.w = self.add_weight(
            shape=(input_shape[-1], self.units),
            initializer="random_normal",
            trainable=True,
        )
        self.b = self.add_weight(
            shape=(self.units,), initializer="random_normal", trainable=True
        )

    def call(self, inputs):
        return tf.matmul(inputs, self.w) + self.b


# 층의 객체를 만듭니다.
linear_layer = Linear(4)

# `build(input_shape)`을 호출하고 가중치를 만듭니다.
y = linear_layer(tf.ones((2, 2)))
```

---
## 그레이디언트

`GradientTape` 컨택스트 안에서 층을 호출하면 자동으로 층 가중치의 그레이디언트를 계산합니다.
이 그레이디언트를 사용해 옵티마이저 객체를 사용하거나 수동으로 층의 가중치를 업데이트할 수 있습니다.
물론 필요하면 업데이트하기 전에 그레이디언트를 수정할 수 있습니다.


```python
# 데이터셋을 준비합니다.
(x_train, y_train), _ = tf.keras.datasets.mnist.load_data()
dataset = tf.data.Dataset.from_tensor_slices(
    (x_train.reshape(60000, 784).astype("float32") / 255, y_train)
)
dataset = dataset.shuffle(buffer_size=1024).batch(64)

# 10개의 유닛(unit)을 가진 (위에서 정의한) 선형 층의 객체를 만듭니다.
linear_layer = Linear(10)

# 정수 타깃을 기대하는 로지스틱 손실 함수 객체를 만듭니다.
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

# 옵티마이저 객체를 만듭니다.
optimizer = tf.keras.optimizers.SGD(learning_rate=1e-3)

# 데이터셋의 배치를 순회합니다.
for step, (x, y) in enumerate(dataset):

    # GradientTape을 시작합니다.
    with tf.GradientTape() as tape:

        # 정방향 계산을 수행합니다.
        logits = linear_layer(x)

        # 배치의 손실을 계산합니다.
        loss = loss_fn(y, logits)

    # 손실에 대한 가중치의 그레이디언트를 얻습니다.
    gradients = tape.gradient(loss, linear_layer.trainable_weights)

    # 선형 층의 가중치를 업데이트합니다.
    optimizer.apply_gradients(zip(gradients, linear_layer.trainable_weights))

    # 로깅
    if step % 100 == 0:
        print("스텝:", step, "손실:", float(loss))
```

<div class="k-default-codeblock">
```
스텝: 0 손실: 2.3961868286132812
스텝: 100 손실: 2.2662627696990967
스텝: 200 손실: 2.1709532737731934
스텝: 300 손실: 2.078907012939453
스텝: 400 손실: 1.9610252380371094
스텝: 500 손실: 1.9671581983566284
스텝: 600 손실: 1.8129470348358154
스텝: 700 손실: 1.7717030048370361
스텝: 800 손실: 1.8067395687103271
스텝: 900 손실: 1.524446725845337

```
</div>
---
## 훈련되는 가중치와 훈련 안되는 가중치

층은 훈련되는 가중치 또는 훈련 안되는 가중치를 만듭니다.
각각 `trainable_weights`와 `non_trainable_weights` 속성으로 참조할 수 있습니다.
다음은 훈련 안되는 가중치를 가진 층입니다.


```python

class ComputeSum(keras.layers.Layer):
    """입력의 합을 반환합니다."""

    def __init__(self, input_dim):
        super(ComputeSum, self).__init__()
        # Create a non-trainable weight.
        self.total = tf.Variable(initial_value=tf.zeros((input_dim,)), trainable=False)

    def call(self, inputs):
        self.total.assign_add(tf.reduce_sum(inputs, axis=0))
        return self.total


my_sum = ComputeSum(2)
x = tf.ones((2, 2))

y = my_sum(x)
print(y.numpy())  # [2. 2.]

y = my_sum(x)
print(y.numpy())  # [4. 4.]

assert my_sum.weights == [my_sum.total]
assert my_sum.non_trainable_weights == [my_sum.total]
assert my_sum.trainable_weights == []
```

<div class="k-default-codeblock">
```
[2. 2.]
[4. 4.]

```
</div>
---
## 층을 가진 층

층은 재귀적으로 중첩되어 더 큰 연산 블록을 구성할 수 있습니다.
각 층은 하위 층의 가중치를 탐색합니다(훈련되는 가중치와 훈련 안되는 가중치 모두).


```python
# 위에서 정의한 `build` 메서드를 가진
# Linear 클래스를 재사용해 보죠.


class MLP(keras.layers.Layer):
    """Linear 층을 단순하게 쌓습니다."""

    def __init__(self):
        super(MLP, self).__init__()
        self.linear_1 = Linear(32)
        self.linear_2 = Linear(32)
        self.linear_3 = Linear(10)

    def call(self, inputs):
        x = self.linear_1(inputs)
        x = tf.nn.relu(x)
        x = self.linear_2(x)
        x = tf.nn.relu(x)
        return self.linear_3(x)


mlp = MLP()

# 처음 `mlp` 객체를 호출하면 가중치를 만듭니다.
y = mlp(tf.ones(shape=(3, 64)))

# 재귀적으로 가중치를 탐색합니다.
assert len(mlp.weights) == 6
```

위에서 직접 만든 MLP 클래스는 다음처럼 내장된 클래스로 만든 것과 동일합니다:


```python
mlp = keras.Sequential(
    [
        keras.layers.Dense(32, activation=tf.nn.relu),
        keras.layers.Dense(32, activation=tf.nn.relu),
        keras.layers.Dense(10),
    ]
)
```

---
## 층이 만든 손실 탐색하기

층은 정방향 계산 동안 `add_loss()` 메서드로 손실를 생성할 수 있습니다.
특히 규제 손실을 다룰 때 유용합니다.
하위 층이 만든 손실은 부모 층이 재귀적으로 탐색합니다.
Layers can create losses during the forward pass via the `add_loss()` method.
This is especially useful for regularization losses.
The losses created by sublayers are recursively tracked by the parent layers.

활성화 규제 손실을 만드는 층입니다:Here's a layer that creates an activity regularization loss:


```python

class ActivityRegularization(keras.layers.Layer):
    """활성화 희소 규제 손실을 만드는 층입니다."""

    def __init__(self, rate=1e-2):
        super(ActivityRegularization, self).__init__()
        self.rate = rate

    def call(self, inputs):
        # `add_loss`를 사용해 입력에 기반한 규제 손실을 만듭니다.
        self.add_loss(self.rate * tf.reduce_sum(inputs))
        return inputs

```

이 층을 사용하는 모든 모델은 이 규제 손실을 기록합니다:


```python
# MLP 블록에 이 손실 층을 사용해 보죠.


class SparseMLP(keras.layers.Layer):
    """희소 규제 손실을 사용하고 Linear 층을 쌓은 모델."""

    def __init__(self):
        super(SparseMLP, self).__init__()
        self.linear_1 = Linear(32)
        self.regularization = ActivityRegularization(1e-2)
        self.linear_3 = Linear(10)

    def call(self, inputs):
        x = self.linear_1(inputs)
        x = tf.nn.relu(x)
        x = self.regularization(x)
        return self.linear_3(x)


mlp = SparseMLP()
y = mlp(tf.ones((10, 10)))

print(mlp.losses)  # 하나의 float32 스칼라를 담은 리스트
```

<div class="k-default-codeblock">
```
[<tf.Tensor: shape=(), dtype=float32, numpy=0.32833752>]

```
</div>
이 손실은 정방향 계산이 시작될 때마다 최상위 층에 의해 초기화됩니다. 즉 누적되지 않습니다.
`layer.losses`는 항상 마지막 정방향 계산에서 만든 손실만 가지고 있습니다.
일반적으로 훈련 반복문을 만들 때 그레이디언트를 계산하기 전에 이 손실을 더합니다.


```python
# *마지막* 정방향 계산의 손실이 저장됩니다.
mlp = SparseMLP()
mlp(tf.ones((10, 10)))
assert len(mlp.losses) == 1
mlp(tf.ones((10, 10)))
assert len(mlp.losses) == 1  # 누적되지 않습니다.

# 훈련 반복문에서 어떻게 이 손실을 사용하는지 알아보죠.

# 데이터셋을 준비합니다.
(x_train, y_train), _ = tf.keras.datasets.mnist.load_data()
dataset = tf.data.Dataset.from_tensor_slices(
    (x_train.reshape(60000, 784).astype("float32") / 255, y_train)
)
dataset = dataset.shuffle(buffer_size=1024).batch(64)

# 새로운 MLP
mlp = SparseMLP()

# 손실과 옵티마이저
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
optimizer = tf.keras.optimizers.SGD(learning_rate=1e-3)

for step, (x, y) in enumerate(dataset):
    with tf.GradientTape() as tape:

        # 정방향 계산
        logits = mlp(x)

        # 이 배치에 대한 외부 손실 값
        loss = loss_fn(y, logits)

        # 정방향 계산 동안 만들어진 손실을 더합니다.
        loss += sum(mlp.losses)

        # 손실에 대한 가중치의 그레이디언트를 구합니다.
        gradients = tape.gradient(loss, mlp.trainable_weights)

    # 선형 층의 가중치를 업데이트합니다.
    optimizer.apply_gradients(zip(gradients, mlp.trainable_weights))

    # 로깅.
    if step % 100 == 0:
        print("스텝:", step, "손실:", float(loss))
```

<div class="k-default-codeblock">
```
스텝: 0 손실: 6.053885459899902
스텝: 100 손실: 2.6189463138580322
스텝: 200 손실: 2.4236607551574707
스텝: 300 손실: 2.366830587387085
스텝: 400 손실: 2.356532335281372
스텝: 500 손실: 2.355611801147461
스텝: 600 손실: 2.348114013671875
스텝: 700 손실: 2.3352129459381104
스텝: 800 손실: 2.3287711143493652
스텝: 900 손실: 2.3179738521575928

```
</div>
---
## 훈련 지표 기록하기

케라스는 `tf.keras.metrics.AUC`나 `tf.keras.metrics.PrecisionAtRecall` 같이
다양한 측정 지표를 기본으로 제공합니다.
몇 줄의 코드로 사용자 정의 지표를 쉽게 만들 수도 있습니다.

사용자 정의 훈련 반복문에서 지표를 사용하는 방법은 다음과 같습니다:

- 측정 지표 객체를 만듭니다. 예를 들면 `metric = tf.keras.metrics.AUC()`
- 각 배치 데이터에 대해 `metric.udpate_state(targets, predictions)` 메서드를 호출합니다.
- `metric.result()`로 결과를 얻습니다.
- `metric.reset_states()`를 사용해 에포크 끝이나 평가를 시작할 때 지표의 상태를 초기화합니다.

다음은 간단한 예입니다:


```python
# 측정 지표 객체를 만듭니다.
accuracy = tf.keras.metrics.SparseCategoricalAccuracy()

# 층, 손실, 옵티마이저를 준비합니다.
model = keras.Sequential(
    [
        keras.layers.Dense(32, activation="relu"),
        keras.layers.Dense(32, activation="relu"),
        keras.layers.Dense(10),
    ]
)
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)

for epoch in range(2):
    # 데이터셋의 배치에 대해 반복합니다.
    for step, (x, y) in enumerate(dataset):
        with tf.GradientTape() as tape:
            logits = model(x)
            # 이 배치의 손실을 계산합니다.
            loss_value = loss_fn(y, logits)

        # `accuracy` 지표의 상태를 업데이트합니다.
        accuracy.update_state(y, logits)

        # 손실 값을 최소화하기 위해 모델의 가중치를 업데이트합니다.
        gradients = tape.gradient(loss_value, model.trainable_weights)
        optimizer.apply_gradients(zip(gradients, model.trainable_weights))

        # 현재 정확도 값을 기록합니다.
        if step % 200 == 0:
            print("에포크:", epoch, "스텝:", step)
            print("지금까지 계산한 전체 정확도: %.3f" % accuracy.result())

    # 에포크 끝에서 지표의 상태를 초기화합니다.
    accuracy.reset_states()
```

<div class="k-default-codeblock">
```
에포크: 0 스텝: 0
지금까지 계산한 전체 정확도: 0.156
에포크: 0 스텝: 200
지금까지 계산한 전체 정확도: 0.778
에포크: 0 스텝: 400
지금까지 계산한 전체 정확도: 0.839
에포크: 0 스텝: 600
지금까지 계산한 전체 정확도: 0.864
에포크: 0 스텝: 800
지금까지 계산한 전체 정확도: 0.878
에포크: 1 스텝: 0
지금까지 계산한 전체 정확도: 0.922
에포크: 1 스텝: 200
지금까지 계산한 전체 정확도: 0.935
에포크: 1 스텝: 400
지금까지 계산한 전체 정확도: 0.937
에포크: 1 스텝: 600
지금까지 계산한 전체 정확도: 0.939
에포크: 1 스텝: 800
지금까지 계산한 전체 정확도: 0.940

```
</div>
또한 `self.add_loss()` 메서드와 비슷하게 층에서 `self.add_metric()` 메서드를 사용할 수 있습니다.
이 메서드는 전달한 값의 평균을 계산합니다.
층이나 모델의 `layer.reset_metrics()` 메서드를 호출하여 초기화할 수 있습니다.

---
## 컴파일된 함수

즉시 실행은 디버깅에 좋지만 정적 그래프로 컴파일하면 더 높은 성능을 얻을 수 있습니다.
정적 그래프는 연구자에게 안성맞춤입니다.
`tf.function` 데코레이터로 감싸면 어떤 함수도 컴파일할 수 있습니다.


```python
# 층, 손실, 옵티마이저를 준비합니다.
model = keras.Sequential(
    [
        keras.layers.Dense(32, activation="relu"),
        keras.layers.Dense(32, activation="relu"),
        keras.layers.Dense(10),
    ]
)
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)

# 훈련 스텝 함수를 만듭니다.


@tf.function  # 속도를 높입니다.
def train_on_batch(x, y):
    with tf.GradientTape() as tape:
        logits = model(x)
        loss = loss_fn(y, logits)
        gradients = tape.gradient(loss, model.trainable_weights)
    optimizer.apply_gradients(zip(gradients, model.trainable_weights))
    return loss


# 데이터셋을 준비합니다.
(x_train, y_train), _ = tf.keras.datasets.mnist.load_data()
dataset = tf.data.Dataset.from_tensor_slices(
    (x_train.reshape(60000, 784).astype("float32") / 255, y_train)
)
dataset = dataset.shuffle(buffer_size=1024).batch(64)

for step, (x, y) in enumerate(dataset):
    loss = train_on_batch(x, y)
    if step % 100 == 0:
        print("Step:", step, "Loss:", float(loss))
```

<div class="k-default-codeblock">
```
Step: 0 Loss: 2.2933919429779053
Step: 100 Loss: 0.6313902735710144
Step: 200 Loss: 0.4229546785354614
Step: 300 Loss: 0.4389037489891052
Step: 400 Loss: 0.18877071142196655
Step: 500 Loss: 0.35450366139411926
Step: 600 Loss: 0.3262334167957306
Step: 700 Loss: 0.36012864112854004
Step: 800 Loss: 0.32050561904907227
Step: 900 Loss: 0.23951712250709534

```
</div>
---
## 훈련 모드 & 추론 모드

`BatchNormalization`과 `Dropout` 같은 일부 층은 훈련과 추론 시에 행동이 다릅니다.
이런 층을 사용할 때는 `call` 메서드에 `training` (불리언) 매개변수를 지정하는 것이 좋습니다.

`call` 메서드에 이 매개변수를 지정하면 케라스가 기본으로 제공하는
훈련과 평가 반복을 사용할 수 있고(예를 들어 `fit` 메서드),
훈련과 추론 모드로 층을 사용할 때 오류를 줄일 수 있습니다.


```python

class Dropout(keras.layers.Layer):
    def __init__(self, rate):
        super(Dropout, self).__init__()
        self.rate = rate

    def call(self, inputs, training=None):
        if training:
            return tf.nn.dropout(inputs, rate=self.rate)
        return inputs


class MLPWithDropout(keras.layers.Layer):
    def __init__(self):
        super(MLPWithDropout, self).__init__()
        self.linear_1 = Linear(32)
        self.dropout = Dropout(0.5)
        self.linear_3 = Linear(10)

    def call(self, inputs, training=None):
        x = self.linear_1(inputs)
        x = tf.nn.relu(x)
        x = self.dropout(x, training=training)
        return self.linear_3(x)


mlp = MLPWithDropout()
y_train = mlp(tf.ones((2, 2)), training=True)
y_test = mlp(tf.ones((2, 2)), training=False)
```

---
## 함수형 API를 사용한 모델 구성

딥러닝 모델을 만들기 위해 항상 객체지향 프로그래밍을 사용할 필요는 없습니다.
지금까지 소개한 모든 층은 다음과 같은 함수 스타일로 사용할 수
있습니다(이를 "함수형(Functional) API"라고 부릅니다):


```python
# `Input` 객체를 사용해 입력의 크기와 데이터 타입을 기술합니다.
# 딥러닝에서 하나의 "타입"을 정의하는 것과 같습니다.
# shape 매개변수는 샘플 단위입니다. 즉 배치 크기는 포함하지 않습니다.
# 함수형 API는 샘플 단위의 변환을 정의하는데 초점을 맞춥니다.
# 만들어진 모델은 자동으로 샘플 단위의 변환을 배치로 수행하기 때문에
# 배치 데이터에서 사용할 수 있습니다.
inputs = tf.keras.Input(shape=(16,), dtype="float32")

# 이 "타입" 객체로 층을 호출하면
# 업데이트된 타입이 반환됩니다(새로운 크기와 데이터 타입).
x = Linear(32)(inputs)  # 앞서 정의한 Linear 층을 재사용합니다.
x = Dropout(0.5)(x)  # 앞서 정의한 Linear 층을 재사용합니다.
outputs = Linear(10)(x)

# 입력과 출력을 지정하여 함수형 `Model`을 정의합니다.
# 모델 자체는 다른 것과 같은 층입니다.
model = tf.keras.Model(inputs, outputs)

# 함수형 모델은 데이터에 호출하기 전에 이미 가중치를 가지고 있습니다.
# 미리 입력 크기를 (`Input`에) 지정했기 때문입니다.
assert len(model.weights) == 4

# 시험 삼아 샘플 데이터에 모델을 호출해 보죠.
y = model(tf.ones((2, 16)))
assert y.shape == (2, 10)

# `__call__` 메서드의 `training` 매개변수를 사용할 수 있습니다
# (`Dropout` 층으로 매개변수가 전달됩니다).
y = model(tf.ones((2, 16)), training=True)
```

함수형 API는 서브클래싱보다 간결하고 몇가지 다른 장점을 제공합니다(일반적으로
타입이 없는 객체지향 개발에 비해 타입이 있는 함수형 언어의 장점과 동일합니다).
하지만 유향 비순환 그래프(directed acyclic graph, DAG)로 정의하는 층에만 사용할 수 있습니다.
순환 신경망은 서브클래싱 층으로 정의해야 합니다.

함수형 API에 대한 더 자세한 내용은 [여기를](/guides/functional_api/) 참고하세요.

연구의 작업 흐름에 따라 객체지향 모델과 함수형 모델을 입맛에 맞게 섞어 쓸 수 있습니다.

`Model` 클래스는 기본적으로 훈련과 평가 기능을 제공합니다(`fit()`과 `evaluate()`).
객체지향 모델에서 이 기능을 사용하고 싶다면 언제든지 `Model` 클래스를 상속할 수 있습니다(`Layer`를
서브클래싱하는 것과 동일하게 작동합니다).

---
## 엔드-투-엔드 예제 1: 변이형 오토인코더

지금까지 배운 내용은 다음과 같습니다:

- `Layer`는 (`__init__`나 `build`에서 만든) 상태와 (`call`에서 정의한) 계산을 캡슐화합니다.
- 층을 재귀적으로 중첩하여 더 큰 새로운 블록을 만들 수 있습니다.
- `GradientTape`의 `with` 블록 안에서 모델을 호출하여 훈련 반복 과정을 마음껏 해킹할 수 있습니다.
그다음 옵티마이저를 사용해 추출한 그레이디언트를 적용합니다.
- `@tf.function` 데코레이터를 사용해 훈련 반복의 속도를 높일 수 있습니다.
- 층은 `self.add_loss()`를 사용해 손실(일반적으로 규제 손실)을 만들고 기록할 수 있습니다.

이를 사용해 엔드-투-엔드 예제를 만들어 보겠습니다:
변이형 오토인코더(Variational AutoEncoder, VAE)를 만들고 MNIST 데이터셋에서 훈련해 보죠.

이 VAE는 `Layer`의 서브클래스입니다.
또한 `Layer`를 서브클래싱한 층을 조합하여 구성합니다.
규제 손실(KL 발산)도 사용하겠습니다.

모델을 정의합니다.

먼저, `Sampling` 층을 사용해 MNIST 숫자 이미지를
잠재 공간의 세 값 `(z_mean, z_log_var, z)`에 매핑하는 `Encoder` 클래스를 정의합니다.


```python
from tensorflow.keras import layers


class Sampling(layers.Layer):
    """(z_mean, z_log_var)를 사용해 숫자 인코딩 벡터 z를 샘플링합니다."""

    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon


class Encoder(layers.Layer):
    """MNIST 숫자 이미지를 (z_mean, z_log_var, z) 세 값으로 매핑합니다."""

    def __init__(self, latent_dim=32, intermediate_dim=64, **kwargs):
        super(Encoder, self).__init__(**kwargs)
        self.dense_proj = layers.Dense(intermediate_dim, activation=tf.nn.relu)
        self.dense_mean = layers.Dense(latent_dim)
        self.dense_log_var = layers.Dense(latent_dim)
        self.sampling = Sampling()

    def call(self, inputs):
        x = self.dense_proj(inputs)
        z_mean = self.dense_mean(x)
        z_log_var = self.dense_log_var(x)
        z = self.sampling((z_mean, z_log_var))
        return z_mean, z_log_var, z

```

그다음 확률적 잠재 공간의 좌표를 MNIST 숫자 이미지로 매핑하는 `Decoder` 클래스를 정의합니다.


```python

class Decoder(layers.Layer):
    """인코딩된 벡터 z를 숫자 이미지로 되돌립니다."""

    def __init__(self, original_dim, intermediate_dim=64, **kwargs):
        super(Decoder, self).__init__(**kwargs)
        self.dense_proj = layers.Dense(intermediate_dim, activation=tf.nn.relu)
        self.dense_output = layers.Dense(original_dim, activation=tf.nn.sigmoid)

    def call(self, inputs):
        x = self.dense_proj(inputs)
        return self.dense_output(x)

```

마지막으로 `VariationalAutoEncoder`는 인코더와 디코더를 연결하고
`add_loss()` 메서드를 사용해 KL 발산 규제를 추가합니다.


```python

class VariationalAutoEncoder(layers.Layer):
    """인코더와 디코더를 연결하여 엔드-투-엔드 모델을 만듭니다."""

    def __init__(self, original_dim, intermediate_dim=64, latent_dim=32, **kwargs):
        super(VariationalAutoEncoder, self).__init__(**kwargs)
        self.original_dim = original_dim
        self.encoder = Encoder(latent_dim=latent_dim, intermediate_dim=intermediate_dim)
        self.decoder = Decoder(original_dim, intermediate_dim=intermediate_dim)

    def call(self, inputs):
        z_mean, z_log_var, z = self.encoder(inputs)
        reconstructed = self.decoder(z)
        # KL 발산 규제 손실을 추가합니다.
        kl_loss = -0.5 * tf.reduce_mean(
            z_log_var - tf.square(z_mean) - tf.exp(z_log_var) + 1
        )
        self.add_loss(kl_loss)
        return reconstructed

```

이제 훈련 반복문을 만듭니다.
속도를 높이기 위해 훈련 스텝을 `@tf.function`으로 감싸서 그래프로 컴파일합니다.


```python
# 변이형 오토인코더 모델
vae = VariationalAutoEncoder(original_dim=784, intermediate_dim=64, latent_dim=32)

# 손실과 옵티마이저
loss_fn = tf.keras.losses.MeanSquaredError()
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)

# 데이터셋을 준비합니다.
(x_train, _), _ = tf.keras.datasets.mnist.load_data()
dataset = tf.data.Dataset.from_tensor_slices(
    x_train.reshape(60000, 784).astype("float32") / 255
)
dataset = dataset.shuffle(buffer_size=1024).batch(32)


@tf.function
def training_step(x):
    with tf.GradientTape() as tape:
        reconstructed = vae(x)  # 입력의 재구성을 만듭니다.
        # 손실을 계산합니다.
        loss = loss_fn(x, reconstructed)
        loss += sum(vae.losses)  # KL 발산을 추가합니다.
    # VAE의 가중치를 업데이트합니다.
    grads = tape.gradient(loss, vae.trainable_weights)
    optimizer.apply_gradients(zip(grads, vae.trainable_weights))
    return loss


losses = []  # 손실을 기록합니다.
for step, x in enumerate(dataset):
    loss = training_step(x)
    # 로깅
    losses.append(float(loss))
    if step % 100 == 0:
        print("스텝:", step, "손실:", sum(losses) / len(losses))

    # 1,000번 스텝 후에 멈춥니다.
    # 수렴할 때까지 모델을 훈련하는 것은 독자들에게 숙제로 남겨 놓겠습니다.
    if step >= 1000:
        break
```

<div class="k-default-codeblock">
```
스텝: 0 손실: 0.3205374479293823
스텝: 100 손실: 0.1252927901868773
스텝: 200 손실: 0.09960039123077298
스텝: 300 손실: 0.08947645469361365
스텝: 400 손실: 0.08461058784527077
스텝: 500 손실: 0.08142319865926297
스텝: 600 손실: 0.07906099149148396
스텝: 700 손실: 0.07777281613029359
스텝: 800 손실: 0.07654486606816079
스텝: 900 손실: 0.07558360493242675
스텝: 1000 손실: 0.07465120082298717

```
</div>
여기서 볼 수 있듯이 케라스에서는 이런 종류의 모델을 빠르고 간단하게 만들고 훈련할 수 있습니다.

어쩌면 위 코드가 조금 장황하다고 생각할 수 있습니다.
상세 사항을 모두 직접 만들었습니다. 이렇게 하면 유연성이 극대화되지만 작업을 조금 해야 합니다.

함수형 API 버전의 VAE를 살펴 보죠:


```python
original_dim = 784
intermediate_dim = 64
latent_dim = 32

# 인코더 모델을 정의합니다.
original_inputs = tf.keras.Input(shape=(original_dim,), name="encoder_input")
x = layers.Dense(intermediate_dim, activation="relu")(original_inputs)
z_mean = layers.Dense(latent_dim, name="z_mean")(x)
z_log_var = layers.Dense(latent_dim, name="z_log_var")(x)
z = Sampling()((z_mean, z_log_var))
encoder = tf.keras.Model(inputs=original_inputs, outputs=z, name="encoder")

# 디코더 모델을 정의합니다.
latent_inputs = tf.keras.Input(shape=(latent_dim,), name="z_sampling")
x = layers.Dense(intermediate_dim, activation="relu")(latent_inputs)
outputs = layers.Dense(original_dim, activation="sigmoid")(x)
decoder = tf.keras.Model(inputs=latent_inputs, outputs=outputs, name="decoder")

# VAE 모델을 정의합니다.
outputs = decoder(z)
vae = tf.keras.Model(inputs=original_inputs, outputs=outputs, name="vae")

# KL 발산 규제 손실을 추가합니다.
kl_loss = -0.5 * tf.reduce_mean(z_log_var - tf.square(z_mean) - tf.exp(z_log_var) + 1)
vae.add_loss(kl_loss)
```

훨씬 간소하지 않나요?

이 경우에도 케라스는 `Model` 클래스에 기본적으로 훈련 & 평가 반복을 제공합니다(`fit()`과 `evaluate()`).
확인해 보죠:


```python
# 손실과 옵티마이저
loss_fn = tf.keras.losses.MeanSquaredError()
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)

# 데이터셋을 준비합니다.
(x_train, _), _ = tf.keras.datasets.mnist.load_data()
dataset = tf.data.Dataset.from_tensor_slices(
    x_train.reshape(60000, 784).astype("float32") / 255
)
dataset = dataset.map(lambda x: (x, x))  # 입력과 타깃으로 x_train을 사용합니다.
dataset = dataset.shuffle(buffer_size=1024).batch(32)

# 훈련을 위해 모델을 설정합니다.
vae.compile(optimizer, loss=loss_fn)

# 모델을 훈련합니다.
vae.fit(dataset, epochs=1)
```

<div class="k-default-codeblock">
```
1875/1875 [==============================] - 2s 1ms/step - loss: 0.0714

<tensorflow.python.keras.callbacks.History at 0x7f4f0c112f28>

```
</div>
함수형 API와 `fit` 메서드를 사용하여 65줄의 코드를 25줄로 줄였습니다(모델 정의와 훈련을 포함했습니다).
케라스의 철학은 이렇게 생산성을 높일 수 있는 기능을 제공하는 것입니다.
동시에 모든 것을 직접 만들어 상세 내용을 완벽히 제어할 수 있습니다.
두 문단 앞에서 만들었던 저수준 훈련 반복문을 참고하세요.

---
## 엔드-투-엔드 예제 2: 하이퍼네트워크

다른 종류의 에제인 하이퍼네트워크(hypernetwork)를 살펴 보겠습니다.

하이퍼네트워크는 가중치가 (일반적으로 더 작은) 다른 신경망에 의해 생성되는 심층 신경망입니다.

아주 작은 하이퍼네트워크를 만들어 보죠.
2개의 층을 가진 신경망을 사용해 3개의 층을 가진 신경망의 가중치를 생성하겠습니다.



```python
import numpy as np

input_dim = 784
classes = 10

# 레이블을 예측하기 위해 사용할 모델입니다(하이퍼네트워크).
outer_model = keras.Sequential(
    [keras.layers.Dense(64, activation=tf.nn.relu), keras.layers.Dense(classes),]
)

# 가중치를 만들 필요가 없으므로 미리 만들어졌다고 층을 설정하겠습니다.
# 이렇게 하면 `outer_model`이 새로운 변수를 만들지 않습니다.
for layer in outer_model.layers:
    layer.built = True

# 생성할 가중치 개수입니다.
# 하이퍼네트워크에 있는 층마다 output_dim * input_dim + output_dim개의 가중치가 필요합니다.
num_weights_to_generate = (classes * 64 + classes) + (64 * input_dim + 64)

# `outer_model` 모델의 가중치를 생성하는 모델입니다.
inner_model = keras.Sequential(
    [
        keras.layers.Dense(16, activation=tf.nn.relu),
        keras.layers.Dense(num_weights_to_generate, activation=tf.nn.sigmoid),
    ]
)
```

훈련 반복을 구현합니다. 배치 데이터에 대해 다음을 수행합니다:

- `inner_model`을 사용해 `weights_pred` 가중치 배열을 생성합니다.
- 이 가중치를 `outer_model`의 커널과 편향 텐서로 바꿉니다.
- `outer_model`의 정방향 계산을 실행하여 MNIST 데이터에 대한 예측을 계산합니다.
- `inner_model`의 가중치로 역전파하여 최종 분류 손실을 최소화합니다.


```python
# 손실과 옵티마이저
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)

# 데이터셋을 준비합니다.
(x_train, y_train), _ = tf.keras.datasets.mnist.load_data()
dataset = tf.data.Dataset.from_tensor_slices(
    (x_train.reshape(60000, 784).astype("float32") / 255, y_train)
)

# 실험을 위해 배치 크기 1을 사용하겠습니다.
dataset = dataset.shuffle(buffer_size=1024).batch(1)


@tf.function
def train_step(x, y):
    with tf.GradientTape() as tape:
        # outer_model의 가중치를 예측합니다.
        weights_pred = inner_model(x)

        # outer_model의 w와 b의 크기에 맞게 바꿉니다.
        # 첫 번째 층의 커널
        start_index = 0
        w0_shape = (input_dim, 64)
        w0_coeffs = weights_pred[:, start_index : start_index + np.prod(w0_shape)]
        w0 = tf.reshape(w0_coeffs, w0_shape)
        start_index += np.prod(w0_shape)
        # 첫 번째 층의 편향
        b0_shape = (64,)
        b0_coeffs = weights_pred[:, start_index : start_index + np.prod(b0_shape)]
        b0 = tf.reshape(b0_coeffs, b0_shape)
        start_index += np.prod(b0_shape)
        # 두 번째 층의 커널
        w1_shape = (64, classes)
        w1_coeffs = weights_pred[:, start_index : start_index + np.prod(w1_shape)]
        w1 = tf.reshape(w1_coeffs, w1_shape)
        start_index += np.prod(w1_shape)
        # 첫 번째 층의 편향
        b1_shape = (classes,)
        b1_coeffs = weights_pred[:, start_index : start_index + np.prod(b1_shape)]
        b1 = tf.reshape(b1_coeffs, b1_shape)
        start_index += np.prod(b1_shape)

        # outer_model의 가중치 변수로 설정합니다.
        outer_model.layers[0].kernel = w0
        outer_model.layers[0].bias = b0
        outer_model.layers[1].kernel = w1
        outer_model.layers[1].bias = b1

        # outer_model의 추론을 수행합니다.
        preds = outer_model(x)
        loss = loss_fn(y, preds)

    # inner_model만 훈련합니다.
    grads = tape.gradient(loss, inner_model.trainable_weights)
    optimizer.apply_gradients(zip(grads, inner_model.trainable_weights))
    return loss


losses = []  # 손실을 기록합니다.
for step, (x, y) in enumerate(dataset):
    loss = train_step(x, y)

    # 로깅
    losses.append(float(loss))
    if step % 100 == 0:
        print("스텝:", step, "손실:", sum(losses) / len(losses))

    # 1,000번 스텝 후에 멈춥니다.
    # 수렴할 때까지 모델을 훈련하는 것은 독자들에게 숙제로 남겨 놓겠습니다.
    if step >= 1000:
        break
```

<div class="k-default-codeblock">
```
스텝: 0 손실: 2.5015578269958496
스텝: 100 손실: 2.51707255545229
스텝: 200 손실: 2.2531549397355586
스텝: 300 손실: 2.084034193358214
스텝: 400 손실: 1.9536270303820071
스텝: 500 손실: 1.9124673586337015
스텝: 600 손실: 1.7858441306095352
스텝: 700 손실: 1.7136887696555623
스텝: 800 손실: 1.6410405037219724
스텝: 900 손실: 1.5765142734374127
스텝: 1000 손실: 1.5436656988156658

```
</div>
케라스로 어떤 연구 아이디어를 구현하더라도 쉽고 매우 생산적입니다.
하루에 25개의 아이디어를 실험해 보세요(평균적으로 실험당 20분입니다)!

케라스는 가능한 빠르게 아이디어에서 결과를 만들 수 있도록 설계되었습니다.
이것이 위대한 연구를 수행하는 핵심 열쇠라고 믿기 때문입니다.

이 가이드가 도움이 되었기를 바랍니다. 케라스로 무언가 만들었다면 알려주세요!
