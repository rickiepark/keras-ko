"""
Title: 엔지니어에게 맞는 케라스 소개
Author: [fchollet](https://twitter.com/fchollet)
Date created: 2020/04/01
Last modified: 2020/04/28
Description: 케라스로 실전 머신러닝 솔루션을 만들기 위해 알아야 할 모든 것.
"""

"""
## 설정
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras

"""
## 소개

케라스로 제품에 딥러닝을 적용하고 싶은 머신러닝 엔지니어인가요? 이 가이드에서 케라스 API의 핵심 부분을 소개하겠습니다.

이 가이드에서 다음 내용을 배울 수 있습니다:

- 모델을 훈련하기 전에 데이터를 준비하는 방법(넘파이 배열이나 `tf.data.Dataset` 객체로 변환합니다).
- 데이터 전처리 방법. 예를 들면 특성 정규화나 어휘 사전 구축.
- 케라스 함수형 API로 데이터에서 예측을 만드는 모델 구축 방법.
- 케라스의 기본 `fit()` 메서드로 체크포인팅(checkpointing), 성능 지표 모니터링, 내결함성(fault tolerance)을 고려한 모델 훈련 방법.
- 테스트 데이터에서 모델 평가하고 새로운 데이터에서 모델을 사용해 추론하는 방법.
- GAN과 같은 모델을 만들기 위해 `fit()` 메서드를 커스터마이징하는 방법.
- 여러 개의 GPU를 사용해 훈련 속도를 높이는 방법.
- 하이퍼파라미터를 튜닝하여 모델의 성능을 높이는 방법.

이 문서 끝에 다음 주제에 대한 엔드-투-엔드 예제 링크를 소개하겠습니다:

- 이미지 분류
- 텍스트 분류
- 신용 카드 부정 거래 감지


"""

"""
## 데이터 적재와 전처리

신경망은 텍스트 파일, JPEG 이미지 파일, CSV 파일 같은 원시 데이터를 그대로 처리하지 않습니다.
신경망은 **벡터화**되거나 **표준화**된 표현을 처리합니다.

- 텍스트 파일을 문자열 텐서로 읽어 단어로 분리합니다. 마지막에 단어를 정수 텐서로 인덱싱하고 변환합니다.
- 이미지를 읽어 정수 텐서로 디코딩합니다. 그다음 부동 소수로 변환하고 (보통 0에서 1사이) 작은 값으로 정규화합니다.
- CSV 데이터를 파싱하여 정수 특성은 부동 소수 텐서로 변환하고, 범주형 특성은 정수 텐서로 인덱싱하고 변환합니다.
그다음 일반적으로 각 특성을 평균 0, 단위 분산으로 정규화합니다.

먼저 데이터를 적재해 보죠.

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



"""

"""
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


"""

from tensorflow.keras.layers.experimental.preprocessing import TextVectorization

# dtype이 `string`인 예제 훈련 데이터.
training_data = np.array([["This is the 1st sample."], ["And here's the 2nd sample."]])

# TextVectorization 층 객체를 만듭니다.
# 정수 토큰 인덱스 또는 토큰의 밀집 표현(예를 들어 멀티-핫(multi-hot)이나 TF-IDF)을 반환하도록 설정할 수 있습니다.
# 텍스트 표준화와 텍스트 분할 알고리즘을 완전히 커스터마이징할 수 있습니다.
vectorizer = TextVectorization(output_mode="int")

# 배열이나 데이터셋에 대해 `adapt` 메서드를 호출하면 이 데이터를 사용해 어휘 인덱스를 생성합니다.
# 이 어휘 인덱스는 새로운 데이터를 처리할 때 재사용됩니다.
vectorizer.adapt(training_data)

# `adapt`를 호출하고 나면 이 메서드가 데이터에서 보았던 n-그램(n-gram)을 인코딩할 수 있습니다.
# 본적 없는 n-그램은 OOB(out-of-vocabulary) 토큰으로 인코딩됩니다.
integer_data = vectorizer(training_data)
print(integer_data)

"""
**Example: turning strings into sequences of one-hot encoded bigrams**
"""

from tensorflow.keras.layers.experimental.preprocessing import TextVectorization

# Example training data, of dtype `string`.
training_data = np.array([["This is the 1st sample."], ["And here's the 2nd sample."]])

# Create a TextVectorization layer instance. It can be configured to either
# return integer token indices, or a dense token representation (e.g. multi-hot
# or TF-IDF). The text standardization and text splitting algorithms are fully
# configurable.
vectorizer = TextVectorization(output_mode="binary", ngrams=2)

# Calling `adapt` on an array or dataset makes the layer generate a vocabulary
# index for the data, which can then be reused when seeing new data.
vectorizer.adapt(training_data)

# After calling adapt, the layer is able to encode any n-gram it has seen before
# in the `adapt()` data. Unknown n-grams are encoded via an "out-of-vocabulary"
# token.
integer_data = vectorizer(training_data)
print(integer_data)

"""
**Example: normalizing features**

"""

from tensorflow.keras.layers.experimental.preprocessing import Normalization

# Example image data, with values in the [0, 255] range
training_data = np.random.randint(0, 256, size=(64, 200, 200, 3)).astype("float32")

normalizer = Normalization(axis=-1)
normalizer.adapt(training_data)

normalized_data = normalizer(training_data)
print("var: %.4f" % np.var(normalized_data))
print("mean: %.4f" % np.mean(normalized_data))

"""
**Example: rescaling & center-cropping images**

Both the `Rescaling` layer and the `CenterCrop` layer are stateless, so it isn't
 necessary to call `adapt()` in this case.
"""

from tensorflow.keras.layers.experimental.preprocessing import CenterCrop
from tensorflow.keras.layers.experimental.preprocessing import Rescaling

# Example image data, with values in the [0, 255] range
training_data = np.random.randint(0, 256, size=(64, 200, 200, 3)).astype("float32")

cropper = CenterCrop(height=150, width=150)
scaler = Rescaling(scale=1.0 / 255)

output_data = scaler(cropper(training_data))
print("shape:", output_data.shape)
print("min:", np.min(output_data))
print("max:", np.max(output_data))

"""
## Building models with the Keras Functional API

A "layer" is a simple input-output transformation (such as the scaling &
center-cropping transformations above). For instance, here's a linear projection layer
 that maps its inputs to a 16-dimensional feature space:

```python
dense = keras.layers.Dense(units=16)
```

A "model" is a directed acyclic graph of layers. You can think of a model as a
"bigger layer" that encompasses multiple sublayers and that can be trained via exposure
 to data.

The most common and most powerful way to build Keras models is the Functional API. To
build models with the Functional API, you start by specifying the shape (and
optionally the dtype) of your inputs. If any dimension of your input can vary, you can
specify it as `None`. For instance, an input for 200x200 RGB image would have shape
`(200, 200, 3)`, but an input for RGB images of any size would have shape `(None,
 None, 3)`.
"""

# Let's say we expect our inputs to be RGB images of arbitrary size
inputs = keras.Input(shape=(None, None, 3))

"""
After defining your input(s), you chain layer transformations on top of your inputs,
 until your final output:
"""

from tensorflow.keras import layers

# Center-crop images to 150x150
x = CenterCrop(height=150, width=150)(inputs)
# Rescale images to [0, 1]
x = Rescaling(scale=1.0 / 255)(x)

# Apply some convolution and pooling layers
x = layers.Conv2D(filters=32, kernel_size=(3, 3), activation="relu")(x)
x = layers.MaxPooling2D(pool_size=(3, 3))(x)
x = layers.Conv2D(filters=32, kernel_size=(3, 3), activation="relu")(x)
x = layers.MaxPooling2D(pool_size=(3, 3))(x)
x = layers.Conv2D(filters=32, kernel_size=(3, 3), activation="relu")(x)

# Apply global average pooling to get flat feature vectors
x = layers.GlobalAveragePooling2D()(x)

# Add a dense classifier on top
num_classes = 10
outputs = layers.Dense(num_classes, activation="softmax")(x)

"""
Once you have defined the directed acyclic graph of layers that turns your input(s) into
 your outputs, instantiate a `Model` object:
"""

model = keras.Model(inputs=inputs, outputs=outputs)

"""
This model behaves basically like a bigger layer. You can call it on batches of data, like
 this:
"""

data = np.random.randint(0, 256, size=(64, 200, 200, 3)).astype("float32")
processed_data = model(data)
print(processed_data.shape)

"""
You can print a summary of how your data gets transformed at each stage of the model.
 This is useful for debugging.

Note that the output shape displayed for each layers includes the **batch size**. Here
 the batch size is None, which indicates our model can process batchs of any size.
"""

model.summary()

"""
The Functional API also makes it easy to build models that have multiple inputs (for
instance, an image *and* its metadata) or multiple outputs (for instance, predicting
the class of the image *and* the likelihood that a user will click on it). For a
 deeper dive into what you can do, see our
[guide to the Functional API](/guides/functional_api/).
"""

"""
## Training models with `fit()`

At this point, you know:

- How to prepare your data (e.g. as a NumPy array or a `tf.data.Dataset` object)
- How to build a model that will process your data

The next step is to train your model on your data. The `Model` class features a
built-in training loop, the `fit()` method. It accepts `Dataset` objects, Python
 generators that yield batches of data, or NumPy arrays.

Before you can call `fit()`, you need to specify an optimizer and a loss function (we
 assume you are already familiar with these concepts). This is the `compile()` step:

```python
model.compile(optimizer=keras.optimizers.RMSprop(learning_rate=1e-3),
              loss=keras.losses.CategoricalCrossentropy())
```

Loss and optimizer can be specified via their string identifiers (in this case
their default constructor argument values are used):


```python
model.compile(optimizer='rmsprop', loss='categorical_crossentropy')
```

Once your model is compiled, you can start "fitting" the model to the data.
Here's what fitting a model looks like with NumPy data:

```python
model.fit(numpy_array_of_samples, numpy_array_of_labels,
          batch_size=32, epochs=10)
```

Besides the data, you have to specify two key parameters: the `batch_size` and
the number of epochs (iterations on the data). Here our data will get sliced on batches
 of 32 samples, and the model will iterate 10 times over the data during training.

Here's what fitting a model looks like with a dataset:

```python
model.fit(dataset_of_samples_and_labels, epochs=10)
```

Since the data yielded by a dataset is expect to be already batched, you don't need to
 specify the batch size here.

Let's look at it in practice with a toy example model that learns to classify MNIST
 digits:
"""

# Get the data as Numpy arrays
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# Build a simple model
inputs = keras.Input(shape=(28, 28))
x = layers.experimental.preprocessing.Rescaling(1.0 / 255)(inputs)
x = layers.Flatten()(x)
x = layers.Dense(128, activation="relu")(x)
x = layers.Dense(128, activation="relu")(x)
outputs = layers.Dense(10, activation="softmax")(x)
model = keras.Model(inputs, outputs)
model.summary()

# Compile the model
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy")

# Train the model for 1 epoch from Numpy data
batch_size = 64
print("Fit on NumPy data")
history = model.fit(x_train, y_train, batch_size=batch_size, epochs=1)

# Train the model for 1 epoch using a dataset
dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(batch_size)
print("Fit on Dataset")
history = model.fit(dataset, epochs=1)

"""
The `fit()` call returns a "history" object which records what happened over the course
of training. The `history.history` dict contains per-epoch timeseries of metrics
values (here we have only one metric, the loss, and one epoch, so we only get a single
 scalar):
"""

print(history.history)

"""
For a detailed overview of how to use `fit()`, see the
[guide to training & evaluation with the built-in Keras methods](
  /guides/training_with_built_in_methods/).
"""

"""
### Keeping track of performance metrics

As you're training a model, you want to keep of track of metrics such as classification
accuracy, precision, recall, AUC, etc. Besides, you want to monitor these metrics not
 only on the training data, but also on a validation set.

**Monitoring metrics**

You can pass a list of metric objects to `compile()`, like this:


"""

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=[keras.metrics.SparseCategoricalAccuracy(name="acc")],
)
history = model.fit(dataset, epochs=1)

"""

**Passing validation data to `fit()`**

You can pass validation data to `fit()` to monitor your validation loss & validation
 metrics. Validation metrics get reported at the end of each epoch.

"""

val_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(batch_size)
history = model.fit(dataset, epochs=1, validation_data=val_dataset)

"""
### Using callbacks for checkpointing (and more)

If training goes on for more than a few minutes, it's important to save your model at
 regular intervals during training. You can then use your saved models
to restart training in case your training process crashes (this is important for
multi-worker distributed training, since with many workers at least one of them is
 bound to fail at some point).

An important feature of Keras is **callbacks**, configured in `fit()`. Callbacks are
 objects that get called by the model at different point during training, in particular:

- At the beginning and end of each batch
- At the beginning and end of each epoch

Callbacks are a way to make model trainable entirely scriptable.

You can use callbacks to periodically save your model. Here's a simple example: a
 `ModelCheckpoint` callback
configured to save the model at the end of every epoch. The filename will include the
 current epoch.

```python
callbacks = [
    keras.callbacks.ModelCheckpoint(
        filepath='path/to/my/model_{epoch}',
        save_freq='epoch')
]
model.fit(dataset, epochs=2, callbacks=callbacks)
```
"""

"""
You can also use callbacks to do things like periodically changing the learning of your
optimizer, streaming metrics to a Slack bot, sending yourself an email notification
 when training is complete, etc.

For detailed overview of what callbacks are available and how to write your own, see
the [callbacks API documentation](/api/callbacks/) and the
[guide to writing custom callbacks](/guides/writing_your_own_callbacks/).
"""

"""
### Monitoring training progress with TensorBoard

Staring at the Keras progress bar isn't the most ergonomic way to monitor how your loss
 and metrics are evolving over time. There's a better solution:
[TensorBoard](https://www.tensorflow.org/tensorboard),
a web application that can display real-time graphs of your metrics (and more).

To use TensorBoard with `fit()`, simply pass a `keras.callbacks.TensorBoard` callback
 specifying the directory where to store TensorBoard logs:


```python
callbacks = [
    keras.callbacks.TensorBoard(log_dir='./logs')
]
model.fit(dataset, epochs=2, callbacks=callbacks)
```

You can then launch a TensorBoard instance that you can open in your browser to monitor
 the logs getting written to this location:

```
tensorboard --logdir=./logs
```

What's more, you can launch an in-line TensorBoard tab when training models in Jupyter
 / Colab notebooks.
[Here's more information](https://www.tensorflow.org/tensorboard/tensorboard_in_notebooks).
"""

"""
### After `fit()`: evaluating test performance & generating predictions on new data

Once you have a trained model, you can evaluate its loss and metrics on new data via
 `evaluate()`:
"""

loss, acc = model.evaluate(val_dataset)  # returns loss and metrics
print("loss: %.2f" % loss)
print("acc: %.2f" % acc)

"""
You can also generate NumPy arrays of predictions (the activations of the output
 layer(s) in the model) via `predict()`:
"""

predictions = model.predict(val_dataset)
print(predictions.shape)

"""
## Using `fit()` with a custom training step

By default, `fit()` is configured for **supervised learning**. If you need a different
 kind of training loop (for instance, a GAN training loop), you
can provide your own implementation of the `Model.train_step()` method. This is the
 method that is repeatedly called during `fit()`.

Metrics, callbacks, etc. will work as usual.

Here's a simple example that reimplements what `fit()` normally does:

```python
class CustomModel(keras.Model):
  def train_step(self, data):
    # Unpack the data. Its structure depends on your model and
    # on what you pass to `fit()`.
    x, y = data
    with tf.GradientTape() as tape:
      y_pred = self(x, training=True)  # Forward pass
      # Compute the loss value
      # (the loss function is configured in `compile()`)
      loss = self.compiled_loss(y, y_pred,
                                regularization_losses=self.losses)
    # Compute gradients
    trainable_vars = self.trainable_variables
    gradients = tape.gradient(loss, trainable_vars)
    # Update weights
    self.optimizer.apply_gradients(zip(gradients, trainable_vars))
    # Update metrics (includes the metric that tracks the loss)
    self.compiled_metrics.update_state(y, y_pred)
    # Return a dict mapping metric names to current value
    return {m.name: m.result() for m in self.metrics}

# Construct and compile an instance of CustomModel
inputs = keras.Input(shape=(32,))
outputs = keras.layers.Dense(1)(inputs)
model = CustomModel(inputs, outputs)
model.compile(optimizer='adam', loss='mse', metrics=[...])

# Just use `fit` as usual
model.fit(dataset, epochs=3, callbacks=...)
```

For a detailed overview of how you customize the built-in training & evaluation loops,
 see the guide:
["Customizing what happens in `fit()`"](/guides/customizing_what_happens_in_fit/).
"""

"""
## Debugging your model with eager execution

If you write custom training steps or custom layers, you will need to debug them. The
debugging experience is an integral part of a framework: with Keras, the debugging
 workflow is designed with the user in mind.

By default, your Keras models are compiled to highly-optimized computation graphs that
deliver fast execution times. That means that the Python code you write (e.g. in a
custom `train_step`) is not the code you are actually executing. This introduces a
 layer of indirection that can make debugging hard.

Debugging is best done step by step. You want to be able to sprinkle your code with
`print()` statement to see what your data looks like after every operation, you want
to be able to use `pdb`. You can achieve this by **running your model eagerly**. With
 eager execution, the Python code you write is the code that gets executed.

Simply pass `run_eagerly=True` to `compile()`:

```python
model.compile(optimizer='adam', loss='mse', run_eagerly=True)
```

Of course, the downside is that it makes your model significantly slower. Make sure to
switch it back off to get the benefits of compiled computation graphs once you are
 done debugging!

In general, you will use `run_eagerly=True` every time you need to debug what's
 happening inside your `fit()` call.
"""

"""
## Speeding up training with multiple GPUs

Keras has built-in industry-strength support for multi-GPU training and distributed
 multi-worker training, via the `tf.distribute` API.

If you have multiple GPUs on your machine, you can train your model on all of them by:

- Creating a `tf.distribute.MirroredStrategy` object
- Building & compiling your model inside the strategy's scope
- Calling `fit()` and `evaluate()` on a dataset as usual

```python
# Create a MirroredStrategy.
strategy = tf.distribute.MirroredStrategy()

# Open a strategy scope.
with strategy.scope():
  # Everything that creates variables should be under the strategy scope.
  # In general this is only model construction & `compile()`.
  model = Model(...)
  model.compile(...)

# Train the model on all available devices.
train_dataset, val_dataset, test_dataset = get_dataset()
model.fit(train_dataset, epochs=2, validation_data=val_dataset)

# Test the model on all available devices.
model.evaluate(test_dataset)
```

For a detailed introduction to multi-GPU & distributed training, see
[this guide](/guides/distributed_training/).
"""

"""
## Doing preprocessing synchronously on-device vs. asynchronously on host CPU

You've learned about preprocessing, and you've seen example where we put image
 preprocessing layers (`CenterCrop` and `Rescaling`) directly inside our model.

Having preprocessing happen as part of the model during training
is great if you want to do on-device preprocessing, for instance, GPU-accelerated
feature normalization or image augmentation. But there are kinds of preprocessing that
are not suited to this setup: in particular, text preprocessing with the
`TextVectorization` layer. Due to its sequential nature and due to the fact that it
 can only run on CPU, it's often a good idea to do **asynchronous preprocessing**.

With asynchronous preprocessing, your preprocessing operations will run on CPU, and the
preprocessed samples will be buffered into a queue while your GPU is busy with
previous batch of data. The next batch of preprocessed samples will then be fetched
from the queue to the GPU memory right before the GPU becomes available again
(prefetching). This ensures that preprocessing will not be blocking and that your GPU
 can run at full utilization.

To do asynchronous preprocessing, simply use `dataset.map` to inject a preprocessing
 operation into your data pipeline:
"""

# Example training data, of dtype `string`.
samples = np.array([["This is the 1st sample."], ["And here's the 2nd sample."]])
labels = [[0], [1]]

# Prepare a TextVectorization layer.
vectorizer = TextVectorization(output_mode="int")
vectorizer.adapt(samples)

# Asynchronous preprocessing: the text vectorization is part of the tf.data pipeline.
# First, create a dataset
dataset = tf.data.Dataset.from_tensor_slices((samples, labels)).batch(2)
# Apply text vectorization to the samples
dataset = dataset.map(lambda x, y: (vectorizer(x), y))
# Prefetch with a buffer size of 2 batches
dataset = dataset.prefetch(2)

# Our model should expect sequences of integers as inputs
inputs = keras.Input(shape=(None,), dtype="int64")
x = layers.Embedding(input_dim=10, output_dim=32)(inputs)
outputs = layers.Dense(1)(x)
model = keras.Model(inputs, outputs)

model.compile(optimizer="adam", loss="mse", run_eagerly=True)
model.fit(dataset)

"""
Compare this to doing text vectorization as part of the model:
"""

# Our dataset will yield samples that are strings
dataset = tf.data.Dataset.from_tensor_slices((samples, labels)).batch(2)

# Our model should expect strings as inputs
inputs = keras.Input(shape=(1,), dtype="string")
x = vectorizer(inputs)
x = layers.Embedding(input_dim=10, output_dim=32)(x)
outputs = layers.Dense(1)(x)
model = keras.Model(inputs, outputs)

model.compile(optimizer="adam", loss="mse", run_eagerly=True)
model.fit(dataset)

"""
When training text models on CPU, you will generally not see any performance difference
between the two setups. When training on GPU, however, doing asynchronous buffered
preprocessing on the host CPU while the GPU is running the model itself can result in
 a significant speedup.

After training, if you to export an end-to-end model that includes the preprocessing
 layer(s), this is easy to do, since `TextVectorization` is a layer:

```python
inputs = keras.Input(shape=(1,), dtype='string')
x = vectorizer(inputs)
outputs = trained_model(x)
end_to_end_model = keras.Model(inputs, outputs)
```
"""

"""
## Finding the best model configuration with hyperparameter tuning

Once you have a working model, you're going to want to optimize its configuration --
architecture choices, layer sizes, etc. Human intuition can only go so far, so you'll
 want to leverage a systematic approach: hyperparameter search.

You can use
[Keras Tuner](https://keras-team.github.io/keras-tuner/documentation/tuners/) to find
 the best hyperparameter for your Keras models. It's as easy as calling `fit()`.

Here how it works.

First, place your model definition in a function, that takes a single `hp` argument.
Inside this function, replace any value you want to tune with a call to hyperparameter
 sampling methods, e.g. `hp.Int()` or `hp.Choice()`:

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

The function should return a compiled model.

Next, instantiate a tuner object specifying your optimiation objective and other search
 parameters:


```python
import kerastuner

tuner = kerastuner.tuners.Hyperband(
  build_model,
  objective='val_loss',
  max_epochs=100,
  max_trials=200,
  executions_per_trial=2,
  directory='my_dir')
```

Finally, start the search with the `search()` method, which takes the same arguments as
 `Model.fit()`:

```python
tuner.search(dataset, validation_data=val_dataset)
```

When search is over, you can retrieve the best model(s):

```python
models = tuner.get_best_models(num_models=2)
```

Or print a summary of the results:

```python
tuner.results_summary()
```

"""

"""
## End-to-end examples

To familiarize yourself with the concepts in this introduction, see the following
 end-to-end examples:

- [Text classification](/examples/nlp/text_classification_from_scratch/)
- [Image classification](/examples/vision/image_classification_from_scratch/)
- [Credit card fraud detection](/examples/structured_data/imbalanced_classification/)

"""

"""
## What to learn next

- Learn more about the
[Functional API](/guides/functional_api/).
- Learn more about the
[features of `fit()` and `evaluate()`](/guides/training_with_built_in_methods/).
- Learn more about
[callbacks](/guides/writing_your_own_callbacks/).
- Learn more about
[creating your own custom training steps](/guides/customizing_what_happens_in_fit/).
- Learn more about
[multi-GPU and distributed training](/guides/distributed_training/).
- Learn how to do [transfer learning](/guides/transfer_learning/).
"""
