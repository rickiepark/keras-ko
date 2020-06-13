# 케라스에 대하여

케라스는 파이썬으로 작성된 딥러닝 API이며 [텐서플로](https://github.com/tensorflow/tensorflow) 머신러닝 플랫폼 위에서 실행됩니다.
케라스는 실험을 빨리 수행하는데 초점을 맞추어 개발되었습니다. *가능한 빠르게 아이디어를 결과로 만드는 것이 성공적인 연구의 핵심 요소이기 때문입니다.*

---

## 케라스와 텐서플로 2.0

[텐서플로 2.0](https://www.tensorflow.org/)은 엔드-투-엔드(end-to-end) 오픈소스 머신러닝 플랫폼입니다. 텐서플로를 [미분가능한 프로그래밍](https://en.wikipedia.org/wiki/Differentiable_programming)을 위한 인프라 계층으로 생각할 수 있습니다. 텐서플로를 구성하는 핵심 기능 네가지는 다음과 같습니다:

- CPU, GPU, TPU에서 효율적으로 저수준 텐서 연산을 실행합니다.
- 어떤한 미분가능한 표현식에 대해서도 그레이디언트(gradient)를 계산할 수 있습니다.
- 여러 장치로 계산을 확장할 수 있습니다(예를 들면, Oak Ridge National Lab의 [Summit 수퍼컴퓨터](https://www.olcf.ornl.gov/summit/)는 GPU 27,000개를 가지고 있습니다).
- 프로그램("그래프")를 서버, 브라우저, 모바일, 임베디드 장치와 같은 외부 런타임으로 내보낼 수 있습니다.


Keras is the high-level API of TensorFlow 2.0: an approchable, highly-productive interface for solving machine learning problems,
with a focus on modern deep learning. It provides essential abstractions and building blocks for developing
and shipping machine learning solutions with high iteration velocity.

Keras empowers engineers and researchers to take full advantage of the scalability
and cross-platform capabilities of TensorFlow 2.0: you can run Keras on TPU or on large clusters of GPUs,
and you can export your Keras models to run in the browser or on a mobile device. 

---

## First contact with Keras

The core data structures of Keras are __layers__ and __models__.
The simplest type of model is the [`Sequential` model](/guides/sequential_model/), a linear stack of layers.
For more complex architectures, you should use the [Keras functional API](/guides/functional_api/),
which allows to build arbitrary graphs of layers, or [write models entirely from scratch via subclasssing](/guides/model_subclassing/).

Here is the `Sequential` model:

```python
from tensorflow.keras.models import Sequential

model = Sequential()
```

Stacking layers is as easy as `.add()`:

```python
from tensorflow.keras.layers import Dense

model.add(Dense(units=64, activation='relu'))
model.add(Dense(units=10, activation='softmax'))
```

Once your model looks good, configure its learning process with `.compile()`:

```python
model.compile(loss='categorical_crossentropy',
              optimizer='sgd',
              metrics=['accuracy'])
```

If you need to, you can further configure your optimizer. The Keras philosophy is to simple things simple,
while allowing the user to be fully in control when they need to (the ultimate control being the easy extensibility of the source code via subclassing).

```python
model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.SGD(learning_rate=0.01, momentum=0.9, nesterov=True))
```

You can now iterate on your training data in batches:

```python
# x_train and y_train are Numpy arrays --just like in the Scikit-Learn API.
model.fit(x_train, y_train, epochs=5, batch_size=32)
```

Evaluate your test loss and metrics in one line:

```python
loss_and_metrics = model.evaluate(x_test, y_test, batch_size=128)
```

Or generate predictions on new data:

```python
classes = model.predict(x_test, batch_size=128)
```

What you just saw is the most elementary way to use Keras: it mirrors the Scikit-Learn API.

However, Keras is also a highly-flexible framework suitable to iterate on state-of-the-art research ideas.
Keras follows the principle of **progressive discloure of complexity**: it makes it easy to get started,
yet it makes it possible to handle arbitrarily advanced use cases,
only requiring incremental learning at each step.

In much the same way that you were able to train & evaluate a simple neural network above in a few lines,
you can use Keras to quickly develop new training procedures or exotic model architectures.
Here's a low-level training loop example, combining Keras functionality with the TensorFlow `GradientTape`:

```python
import tensorflow as tf

# Prepare an optimizer.
optimizer = tf.keras.optimizers.Adam()
# Prepare a loss function.
loss_fn = tf.keras.losses.kl_divergence

# Iterate over the batches of a dataset.
for inputs, targets in dataset:
    # Open a GradientTape.
    with tf.GradientTape() as tape:
        # Forward pass.
        predictions = model(inputs)
        # Compute the loss value for this batch.
        loss_value = loss_fn(targets, predictions)

    # Get gradients of loss wrt the weights.
    gradients = tape.gradient(loss_value, model.trainable_weights)
    # Update the weights of the model.
    optimizer.apply_gradients(zip(gradients, model.trainable_weights))
```

The ideas behind deep learning are simple, so why should their implementation be painful?

For more in-depth tutorials about Keras, you can check out:

- [Introduction to Keras for engineers](/getting_started/intro_to_keras_for_engineers/)
- [Introduction to Keras for researchers](/getting_started/intro_to_keras_for_researchers/)
- [Developer guides](/guides/)

---

## Installation & compatibility

Keras comes packaged with TensorFlow 2.0 as `tensorflow.keras`.
To start using Keras, simply [install TensorFlow 2.0](https://www.tensorflow.org/install).

Keras/TensorFlow are compatible with:

- Python 3.5–3.8
- Ubuntu 16.04 or later
- Windows 7 or later
- macOS 10.12.6 (Sierra) or later.


---

## Support

You can ask questions and join the development discussion:

- On the [Keras Google group](https://groups.google.com/forum/#!forum/keras-users).
- On the [Keras Slack channel](https://kerasteam.slack.com). Use [this link](https://keras-slack-autojoin.herokuapp.com/) to request an invitation to the channel.

You can also post **bug reports and feature requests** (only) in [GitHub issues](https://github.com/keras-team/keras/issues). Make sure to read [our guidelines](https://github.com/keras-team/keras/blob/master/CONTRIBUTING.md) first.

---

## 왜 이름이 케라스(Keras)인가요?

>**옮긴이** 아래 문단은 [keras.io/ko/](https://keras.io/ko/#_7)에서 가져왔습니다.

케라스(κέρας)는 그리스어로 _뿔_ 이라는 뜻입니다. _Odyssey_에서 최초로 언급된, 고대 그리스와 라틴 문학의 신화적 존재에 대한 이야기로, 두 가지 꿈의 정령(_Oneiroi_, 단수 _Oneiros_) 중 하나는 상아문을 통해 땅으로 내려와 거짓된 환상으로 사람을 속이며, 다른 하나는 뿔을 통해 내려와 앞으로 벌어질 미래를 예언합니다. 이는 κέρας(뿔) / κραίνω(이뤄지다)와 ἐλέφας(상아) / ἐλεφαίρομαι(속이다)에 대한 언어유희이기도 합니다.

케라스는 초기에 ONEIROS(Open-ended Neuro-Electronic Intelligent Robot Operating System)라는 프로젝트의 일환으로 개발되었습니다.

>_"Oneiroi are beyond our unravelling - who can be sure what tale they tell? Not all that men look for comes to pass. Two gates there are that give passage to fleeting Oneiroi; one is made of horn, one of ivory. The Oneiroi that pass through sawn ivory are deceitful, bearing a message that will not be fulfilled; those that come out through polished horn have truth behind them, to be accomplished for men who see them."_ Homer, Odyssey 19. 562 ff (Shewring translation).

---

>**옮긴이** keras-ko.kr은 [keras.io](https://keras.io)의 비공식 한국어 번역입니다. keras-ko.kr은 최신 버전을 유지하지 않으며 keras.io의 내용과 다를 수 있습니다. 이 사이트에 포함된 코드와 문서에 대한 어떠한 책임도 지지않습니다. 문서 번역에 참여하시려면 [keras-ko](https://github.com/rickiepark/keras-ko) 깃허브를 참고하세요.

