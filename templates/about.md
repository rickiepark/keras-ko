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

케라스는 텐서플로 2.0의 고수준 API입니다. 머신러닝 문제, 특히 최신 딥러닝에 초점을 맞춘 사용하기 쉽고 생산성이 높은 인터페이스입니다.
빠른 속도로 반복하여 머신러닝 솔루션을 개발하고 배포하기 위해 꼭 필요한 기능을 추상화한 기초 구성 요소를 제공합니다.

케라스를 사용하면 엔지니어와 연구자들이 텐서플로 2.0의 확장성과 크로스-플랫폼 능력을 모두 활용할 수 있습니다.
TPU나 대규모 GPU 클러스터에서 케라스를 실행하거나 케라스 모델을 브라우저나 모바일 장치에서 실행할 수 있습니다.

---

## 케라스 맛보기

케라스의 핵심 데이터 구조는 **층**과 **모델**입니다.
가장 간단한 모델은 차례대로 층을 쌓은 [`Sequential` 모델](/guides/sequential_model/)입니다.
더 복잡한 모델을 만들려면 [케라스 함수형 API](/guides/functional_api/)를 사용해야 합니다. 층으로 구성된 임의의 그래프를 만들거나 [서브클래싱을 사용해 완전히 밑바닥부터](/guides/making_new_layers_and_models_via_subclassing/) 만들 수 있습니다.

`Sequential` 모델은 다음과 같습니다.

```python
from tensorflow.keras.models import Sequential

model = Sequential()
```

층을 쌓으려면 `.add()` 메서드를 사용합니다.

```python
from tensorflow.keras.layers import Dense

model.add(Dense(units=64, activation='relu'))
model.add(Dense(units=10, activation='softmax'))
```

모델 구성이 끝나면 `.compile()` 메서드로 학습 과정을 설정합니다.

```python
model.compile(loss='categorical_crossentropy',
              optimizer='sgd',
              metrics=['accuracy'])
```

필요하면 옵티마이저 기본 설정을 바꿀 수 있습니다. 케라스의 철학은 간단한 것은 간단하게 놔두고 필요할 때 사용자가 완전하게 제어할 수 있도록 하는 것입니다(궁극의 제어는 서브클래싱을 통해 소스 코드를 확장하는 것입니다).

```python
model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.SGD(learning_rate=0.01, momentum=0.9, nesterov=True))
```

이제 훈련 데이터 배치에서 반복합니다:

```python
# 사이킷런(Scikit-Learn) API와 비슷하게 x_train과 y_train은 넘파이 배열입니다.
model.fit(x_train, y_train, epochs=5, batch_size=32)
```

다음 한 줄 코드로 손실과 성능을 평가합니다:

```python
loss_and_metrics = model.evaluate(x_test, y_test, batch_size=128)
```

또는 새로운 데이터에 대한 예측을 만듭니다:

```python
classes = model.predict(x_test, batch_size=128)
```

방금 본 것은 매우 기본적인 케라스 사용 방법으로 사이킷런 API와 닮았습니다.

하지만 케라스는 매우 유연한 프레임워크로 최신 연구를 수행하는데 적합합니다.
케라스는 **단계적인 복잡성 노출** 원칙을 따릅니다.
처음 시작할 때는 간단하지만 필요하면 어떤 고급 방식도 다룰 수 있습니다.
단계마다 추가로 필요한 내용을 학습하면 됩니다.

위에서 몇 줄의 코드로 간단한 신경망을 훈련하고 평가하는 것과 비슷하게
케라스를 사용해 새로운 훈련 방식이나 특이한 모델 구조를 빠르게 개발할 수 있습니다.
다음은 텐서플로의 `GradientTape`과 케라스를 연결하여 저수준에서 훈련을 반복하는 예입니다:

```python
import tensorflow as tf

# 옵티마이저를 준비합니다.
optimizer = tf.keras.optimizers.Adam()
# 손실 함수를 준비합니다.
loss_fn = tf.keras.losses.kl_divergence

# 데이터셋 배치를 반복합니다.
for inputs, targets in dataset:
    # GradientTape를 시작합니다.
    with tf.GradientTape() as tape:
        # 정방향 계산을 수행합니다.
        predictions = model(inputs)
        # 이 배치에 대한 손실을 계산합니다.
        loss_value = loss_fn(targets, predictions)

    # 가중치에 대한 손실의 그레이디언트를 구합니다.
    gradients = tape.gradient(loss_value, model.trainable_weights)
    # 모델 가중치를 업데이트합니다.
    optimizer.apply_gradients(zip(gradients, model.trainable_weights))
```

딥러닝 이면에 있는 아이디어는 간단합니다. 따라서 구현이 어려울 필요가 있을까요?

더 자세한 케라스 튜토리얼은 다음을 참고하세요:

- [엔지니어에게 맞는 케라스 소개](/getting_started/intro_to_keras_for_engineers/)
- [연구자에게 맞는 케라스 소개](/getting_started/intro_to_keras_for_researchers/)
- [개발자 가이드](/guides/)

---

## 설치와 호환성

케라스는 텐서플로 2.0에 `tensorflow.keras`로 포함되어 있으므로 케라스를 사용하려면 [텐서플로 2.0을 설치](https://www.tensorflow.org/install)하면 됩니다.

케라스/텐서플로는 다음 환경에서 사용할 수 있습니다:

- 파이썬 3.5–3.8
- 우분투 16.04 이상
- 윈도우 7 이상
- macOS 10.12.6 (시에라) 이상


---

## 지원

개발 포럼에 가입하여 궁금한 점을 질문할 수 있습니다:

- [케라스 구글 그룹](https://groups.google.com/forum/#!forum/keras-users).
- [케라스 슬랙 채널](https://kerasteam.slack.com). 채널에 초대 요청을 하려면 [이 링크](https://keras-slack-autojoin.herokuapp.com/)를 사용하세요.

>**옮긴이** 한국 사용자 모임으로는 [케라스 코리아](https://www.facebook.com/groups/KerasKorea/)가 있습니다.

**버그 리포트와 기능 요청**은 [GitHub issues](https://github.com/keras-team/keras/issues)에서만 받습니다. 먼저 [가이드라인](https://github.com/keras-team/keras/blob/master/CONTRIBUTING.md)을 읽어 보세요.

---

## 왜 이름이 케라스(Keras)인가요?

>**옮긴이** 아래 문단은 [keras.io/ko/](https://keras.io/ko/#_7)에서 가져왔습니다.

케라스(κέρας)는 그리스어로 _뿔_ 이라는 뜻입니다. _Odyssey_에서 최초로 언급된, 고대 그리스와 라틴 문학의 신화적 존재에 대한 이야기로, 두 가지 꿈의 정령(_Oneiroi_, 단수 _Oneiros_) 중 하나는 상아문을 통해 땅으로 내려와 거짓된 환상으로 사람을 속이며, 다른 하나는 뿔을 통해 내려와 앞으로 벌어질 미래를 예언합니다. 이는 κέρας(뿔) / κραίνω(이뤄지다)와 ἐλέφας(상아) / ἐλεφαίρομαι(속이다)에 대한 언어유희이기도 합니다.

케라스는 초기에 ONEIROS(Open-ended Neuro-Electronic Intelligent Robot Operating System)라는 프로젝트의 일환으로 개발되었습니다.

>_"Oneiroi are beyond our unravelling - who can be sure what tale they tell? Not all that men look for comes to pass. Two gates there are that give passage to fleeting Oneiroi; one is made of horn, one of ivory. The Oneiroi that pass through sawn ivory are deceitful, bearing a message that will not be fulfilled; those that come out through polished horn have truth behind them, to be accomplished for men who see them."_ Homer, Odyssey 19. 562 ff (Shewring translation).

---

