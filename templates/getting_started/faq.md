# 케라스 FAQ

자주 묻는 케라스 질문 목록입니다.

## 일반적인 질문

- [(한 대의 컴퓨터에 있는) 여러 GPU에서 어떻게 케라스 모델을 훈련할 수 있나요?](#한-대의-컴퓨터에-있는-여러-gpu에서-어떻게-케라스-모델을-훈련할-수-있나요)
- [여러 대의 머신으로 어떻게 훈련을 분산할 수 있나요?](#여러-대의-머신으로-어떻게-훈련을-분산할-수-있나요)
- [어떻게 TPU로 케라스 모델을 훈련할 수 있나요?](#어떻게-TPU로-케라스-모델을-훈련할-수-있나요)
- [케라스 설정 파일은 어디에 저장되나요?](#케라스-설정-파일은-어디에-저장되나요)
- [케라스에서 하이퍼파라미터 튜닝을 어떻게 하나요?](#케라스에서-하이퍼파라미터-튜닝을-어떻게-하나요)
- [케라스에서 개발 결과를 어떻게 재현할 수 있나요?](#케라스에서-개발-결과를-어떻게-재현할-수-있나요)
- [모델 저장 방법에는 어떤 것이 있나요?](#모델-저장-방법에는-어떤-것이-있나요)
- [모델 저장을 위해 어떻게 HDF5나 h5py를 설치할 수 있나요?](#모델-저장을-위해-어떻게-HDF5나-h5py를-설치할-수-있나요)
- [어떻게 케라스를 인용하나요?](#어떻게-케라스를-인용하나요)

## 훈련과 관련된 질문

- ['샘플', '배치', '에포크'가 무슨 뜻인가요?](#샘플-배치-에포크가-무슨-뜻인가요)
- [훈련 손실이 왜 테스트 손실보다 훨씬 높나요?](#훈련-손실이-왜-테스트-손실보다-훨씬-높나요)
- [How can I use Keras with datasets that don't fit in memory?](#how-can-i-use-keras-with-datasets-that-dont-fit-in-memory)
- [How can I regularly save Keras models during training?](#how-can-i-regularly-save-keras-models-during-training)
- [How can I interrupt training when the validation loss isn't decreasing anymore?](#how-can-i-interrupt-training-when-the-validation-loss-isnt-decreasing-anymore)
- [How can I freeze layers and do fine-tuning?](#how-can-i-freeze-layers-and-do-fine-tuning)
- [What's the difference between the `training` argument in `call()` and the `trainable` attribute?](#whats-the-difference-between-the-training-argument-in-call-and-the-trainable-attribute)
- [In `fit()`, how is the validation split computed?](#in-fit-how-is-the-validation-split-computed)
- [In `fit()`, is the data shuffled during training?](#in-fit-is-the-data-shuffled-during-training)
- [What's the recommended way to monitor my metrics when training with `fit()`?](#whats-the-recommended-way-to-monitor-my-metrics-when-training-with-fit)
- [What if I need to customize what `fit()` does?](#what-if-i-need-to-customize-what-fit-does)
- [How can I train models in mixed precision?](#how-can-i-train-models-in-mixed-precision)

## 모델링과 관련된 질문

- [How can I obtain the output of an intermediate layer (feature extraction)?](#how-can-i-obtain-the-output-of-an-intermediate-layer-feature-extraction)
- [How can I use pre-trained models in Keras?](#how-can-i-use-pre-trained-models-in-keras)
- [How can I use stateful RNNs?](#how-can-i-use-stateful-rnns)


---

## 일반적인 질문


### (한 대의 컴퓨터에 있는) 여러 GPU에서 어떻게 케라스 모델을 훈련할 수 있나요?

하나의 모델을 여러 GPU에서 실행하는 방법은 **데이터 병렬화** 와 **장치 병렬화** 두가지 입니다.
대부분의 경우 데이터 병렬화가 필요할 것입니다.


**1) 데이터 병렬화**

데이터 병렬화는 타깃 모델을 각 장치에 복사하고 이 복사본을 사용해 입력 데이터의 일부분을 처리합니다.

케라스 모델로 데이터 병렬화를 구현하려면 `tf.distribute` API를 사용하는 것이 가장 좋습니다. [케라스와 `tf.distribute`에 대한 가이드](/guides/distributed_training/)를 읽어 보세요.

핵심 내용은 다음과 같습니다:

a) "분산 전략" 객체를 만듭니다. 예를 들면 (모델을 가능한 장치에 복제하고 모델의 상태를 동기화하는) `MirroredStrategy`가 있습니다:

```python
strategy = tf.distribute.MirroredStrategy()
```

b) 이 전략의 범위(scope) 안에서 모델을 만들고 컴파일합니다:

```python
with strategy.scope():
    # 어떤 종류의 모델도 가능합니다. -- 함수형, 서브클래싱 등등
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, 3, activation='relu', input_shape=(28, 28, 1)),
        tf.keras.layers.GlobalMaxPooling2D(),
        tf.keras.layers.Dense(10)
    ])
    model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  optimizer=tf.keras.optimizers.Adam(),
                  metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])
```

중요한 것은 모든 상태 변수는 이 범위 안에서 생성되어야 합니다.
따라서 어떤 변수를 추가로 만들려면 이 범주 안에서 만드세요.

c) `tf.data.Dataset` 객체를 입력으로 `fit()` 메서드를 호출합니다.
분산 전략은 대체적으로 사용자 정의 콜백을 포함하여 모든 콜백과 호환됩니다.
이 메서드는 전략 범위 안에서 호출할 필요가 없습니다. 따라서 새로운 변수를 만들지 않습니다.

```python
model.fit(train_dataset, epochs=12, callbacks=callbacks)
```


**2) 모델 병렬화**

모델 병렬화는 여러 장치에서 한 모델의 다른 부분을 실행하는 것입니다.
이 전략은 병렬 구조를 갖는 모델에 잘 맞습니다. 예를 들면 두 개의 브랜치가 있는 모델입니다.

이를 위해 텐서플로의 장치 범위를 사용합니다. 간단한 예는 다음과 같습니다:

```python
# 공유 LSTM을 사용해 두 개의 다른 시퀀스를 병렬로 인코딩하는 모델
input_a = keras.Input(shape=(140, 256))
input_b = keras.Input(shape=(140, 256))

shared_lstm = keras.layers.LSTM(64)

# 한 GPU에서 첫 번째 시퀀스를 처리합니다.
with tf.device_scope('/gpu:0'):
    encoded_a = shared_lstm(input_a)
# 다른 GPU에서 다음 시퀀스를 처리합니다.
with tf.device_scope('/gpu:1'):
    encoded_b = shared_lstm(input_b)

# CPU에서 결과를 합칩니다.
with tf.device_scope('/cpu:0'):
    merged_vector = keras.layers.concatenate(
        [encoded_a, encoded_b], axis=-1)
```

---

### 여러 대의 머신으로 어떻게 훈련을 분산할 수 있나요?

한 대의 머신에서 병렬화하는 것과 마찬가지로 케라스로 분산 훈련을 하려면 `tf.distribute` API인 [`MultiWorkerMirroredStrategy`](https://www.tensorflow.org/api_docs/python/tf/distribute/experimental/MultiWorkerMirroredStrategy)를 사용하는 것이 가장 좋습니다. [케라스와 `tf.distribute`에 대한 가이드](/guides/distributed_training/)를 읽어 보세요.

분산 훈련은 단일 머신에서 단일 장치를 사용해 훈련하는 것보다 조금 더 많은 수고가 듭니다. 원격 서버 클러스터를 구성하고 "치프(chief)" 머신에서 코드를 실행해야 합니다. 치프 머신은 `TF_CONFIG` 환경 변수에 클러스터 내 다른 머신과 통신하는 방법을 지정합니다. 그다음부터는 단일 머신 다중 GPU를 사용한 훈련과 비슷합니다. 주요 차이점은 분산 전략으로 `MultiWorkerMirroredStrategy`를 사용하는 것입니다.

다음 작업은 필수적이며 중요합니다:

- 클러스터에 있는 모든 워커(worker)에서 데이터를 효율적으로 가져올 수 있도록 데이터셋이 준비되어야 합니다(예를 들어 GCP에서 클러스터를 구성했다면 GCS에 데이터를 놓는 것이 좋습니다).
- 훈련 과정에서 발생할 수 있는 장애에 대비해야 합니다(예를 들어, `ModelCheckpoint` 콜백을 사용합니다).

---

### 어떻게 TPU로 케라스 모델을 훈련할 수 있나요?

TPU는 딥러닝을 위한 고속 & 고효율의 하드웨어 가속기로 구글 클라우드 플랫폼에서 사용할 수 있습니다.
코랩(Colab), AI 플랫폼(ML 엔진), 딥러닝 VM(VM에서 `TPU_NAME` 환경 변수를 설정해야 합니다)에서 TPU를 사용할 수 있습니다.

먼저 [TPU 사용 가이드](https://www.tensorflow.org/guide/tpu)를 읽어 보세요. 간단히 요약하면 다음과 같습니다:

TPU 런타임에 연결한 후(예를 들어, 코랩에서 TPU 런타임을 선택한 다음), `TPUClusterResolver`를 사용해 TPU를 감지해야 합니다.
이렇게 하면 지원하는 모든 플랫폼에서 TPU를 자동으로 감지합니다:

```python
tpu = tf.distribute.cluster_resolver.TPUClusterResolver()  # TPU 감지
print('TPU 워커: ', tpu.cluster_spec().as_dict()['worker'])

tf.config.experimental_connect_to_cluster(tpu)
tf.tpu.experimental.initialize_tpu_system(tpu)
strategy = tf.distribute.experimental.TPUStrategy(tpu)
print('레플리카: ', strategy.num_replicas_in_sync)

with strategy.scope():
    # 여기에서 모델을 만듭니다.
    ...
```

초기 설정 이후 작업 절차는 단일 머신 다중 GPU 훈련과 비슷합니다. 분산 전략으로 `TPUStrategy`를 사용하는 것이 주요 차이점입니다.

다음 작업은 필수적이며 중요합니다:

- 데이터셋이 고정된 크기의 배치를 생성해야 합니다. TPU 그래프는 고정 크기의 입력만 처리할 수 있습니다.
- TPU를 최대한 활용하기 위해 가능한 빠르게 데이터를 읽을 수 있어야 합니다. 이를 위해 [TFRecord 포맷](https://www.tensorflow.org/tutorials/load_data/tfrecord)으로 데이터를 저장하는 것이 좋습니다.
- TPU를 최대한 활용하기 위해 그래프 실행마다 여러 번 경사 하강법 단계를 실행하세요. `compile()` 메서드의 `experimental_steps_per_execution` 매개변수로 지정할 수 있습니다. 모델이 작은 경우 속도가 크게 향상됩니다.

---

### 케라스 설정 파일은 어디에 저장되나요?

모든 케라스 데이터가 저장되는 기본 디렉토리는 다음과 같습니다:

`$HOME/.keras/`

예를 들어 macOS의 경우 `/Users/fchollet/.keras/`에 저장됩니다.

윈도우 사용자는 `$HOME` 대신 `%USERPROFILE%`이 됩니다.

케라스가 (권한 이슈 등으로) 디렉토리를 만들 수 없는 경우에는 `/tmp/.keras/`가 사용됩니다.

케라스 설정은 `$HOME/.keras/keras.json`에 JSON 파일로 저장됩니다. 기본 설정 내용은 다음과 같습니다:

```
{
    "image_data_format": "channels_last",
    "epsilon": 1e-07,
    "floatx": "float32",
    "backend": "tensorflow"
}
```

포함된 필드는 다음과 같습니다:

- 이미지 처리 층과 유틸리티에서 기본으로 사용되는 이미지 데이터 포맷(`channels_last` 또는 `channels_first`).
- 일부 연산에서 0 나눗셈을 방지하기 위해 사용되는 작은 양수 값인 `epsilon`.
- 기본 부동 소수 데이터 타입.
- 기본 백엔드(backend). 호환성을 위해 남아 있습니다. 지금은 텐서플로만 지원합니다.

[`get_file()`](/utils/#get_file)으로 다운로드한 데이터셋 파일은 기본으로 `$HOME/.keras/datasets/`에 저장됩니다. 케라스 애플리케이션에서 다운로드된 모델 가중치는 기본으로 `$HOME/.keras/models/`에 저장됩니다.

---

### 케라스에서 하이퍼파라미터 튜닝을 어떻게 하나요?

[케라스 튜너(Tuner)](https://keras-team.github.io/keras-tuner/)를 사용하세요.

---

### 케라스에서 개발 결과를 어떻게 재현할 수 있나요?

모델을 개발하는 동안 성능 변화가 모델이나 데이터 변경 때문인지 단순히 랜덤 초깃값 때문인지 구분하기 위해 이따금 실행 간에 얻은 결과를 재현하는 것이 필요합니다.

먼저 (프로그램 자체내에서 하는 것이 아니라) 프로그램을 시작하기 전에 `PYTHONHASHSEED` 환경 변수를 `0`으로 지정해야 합니다.
파이썬 3.2.3 이상에서는 일부 해시 기반 연산때문에 재현을 위해서 필수적입니다(예를 들면 셋(set)이나 딕셔너리의 아이템 순서. 자세한 내용은 [파이썬 문서](https://docs.python.org/3.7/using/cmdline.html#envvar-PYTHONHASHSEED)나 [이슈 #2280](https://github.com/keras-team/keras/issues/2280#issuecomment-306959926)를 참고하세요). 다음처럼 파이썬을 실행하기 전에 환경 변수를 지정할 수 있습니다:

```shell
$ cat test_hash.py
print(hash("keras"))
$ python3 test_hash.py                  # 해시 재현 불가능 (파이썬 3.2.3+)
8127205062320133199
$ python3 test_hash.py                  # 해시 재현 불가능 (Python 3.2.3+)
3204480642156461591
$ PYTHONHASHSEED=0 python3 test_hash.py # 해시 재현 가능
4883664951434749476
$ PYTHONHASHSEED=0 python3 test_hash.py # 해시 재현 가능
4883664951434749476
```endshell

또한 GPU에서 실행할 때 `tf.reduce_sum()`과 같은 어떤 연산들의 출력은 결정적이지 않습다. 이는 GPU가 여러 연산을 병렬로 실행하기 때문입니다. 따라서 실행 순서가 항상 보장되지 않습니다. 부동 소수 정밀도가 제한적이기 때문에 몇 개의 실수라도 더하는 순서에 따라 결과가 조금 달라질 수 있습니다. 결정적이지 않은 연산을 의도적으로 피할 수 있지만 텐서플로가 그레이디언트를 계산하기 위해 자동으로 만드는 연산도 있습니다. 따라서 간단한 방법은 CPU에서 실행하는 것입니다. 이렇게 하려면 `CUDA_VISIBLE_DEVICES` 환경 변수를 빈 문자열로 설정합니다. 예를 들면 다음과 같습니다:

```shell
$ CUDA_VISIBLE_DEVICES="" PYTHONHASHSEED=0 python your_program.py
```endshell

다음은 재현 가능한 결과를 얻기 위한 예시 코드입니다:

```python
import numpy as np
import tensorflow as tf
import random as python_random

# 넘파이 랜덤 시드 값을 위해 필요합니다.
np.random.seed(123)

# 파이썬 랜덤 시드 값을 위해 필요합니다.
python_random.seed(123)

# set_seed() 함수는 텐서플로의 랜덤 숫자 생성을 위해 필요합니다.
# 자세한 내용은 다음을 참조하세요:
# https://www.tensorflow.org/api_docs/python/tf/random/set_seed
tf.random.set_seed(1234)

# 나머지 코드가 이어집니다 ...
```

위 과정을 수행하면 코드에서 개별적인 초기화 메서드를 위해 시드를 설정할 필요가 없습니다.
이런 시드는 위에서 지정한 시드를 조합하여 결정되기 때문입니다.

---

### 모델 저장 방법에는 어떤 것이 있나요?

*노트: pickle이나 cPickle을 사용해 케라스 모델을 저장하는 것은 권장되지 않습니다.*

**1) 전체 모델 저장 (설정 + 가중치)**

전체 모델을 저장하면 파일에 다음과 같은 것들이 포함됩니다:

- 모델을 재생성할 수 있는 모델 구조
- 모델의 가중치
- 훈련 설정 (손실, 옵티마이저)
- 중지한 곳부터 훈련을 다시 시작할 수 있는 옵티마이저 상태

텐서플로의 [SavedModel](https://www.tensorflow.org/guide/saved_model) 포맷이 기본값이며 권장됩니다.
텐서플로 2.0 이상에서는 `model.save(your_file_path)`로 저장할 수 있습니다.

명확하게 표현하려면 `model.save(your_file_path, save_format='tf')`와 같이 사용할 수 있습니다.

케라스는 아직 원래 HDF5 저장 포맷을 지원합니다. HDF5 포맷으로 모델을 저장하려면 `model.save(your_file_path, save_format='h5')`을 사용하세요.
`your_file_path`가 `.h5`나 `.keras`로 끝나면 자동으로 이 옵션이 선택됩니다.
`h5py`를 설치하는 방법은 [모델 저장을 위해 어떻게 HDF5나 h5py를 설치할 수 있나요?](#모델-저장을-위해-어떻게-HDF5나-h5py를-설치할-수-있나요) 항목을 참고하세요.

어느 포맷이든지 모델을 저장한 후에 `model = keras.models.load_model(your_file_path)`로 모델 객체를 다시 만들 수 있습니다.

**예제:**

```python
from tensorflow.keras.models import load_model

model.save('my_model')  # HDF5 파일 'my_model.h5'을 만듭니다
del model  # 기존 모델을 삭제합니다

# 이전과 동일한 컴파일된 모델을 반환합니다
model = load_model('my_model')
```


**2) 가중치만 저장**


**모델의 가중치**를 저장하고 싶을 때에도 다음 코드와 같이 HDF5 파일로 저장할 수 있습니다:

```python
model.save_weights('my_model_weights.h5')
```

모델 객체를 만들었다고 가정하고 저장된 가중치를 *동일한* 구조의 모델로 로드할 수 있습니다:

```python
model.load_weights('my_model_weights.h5')
```

미세 조정이나 전이 학습을 위해 가중치를 (일부 층만 같은) *다른* 모델 구조에 로드하려면 *층 이름*으로 로드할 수 있습니다:

```python
model.load_weights('my_model_weights.h5', by_name=True)
```

예제:

```python
"""
원래 모델이 다음과 같다고 가정합니다:

model = Sequential()
model.add(Dense(2, input_dim=3, name='dense_1'))
model.add(Dense(3, name='dense_2'))
...
model.save_weights(fname)
"""

# 새로운 모델
model = Sequential()
model.add(Dense(2, input_dim=3, name='dense_1'))  # 가중치 적재 가능
model.add(Dense(10, name='new_dense'))  # 가중치 적재 불가능

# 원래 모델에서 로드한 가중치는 첫 번째 층인 dense_1에만 영향을 미칩니다.
model.load_weights(fname, by_name=True)
```

`h5py`를 설치하는 방법은 [모델 저장을 위해 어떻게 HDF5나 h5py를 설치할 수 있나요?](#모델-저장을-위해-어떻게-HDF5나-h5py를-설치할-수-있나요) 항목을 참고하세요.


**3) 설정만 저장 (직렬화)**


가중치나 훈련 설정은 제외하고 **모델의 구조**만 저장하려면 다음과 같이 합니다:

```python
# JSON으로 저장합니다.
json_string = model.to_json()
```

생성된 JSON은 사람이 읽을 수 있고 필요하면 수동으로 고칠 수 있습니다.

이 파일에서 새로운 모델을 만들 수 있습니다:

```python
# JSON에서 모델 재생성:
from tensorflow.keras.models import model_from_json
model = model_from_json(json_string)
```


**4) 사용자 정의 층 (또는 사용자 정의 객체) 다루기**

로드하려는 모델에 사용자 정의 층 또는 사용자 정의 클래스나 함수가 포함되어 있다면 `custom_objects` 매개변수를 통해 로딩 메카니즘을 전달할 수 있습니다:

```python
from tensorflow.keras.models import load_model
# "AttentionLayer" 클래스의 객체를 포함한 모델이라고 가정합니다.
model = load_model('my_model.h5', custom_objects={'AttentionLayer': AttentionLayer})
```

또는 [CustomObjectScope](https://keras.io/utils/#customobjectscope)를 사용할 수 있습니다:

```python
from tensorflow.keras.utils import CustomObjectScope

with CustomObjectScope({'AttentionLayer': AttentionLayer}):
    model = load_model('my_model.h5')
```

사용자 정의 객체를 다루는 것은 `load_model`이나 `model_from_json`과 같습니다:

```python
from tensorflow.keras.models import model_from_json
model = model_from_json(json_string, custom_objects={'AttentionLayer': AttentionLayer})
```

---

### 모델 저장을 위해 어떻게 HDF5나 h5py를 설치할 수 있나요?

케라스 모델을 HDF5 파일로 저장하기 위해 케라스는 h5py 파이썬 패키지를 사용합니다.
따라서 미리 설치되어 있어야 합니다.
데비안 기반의 배포판에서는 `libhdf5`도 설치해야 합니다:

<div class="k-default-code-block">
```
sudo apt-get install libhdf5-serial-dev
```
</div>

h5py가 설치되었는지 확인하려면 파이썬 셸을 열고 다음처럼 모듈을 임포트합니다.

```
import h5py
```

에러 없이 임포트되면 정상적으로 설치된 것입니다.
그렇지 않으면 자세한 [설치 가이드](http://docs.h5py.org/en/latest/build.html)를 참고하세요.



---

### 어떻게 케라스를 인용하나요?

케라스가 연구에 도움이 되었다면 논문에 케라스를 인용해 주세요. 다음은 BibTeX에 등록하는 예입니다:

<code style="color: gray;">
@misc{chollet2015keras,<br>
&nbsp;&nbsp;title={Keras},<br>
&nbsp;&nbsp;author={Chollet, Fran\c{c}ois and others},<br>
&nbsp;&nbsp;year={2015},<br>
&nbsp;&nbsp;howpublished={\url{https://keras.io}},<br>
}
</code>

---

## 훈련과 관련된 질문


### '샘플', '배치', '에포크'가 무슨 뜻인가요?


다음 정의는 케라스의 `fit()` 메서드를 올바르게 사용하기 위해 꼭 알아야 합니다:

- **샘플(sample)**: 데이터셋의 한 원소. 예를 들어 하나의 이미지는 합성곱 신경망에서 하나의 **샘플**입니다. 하나의 오디오 클립은 음성 인식 모델을 위한 하나의 **샘플**입니다.

- **배치(batch)**: *N* 개의 샘플 집합. **배치**에 있는 샘플은 독립적이며 병렬로 처리됩니다. 훈련에서 하나의 배치는 모델을 한 번만 업데이트합니다. **배치**는 일반적으로 하나의 샘플보다 입력 데이터의 분포를 더 잘 근사합니다. 배치가 클수록 더 잘 근사합니다. 하지만 이런 배치는 처리 시간이 오래 걸릴 수 있고 여전히 한 번만 모델을 업데이트합니다. 추론(평가나 예측)에서는 메모리가 허락하는한 큰 배치를 만드는 것이 권장됩니다(배치가 클수록 평가나 예측 속도가 빨라지기 때문입니다).

- **에포크(epoch)**: 일반적으로 "전체 데이터셋을 한 번 처리하는 것"으로 정의하는 임의의 기준. 훈련을 여러 단계로 나누기 위해 사용합니다. 로깅과 반복적인 평가에 도움이 됩니다. 케라스 모델의 `fit` 메서드에 `validation_data`나 `validation_split` 매개변수를 사용할 때 매 **에포크** 종료 후에 평가가 수행됩니다. 케라스에서는 **에포크** 종료 시에 실행할 수 있도록 특별히 설계된 [콜백(callback)](/api/callbacks/)을 추가할 수 있습니다. 예를 들어 학습률을 바꾸고 모델을 저장합니다.

---

### 훈련 손실이 왜 테스트 손실보다 훨씬 높나요?


케라스 모델은 훈련과 테스트 두 개의 모드(mode)를 가집니다. 드롭아웃(dropout)이나 L1/L2 가중치 규제는 테스트에서 작동하지 않습니다. 훈련 손실에는 반영되지만 테스트 손실에는 반영되지 않습니다(역주-드롭아웃이 손실 함수 계산에 직접 포함되지 않지만 훈련하는 동안 일부 뉴런을 무작위로 제거하기 때문에 일반적으로 성능이 낮아집니다).

또한 훈련 손실은 훈련 데이터의 각 배치에 대한 손실의 평균입니다. 시간이 지남에 따라 모델이 바뀌므로 일반적으로 에포크의 첫 번째 배치 손실이 마지막 배치 손실보다 높습니다. 반면에 한 에포크의 테스트 손실은 에포크가 완료된 모델을 사용해 계산하기 때문에 손실이 더 낮습니다.


---

### How can I use Keras with datasets that don't fit in memory?

You should use the [`tf.data` API](https://www.tensorflow.org/guide/data) to create `tf.data.Dataset` objects -- an abstraction over a data pipeline
that can pull data from local disk, from a distribtued filesystem, from GCS, etc., as well as efficiently apply various data transformations.

For instance, the utility `tf.keras.preprocessing.image_dataset_from_directory` will create a dataset that reads image data from a local directory.

Dataset objects can be directly passed to `fit()`, or can be iterated over in a custom low-level training loop.

```python
model.fit(dataset, epochs=10)
```

---

### How can I regularly save Keras models during training?

To ensure the ability to recover from an interrupted training run at any time (fault tolerance),
you should use a callback that regularly saves your model to disk. You should also set up
your code to optionally reload that model at startup. Here's a simple example.

```python
import os
from tensorflow import keras

# Prepare a directory to store all the checkpoints.
checkpoint_dir = './ckpt'
if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)


def make_model():
    # Create a new linear regression model.
    model = keras.Sequential([keras.layers.Dense(1)])
    model.compile(optimizer='adam', loss='mse')
    return model


def make_or_restore_model():
    # Either restore the latest model, or create a fresh one
    # if there is no checkpoint available.
    checkpoints = [checkpoint_dir + '/' + name
                   for name in os.listdir(checkpoint_dir)]
    if checkpoints:
        latest_checkpoint = max(checkpoints, key=os.path.getctime)
        print('Restoring from', latest_checkpoint)
        return keras.models.load_model(latest_checkpoint)
    print('Creating a new model')
    return make_model()


model = make_or_restore_model()
callbacks = [
    # This callback saves a SavedModel every 100 batches.
    # We include the training loss in the folder name.
    keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_dir + '/ckpt-loss={loss:.2f}',
        save_freq=100)
]
model.fit(train_data, epochs=10, callbacks=callbacks)
```

Find out more in the [callbacks documentation](/api/callbacks/).


---

### How can I interrupt training when the validation loss isn't decreasing anymore?


You can use an `EarlyStopping` callback:

```python
from tensorflow.keras.callbacks import EarlyStopping

early_stopping = EarlyStopping(monitor='val_loss', patience=2)
model.fit(x, y, validation_split=0.2, callbacks=[early_stopping])
```

Find out more in the [callbacks documentation](/api/callbacks/).

---

### How can I freeze layers and do fine-tuning?

**Setting the `trainable` attribute**

All layers & models have a `layer.trainable` boolean attribute:

```shell
>>> layer = Dense(3)
>>> layer.trainable
True
```endshell

On all layers & models, the `trainable` attribute can be set (to True or False).
When set to `False`, the `layer.trainable_weights` attribute is empty:

```python
>>> layer = Dense(3)
>>> layer.build(input_shape=(3, 3)) # Create the weights of the layer
>>> layer.trainable
True
>>> layer.trainable_weights
[<tf.Variable 'kernel:0' shape=(3, 3) dtype=float32, numpy=
array([[...]], dtype=float32)>, <tf.Variable 'bias:0' shape=(3,) dtype=float32, numpy=array([...], dtype=float32)>]
>>> layer.trainable = False
>>> layer.trainable_weights
[]
```

Setting the `trainable` attribute on a layer recursively sets it on all children layers (contents of `self.layers`).


**1) When training with `fit()`:**

To do fine-tuning with `fit()`, you would:

- Instantiate a base model and load pre-trained weights
- Freeze that base model
- Add trainable layers on top
- Call `compile()` and `fit()`

Like this:

```python
model = Sequential([
    ResNet50Base(input_shape=(32, 32, 3), weights='pretrained'),
    Dense(10),
])
model.layers[0].trainable = False  # Freeze ResNet50Base.

assert model.layers[0].trainable_weights == []  # ResNet50Base has no trainable weights.
assert len(model.trainable_weights) == 2  # Just the bias & kernel of the Dense layer.

model.compile(...)
model.fit(...)  # Train Dense while excluding ResNet50Base.
```

You can follow a similar workflow with the Functional API or the model subclassing API.
Make sure to call `compile()` *after* changing the value of `trainable` in order for your
changes to be taken into account. Calling `compile()` will freeze the state of the training step of the model.


**2) When using a custom training loop:**

When writing a training loop, make sure to only update
weights that are part of `model.trainable_weights` (and not all `model.weights`).

```python
model = Sequential([
    ResNet50Base(input_shape=(32, 32, 3), weights='pretrained'),
    Dense(10),
])
model.layers[0].trainable = False  # Freeze ResNet50Base.

# Iterate over the batches of a dataset.
for inputs, targets in dataset:
    # Open a GradientTape.
    with tf.GradientTape() as tape:
        # Forward pass.
        predictions = model(inputs)
        # Compute the loss value for this batch.
        loss_value = loss_fn(targets, predictions)

    # Get gradients of loss wrt the *trainable* weights.
    gradients = tape.gradient(loss_value, model.trainable_weights)
    # Update the weights of the model.
    optimizer.apply_gradients(zip(gradients, model.trainable_weights))
```


**Interaction between `trainable` and `compile()`**

Calling `compile()` on a model is meant to "freeze" the behavior of that model. This implies that the `trainable`
attribute values at the time the model is compiled should be preserved throughout the lifetime of that model,
until `compile` is called again. Hence, if you change `trainable`, make sure to call `compile()` again on your
model for your changes to be taken into account.

For instance, if two models A & B share some layers, and:

- Model A gets compiled
- The `trainable` attribute value on the shared layers is changed
- Model B is compiled

Then model A and B are using different `trainable` values for the shared layers. This mechanism is
critical for most existing GAN implementations, which do:

```python
discriminator.compile(...)  # the weights of `discriminator` should be updated when `discriminator` is trained
discriminator.trainable = False
gan.compile(...)  # `discriminator` is a submodel of `gan`, which should not be updated when `gan` is trained
```



---

### What's the difference between the `training` argument in `call()` and the `trainable` attribute?


`training` is a boolean argument in `call` that determines whether the call
should be run in inference mode or training mode. For example, in training mode,
a `Dropout` layer applies random dropout and rescales the output. In inference mode, the same
layer does nothing. Example:

```python
y = Dropout(0.5)(x, training=True)  # Applies dropout at training time *and* inference time
```

`trainable` is a boolean layer attribute that determines the trainable weights
of the layer should be updated to minimize the loss during training. If `layer.trainable` is set to `False`,
then `layer.trainable_weights` will always be an empty list. Example:

```python
model = Sequential([
    ResNet50Base(input_shape=(32, 32, 3), weights='pretrained'),
    Dense(10),
])
model.layers[0].trainable = False  # Freeze ResNet50Base.

assert model.layers[0].trainable_weights == []  # ResNet50Base has no trainable weights.
assert len(model.trainable_weights) == 2  # Just the bias & kernel of the Dense layer.

model.compile(...)
model.fit(...)  # Train Dense while excluding ResNet50Base.
```

As you can see, "inference mode vs training mode" and "layer weight trainability" are two very different concepts.

You could imagine the following: a dropout layer where the scaling factor is learned during training, via
backpropagation. Let's name it `AutoScaleDropout`.
This layer would have simultaneously a trainable state, and a different behavior in inference and training.
Because the `trainable` attribute and the `training` call argument are independent, you can do the following:

```python
layer = AutoScaleDropout(0.5)

# Applies dropout at training time *and* inference time
# *and* learns the scaling factor during training
y = layer(x, training=True)

assert len(layer.trainable_weights) == 1
```

```python
# Applies dropout at training time *and* inference time
# with a *frozen* scaling factor

layer = AutoScaleDropout(0.5)
layer.trainable = False
y = layer(x, training=True)
```


***Special case of the `BatchNormalization` layer***

Consider a `BatchNormalization` layer in the frozen part of a model that's used for fine-tuning.

It has long been debated whether the moving statistics of the `BatchNormalization` layer should
stay frozen or adapt to the new data. Historically, `bn.trainable = False`
would only stop backprop but would not prevent the training-time statistics
update. After extensive testing, we have found that it is *usually* better to freeze the moving statistics
in fine-tuning use cases. **Starting in TensorFlow 2.0, setting `bn.trainable = False`
will *also* force the layer to run in inference mode.**

This behavior only applies for `BatchNormalization`. For every other layer, weight trainability and
"inference vs training mode" remain independent.



---

### In `fit()`, how is the validation split computed?


If you set the `validation_split` argument in `model.fit` to e.g. 0.1, then the validation data used will be the *last 10%* of the data. If you set it to 0.25, it will be the last 25% of the data, etc. Note that the data isn't shuffled before extracting the validation split, so the validation is literally just the *last* x% of samples in the input you passed.

The same validation set is used for all epochs (within a same call to `fit`).

Note that the `validation_split` option is only available if your data is passed as Numpy arrays (not `tf.data.Datasets`, which are not indexable).


---

### In `fit()`, is the data shuffled during training?

If you pass your data as NumPy arrays and if the `shuffle` argument in `model.fit()` is set to `True` (which is the default), the training data will be globally randomly shuffled at each epoch.

If you pass your data as a `tf.data.Dataset` object and if the `shuffle` argument in `model.fit()` is set ot `True`, the dataset will be locally shuffled (buffered shuffling).

When using `tf.data.Dataset` objects, prefer shuffling your data beforehand (e.g. by calling `dataset = dataset.shuffle(buffer_size)`) so as to be in control of the buffer size.

Validation data is never shuffled.


---

### What's the recommended way to monitor my metrics when training with `fit()`?

Loss values and metric values are reported via the default progress bar displayed by calls to `fit()`.
However, staring at changing ascii numbers in a console ins't an optimal metric-monitoring experience.
We recommend the use of [TensorBoard](https://www.tensorflow.org/tensorboard), which will display nice-looking graphs of your training and validation metrics, regularly
updated during training, which you can access from your browser.

You can use TensorBoard with `fit()` via the [`TensorBoard` callback](/api/callbacks/tensorboard/).

---

### What if I need to customize what `fit()` does?

You have two options:

**1) Write a low-level custom training looop**

This is a good option if you want to be in control of every last little detail. But it can be somewhat verbose. Example:

```python
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

This examples does not include a lot of essential functionality like displaying a progress bar, calling callbacks,
updating metrics, etc. You would have to do this yourself. It's not difficult at all, but it's a bit of work.


**2) Subclass the `Model` class and override the `train_step` (and `test_step`) methods**

This is a better option if you want to use custom update rules but still want to leverage the functionality provided by `fit()`,
such as callbacks, efficient step fusing, etc.

Note that this pattern does not prevent you from building models with the Functional API (or even Sequential models).

The example below shows a Functional model with a custom `train_step`.

```python
from tensorflow import keras
import tensorflow as tf
import numpy as np

class MyCustomModel(keras.Model):

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


# Construct and compile an instance of MyCustomModel
inputs = keras.Input(shape=(32,))
outputs = keras.layers.Dense(1)(inputs)
model = MyCustomModel(inputs, outputs)
model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])

# Just use `fit` as usual
x = np.random.random((1000, 32))
y = np.random.random((1000, 1))
model.fit(x, y, epochs=10)
```

You can also easily add support for sample weighting:

```python
class MyCustomModel(keras.Model):

    def train_step(self, data):
        # Unpack the data. Its structure depends on your model and
        # on what you pass to `fit()`.
        if len(data) == 3:
            x, y, sample_weight = data
        else:
            x, y = data

        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)  # Forward pass
            # Compute the loss value.
            # The loss function is configured in `compile()`.
            loss = self.compiled_loss(y, y_pred,
                                      sample_weight=sample_weight,
                                      regularization_losses=self.losses)

        # Compute gradients
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)

        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        # Update the metrics.
        # Metrics are configured in `compile()`.
        self.compiled_metrics.update_state(
            y, y_pred, sample_weight=sample_weight)

        # Return a dict mapping metric names to current value.
        # Note that it will include the loss (tracked in self.metrics).
        return {m.name: m.result() for m in self.metrics}


# Construct and compile an instance of MyCustomModel
inputs = keras.Input(shape=(32,))
outputs = keras.layers.Dense(1)(inputs)
model = MyCustomModel(inputs, outputs)
model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])

# You can now use sample_weight argument
x = np.random.random((1000, 32))
y = np.random.random((1000, 1))
sw = np.random.random((1000, 1))
model.fit(x, y, sample_weight=sw, epochs=10)
```

Similarly, you can also customize evaluation by overriding `test_step`:

```python
class MyCustomModel(keras.Model):

    def test_step(self, data):
      # Unpack the data
      x, y = data
      # Compute predictions
      y_pred = self(x, training=False)
      # Updates the metrics tracking the loss
      self.compiled_loss(
          y, y_pred, regularization_losses=self.losses)
      # Update the metrics.
      self.compiled_metrics.update_state(y, y_pred)
      # Return a dict mapping metric names to current value.
      # Note that it will include the loss (tracked in self.metrics).
      return {m.name: m.result() for m in self.metrics}
```

---

### How can I train models in mixed precision?

Keras has built-in support for mixed precision training on GPU and TPU.
See [this extensive guide](https://www.tensorflow.org/guide/keras/mixed_precision).

---

## 모델링과 관련된 질문


### How can I obtain the output of an intermediate layer (feature extraction)?

In the Functional API and Sequential API, if a layer has been called exactly once, you can retrieve its output via `layer.output` and its input via `layer.input`.
This enables you do quickly instantiate feature-extraction models, like this one:

```python
from tensorflow import keras
from tensorflow.keras import layers

model = Sequential([
    layers.Conv2D(32, 3, activation='relu'),
    layers.Conv2D(32, 3, activation='relu'),
    layers.MaxPooling2D(2),
    layers.Conv2D(32, 3, activation='relu'),
    layers.Conv2D(32, 3, activation='relu'),
    layers.GlobalMaxPooling2D(),
    layers.Dense(10),
])
extractor = keras.Model(inputs=model.inputs,
                        outputs=[layer.output for layer in model.layers])
features = extractor(data)
```

Naturally, this is not possible with models that are subclasses of `Model` that override `call`.

Here's another example: instantiating a `Model` that returns the output of a specific named layer:

```python
model = ...  # create the original model

layer_name = 'my_layer'
intermediate_layer_model = keras.Model(inputs=model.input,
                                       outputs=model.get_layer(layer_name).output)
intermediate_output = intermediate_layer_model(data)
```

---

### How can I use pre-trained models in Keras?

You could leverage the [models available in `keras.applications`](/api/applications/), or the models available on [TensorFlow Hub](https://www.tensorflow.org/hub).
TensorFlow Hub is well-integrated with Keras.

---

### How can I use stateful RNNs?


Making a RNN stateful means that the states for the samples of each batch will be reused as initial states for the samples in the next batch.

When using stateful RNNs, it is therefore assumed that:

- all batches have the same number of samples
- If `x1` and `x2` are successive batches of samples, then `x2[i]` is the follow-up sequence to `x1[i]`, for every `i`.

To use statefulness in RNNs, you need to:

- explicitly specify the batch size you are using, by passing a `batch_size` argument to the first layer in your model. E.g. `batch_size=32` for a 32-samples batch of sequences of 10 timesteps with 16 features per timestep.
- set `stateful=True` in your RNN layer(s).
- specify `shuffle=False` when calling `fit()`.

To reset the states accumulated:

- use `model.reset_states()` to reset the states of all layers in the model
- use `layer.reset_states()` to reset the states of a specific stateful RNN layer

Example:

```python
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np

x = np.random.random((32, 21, 16))  # this is our input data, of shape (32, 21, 16)
# we will feed it to our model in sequences of length 10

model = keras.Sequential()
model.add(layers.LSTM(32, input_shape=(10, 16), batch_size=32, stateful=True))
model.add(layers.Dense(16, activation='softmax'))

model.compile(optimizer='rmsprop', loss='categorical_crossentropy')

# we train the network to predict the 11th timestep given the first 10:
model.train_on_batch(x[:, :10, :], np.reshape(x[:, 10, :], (32, 16)))

# the state of the network has changed. We can feed the follow-up sequences:
model.train_on_batch(x[:, 10:20, :], np.reshape(x[:, 20, :], (32, 16)))

# let's reset the states of the LSTM layer:
model.reset_states()

# another way to do it in this case:
model.layers[0].reset_states()
```

Note that the methods `predict`, `fit`, `train_on_batch`, `predict_classes`, etc. will *all* update the states of the stateful layers in a model. This allows you to do not only stateful training, but also stateful prediction.


---

