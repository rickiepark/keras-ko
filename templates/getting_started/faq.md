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
- [케라스에서는 메모리 용량보다 큰 데이터셋을 어떻게 사용하나요?](#케라스에서는-메모리-용량보다-큰-데이터셋을-어떻게-사용하나요)
- [훈련 도중에 주기적으로 케라스 모델을 저장하는 방법은 무엇인가요?](#훈련-도중에-주기적으로-케라스-모델을-저장하는-방법은-무엇인가요)
- [검증 손실이 더 이상 감소하지 않을 때 어떻게 훈련을 중단할 수 있나요?](#검증-손실이-더-이상-감소하지-않을-때-어떻게-훈련을-중단할-수-있나요)
- [어떻게 층을 동결하고 미세 조정할 수 있나요?](#어떻게-층을-동결하고-미세-조정할-수-있나요)
- [`call()` 메서드의 `training` 매개변수와 `trainable` 속성의 차이점은 무엇인가요?](#call-메서드의-training-매개변수와-trainable-속성의-차이점은-무엇인가요)
- [`fit()` 메서드에서 검증 세트는 어떻게 계산하나요?](#fit-메서드에서-검증-세트는-어떻게-계산하나요)
- [`fit()` 메서드에서 훈련하는 동안 데이터를 섞나요?](#fit-메서드에서-훈련하는-동안-데이터를-섞나요)
- [`fit()`으로 훈련할 때 측정 값을 어떻게 모니터링하는 것이 좋을까요?](#fit으로-훈련할-때-측정-값을-어떻게-모니터링하는-것이-좋을까요)
- [어떻게 `fit()` 메서드를 커스터마이징할 수 있나요?](#어떻게-fit-메서드를-커스터마이징할-수-있나요)
- [어떻게 모델을 혼합 정밀도로 훈련할 수 있나요?](#어떻게-모델을-혼합-정밀도로-훈련할-수-있나요)

## 모델링과 관련된 질문

- [어떻게 중간층의 출력(특성 추출)을 얻을 수 있나요?](#어떻게-중간층의-출력특성-추출을-얻을-수-있나요)
- [케라스에서 사전 훈련된 모델을 사용할 수 있나요?](#케라스에서-사전-훈련된-모델을-사용할-수-있나요)
- [상태가 있는 RNN을 어떻게 사용하나요?](#상태가-있는-RNN을-어떻게-사용하나요)


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


케라스 모델은 훈련과 테스트 두 개의 모드(mode)를 가집니다. 드롭아웃(dropout)이나 L1/L2 가중치 규제는 테스트에서 작동하지 않습니다. 훈련 손실에는 반영되지만 테스트 손실에는 반영되지 않습니다(**옮긴이**-드롭아웃이 손실 함수 계산에 직접 포함되지 않지만 훈련하는 동안 일부 뉴런을 무작위로 제거하기 때문에 일반적으로 성능이 낮아집니다).

또한 케라스가 출력하는 훈련 손실은 **현재 에포크에서** 훈련 데이터의 각 배치에 대한 손실의 평균입니다. 시간이 지남에 따라 모델이 바뀌므로 일반적으로 에포크의 첫 번째 배치 손실이 마지막 배치 손실보다 높습니다. 이는 에포크별 평균을 낮출 수 있습니다. 반면에 한 에포크의 테스트 손실은 에포크가 완료된 모델을 사용해 계산하기 때문에 손실이 더 낮습니다.


---

### 케라스에서는 메모리 용량보다 큰 데이터셋을 어떻게 사용하나요?

[`tf.data` API](https://www.tensorflow.org/guide/data)를 사용해 `tf.data.Dataset` 객체를 만들어야 합니다. 이 객체는 일종의 데이터 파이프라인 추상화이며 로컬 디스크, 분산 파일 시스템, GCS(Google Cloud Storage) 등에서 데이터를 읽을 수 있습니다. 또한 다양한 데이터 변환 작업을 효율적으로 수행할 수 있습니다.

예를 들어 [`tf.keras.preprocessing.image_dataset_from_directory`](https://keras.io/api/preprocessing/image/#imagedatasetfromdirectory-function) 유틸리티는 로컬 디렉토리에서 이미지 데이터를 읽는 데이터셋을 만듭니다.
비슷하게 [`tf.keras.preprocessing.text_dataset_from_directory`](https://keras.io/api/preprocessing/text/#textdatasetfromdirectory-function) 유틸리티는 로컬 디렉토리에서 텍스트 파일을 읽는 데이터셋을 만듭니다.

데이터셋 객체는 `fit()` 메서드에 직접 전달하거나 사용자가 정의한 저수준 훈련 반복문에서 사용할 수 있습니다.

```python
model.fit(dataset, epochs=10, validation_data=val_dataset)
```

---

### 훈련 도중에 주기적으로 케라스 모델을 저장하는 방법은 무엇인가요?

언제든지 예상치 못한 훈련 중단으로부터 복구(장애 허용 능력(fault tolerance))할 수 있으려면 콜백을 사용해 모델을 주기적으로 디스크에 저장해야 합니다. 훈련을 시작할 때 모델을 다시 적재할 수 있도록 코드를 만들어야 합니다. 간단한 예는 다음과 같습니다.

```python
import os
from tensorflow import keras

# 체크포인트를 저장할 디렉토리를 준비합니다.
checkpoint_dir = './ckpt'
if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)


def make_model():
    # 새로운 선형 회귀 모델을 만듭니다.
    model = keras.Sequential([keras.layers.Dense(1)])
    model.compile(optimizer='adam', loss='mse')
    return model


def make_or_restore_model():
    # 최신 모델을 복원하거나 체크포인트가 없으면 새로운 모델을 만듭니다.
    checkpoints = [checkpoint_dir + '/' + name
                   for name in os.listdir(checkpoint_dir)]
    if checkpoints:
        latest_checkpoint = max(checkpoints, key=os.path.getctime)
        print('복원한 체크포인트:', latest_checkpoint)
        return keras.models.load_model(latest_checkpoint)
    print('새로운 모델 생성')
    return make_model()


model = make_or_restore_model()
callbacks = [
    # 이 콜백은 100번의 배치마다 SavedModel 파일을 저장합니다.
    # 파일 이름에 훈련 손실이 포함되었습니다.
    keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_dir + '/ckpt-loss={loss:.2f}',
        save_freq=100)
]
model.fit(train_data, epochs=10, callbacks=callbacks)
```

더 자세한 내용은 [콜백 문서](/api/callbacks/)를 참고하세요.


---

### 검증 손실이 더 이상 감소하지 않을 때 어떻게 훈련을 중단할 수 있나요?


`EarlyStopping` 콜백을 사용할 수 있습니다:

```python
from tensorflow.keras.callbacks import EarlyStopping

early_stopping = EarlyStopping(monitor='val_loss', patience=2)
model.fit(x, y, validation_split=0.2, callbacks=[early_stopping])
```

더 자세한 내용은 [콜백 문서](/api/callbacks/)를 참고하세요.

---

### 어떻게 층을 동결하고 미세 조정할 수 있나요?

**`trainable` 속성을 설정합니다.**

모든 층과 모델은 `layer.trainable` 불리언 속성을 가지고 있습니다:

```shell
>>> layer = Dense(3)
>>> layer.trainable
True
```endshell

모든 층과 모델에서 `trainable` 속성을 설정할 수 있습니다(`True` 또는 `False`).
`False`로 설정하면 `layer.trainable_weights` 속성이 비워집니다:

```python
>>> layer = Dense(3)
>>> layer.build(input_shape=(3, 3)) # 층의 가중치가 만들어 집니다.
>>> layer.trainable
True
>>> layer.trainable_weights
[<tf.Variable 'kernel:0' shape=(3, 3) dtype=float32, numpy=
array([[...]], dtype=float32)>, <tf.Variable 'bias:0' shape=(3,) dtype=float32, numpy=array([...], dtype=float32)>]
>>> layer.trainable = False
>>> layer.trainable_weights
[]
```

한 층의 `trainable` 속성을 설정하면 재귀적으로 (`self.layers`에 담긴) 모든 하위 층을 설정합니다.


**1) `fit()` 메서드로 훈련할 때:**

`fit()` 메서드로 미세 조정(fine-tuning)을 수행하려면 다음 과정을 따릅니다:

- 기반 모델(base model)의 객체를 만들고 사전 훈련된 가중치를 적재합니다.
- 기반 모델을 동결합니다.
- 맨 위에 훈련 가능한 층을 추가합니다.
- `compile()` 메서드와 `fit()` 메서드를 호출합니다.

다음과 같습니다:

```python
model = Sequential([
    ResNet50Base(input_shape=(32, 32, 3), weights='pretrained'),
    Dense(10),
])
model.layers[0].trainable = False  # ResNet50Base 동결

assert model.layers[0].trainable_weights == []  # ResNet50Base는 훈련 가능한 가중치가 없습니다.
assert len(model.trainable_weights) == 2  # Dense 층의 커널과 절편만 있습니다.

model.compile(...)
model.fit(...)  # ResNet50Base를 제외하고 Dense 층만 훈련합니다.
```

함수형 API나 모델 서브클래싱 API에서 비슷한 작업 흐름을 따를 수 있습니다.
`trainable` 속성 값을 바꾼 *후에* 반드시 `compile()` 메서드를 호출해야 변경 사항이 반영됩니다.
`compile()` 메서드가 모델의 훈련 단계 상태를 바꿉니다.


**2) 사용자 정의 반복을 사용할 때:**

훈련 반복을 만들 때는 `model.trainable_weights` 가중치만 업데이트해야 합니다(`model.weights` 전체가 아닙니다).

```python
model = Sequential([
    ResNet50Base(input_shape=(32, 32, 3), weights='pretrained'),
    Dense(10),
])
model.layers[0].trainable = False  # ResNet50Base 동결

# 데이터셋의 배치를 반복합니다.
for inputs, targets in dataset:
    # Open a GradientTape을 시작합니다.
    with tf.GradientTape() as tape:
        # 정방향 계산을 수행합니다.
        predictions = model(inputs)
        # 배치에 대한 손실 값을 계산합니다.
        loss_value = loss_fn(targets, predictions)

    # *훈련 가능한* 가중치에 대한 손실의 그레이디언트를 계산합니다.
    gradients = tape.gradient(loss_value, model.trainable_weights)
    # 모델의 가중치를 업데이트합니다.
    optimizer.apply_gradients(zip(gradients, model.trainable_weights))
```


**`trainable` 속성과 `compile()` 메서드 간의 상호작용**

모델의 `compile()` 메서드를 호출하면 모델의 행동을 "동결"한다는 의미입니다.
모델을 컴파일할 당시 `trainable` 속성의 값이 다시 `compile()` 메서드를 호출하기 전까지 모델의 전체 생명주기 동안 보존된다는 뜻입니다.
따라서 `trainable` 속성을 바꾸려면 모델의 `compile()` 메서드를 다시 호출해서 변경 사항을 반영해야 합니다.

예를 들어, 두 모델 A와 B가 어떤 층을 공유할 때:

- 모델 A를 컴파일합니다.
- 공유 층의 `trainable` 속성 값이 바뀝니다.
- 모델 B를 컴파일합니다.

이 경우 모델 A와 B는 공유 층에 대해 다른 `trainable` 값을 사용합니다.
이런 메카니즘은 GAN을 구현할 때 아주 중요합니다:

```python
discriminator.compile(...)  # `discriminator`의 가중치는 `discriminator`가 훈련될 때 업데이트됩니다.
discriminator.trainable = False
gan.compile(...)  # `discriminator`는 `gan`의 서브 모델이지만 `gan`이 훈련될 때 업데이트되지 않습니다.
```



---

### `call()` 메서드의 `training` 매개변수와 `trainable` 속성의 차이점은 무엇인가요?


`training`은 `call()` 메서드이 불리언 매개변수로 이 메서드 호출을 추론 모드로 실행할지 훈련 모드로 실행할지 결정합니다. 예를 들어 훈련 모드에서 `Dropout` 층은 랜덤한 드롭아웃을 적용하고 출력의 스케일을 조정합니다. 추론 모드에서는 이 층은 아무런 역할을 수행하지 않습니다. 예를 들어:

```python
y = Dropout(0.5)(x, training=True)  # 훈련 *그리고* 추론 시에 드롭아웃을 적용합니다.
```

층의 `trainable` 불리언 속성은 훈련하는 동안 손실을 최소화하기 위해 층의 훈련 가능한 가중치를 업데이트할지 결정합니다. `layer.trainable`을 `False`로 지정하면 `layer.trainable_weights`는 항상 빈 리스트가 됩니다. 예를 들어:

```python
model = Sequential([
    ResNet50Base(input_shape=(32, 32, 3), weights='pretrained'),
    Dense(10),
])
model.layers[0].trainable = False  # ResNet50Base 동결

assert model.layers[0].trainable_weights == []  # ResNet50Base는 훈련 가능한 가중치가 없습니다.
assert len(model.trainable_weights) == 2  # Dense 층의 커널과 절편만 있습니다.

model.compile(...)
model.fit(...)  # ResNet50Base을 제외한 Dense 층만 훈련합니다.
```

여기서 볼 수 있듯이 "추론 모드 vs 훈련 모드"와 "층의 가중치 훈련 여부"는 다른 개념입니다.

훈련하는 동안 역전파를 통해 스케일링 비율을 학습하는 드롭아웃 층이 있다고 가정해 보죠.
이 층의 이름을 `AutoScaleDropout`이라 하겠습니다.
이 층은 추론과 훈련 모드에서 동작이 다르고 훈련 가능한 변수도 가지고 있습니다.
`trainable` 속성과 `training` 매개변수는 독립적이기 때문에 다음처럼 사용할 수 있습니다:

```python
layer = AutoScaleDropout(0.5)

# 훈련 모드와 추론 모드에 드롭아웃을 적용합니다.
# 또한 훈련하는 동안 스케일링 비율을 학습니다.
y = layer(x, training=True)

assert len(layer.trainable_weights) == 1
```

```python
# 훈련 모드와 추론 모드에 드롭아웃을 적용합니다.
# 스케일링 비율을 동결합니다.

layer = AutoScaleDropout(0.5)
layer.trainable = False
y = layer(x, training=True)
```


***특별한 `BatchNormalization` 층의 경우***

모델을 미세 조정할 때 동결 부분에 있는 `BatchNormalization` 층을 생각해 보죠.
Consider a `BatchNormalization` layer in the frozen part of a model that's used for fine-tuning.

`BatchNormalization` 층의 이동 통곗값(moving statistic)을 새로운 데이터에 동결할지 변경할지 오랫동안 논란이 있었습니다.
과거에 `bn.trainable = False`는 역전파만 중지하고 훈련시 통곗값 업데이트를 막지 않습니다.
광범위한 테스트 끝에 미세 조정의 경우 이동 통곗값을 동결하는 것이 *일반적으로* 좋다는 것을 알았습니다.
**텐서플로 2.0부터는 `bn.trainable = False`로 지정하면 이 층을 추론 모드에서 실행합니다.**

이런 동작은 `BatchNormalization` 층에만 적용됩니다. 다른 모든 층은 가중치 학습 여부와 "추론 모드 vs 훈련 모드"가 독립적으로 구분됩니다.



---

### `fit()` 메서드에서 검증 세트는 어떻게 계산하나요?

`model.fit` 메서드에서 `validation_split` 매개변수를 (가령 0.1로) 지정하면, 데이터의 *마지막 10%*를 검증 세트로 사용합니다. 0.25로 지정하면 데이터의 마지막 25%를 사용하는 식입니다. 검증 데이터를 뽑아내기 전에 데이터를 섞지 않습니다. 따라서 검증 세트는 글자 그대로 입력 샘플의 *마지막* x%가 됩니다.

동일한 검증 세트가 (한 번의 `fit` 호출 안의) 모든 에포크에 사용됩니다.

`validation_split` 옵션은 데이터가 넘파이 배열로 전달될 때만 사용할 수 있습니다(`tf.data.Datasets`는 인덱스 참조가 안됩니다).


---

### `fit()` 메서드에서 훈련하는 동안 데이터를 섞나요?

입력 데이터가 넘파이 배열이고 `model.fit()` 메서드의 `shuffle` 매개변수가 `True`(기본값입니다)이면 에포크마다 훈련 데이터를 랜덤하게 셔플링합니다.

데이터가 `tf.data.Dataset` 객체이고 `model.fit()` 메서드의 `shuffle` 매개변수가 `True`이면 데이터셋은 지역적으로 셔플링됩니다(버퍼 셔플링).

`tf.data.Dataset` 객체를 사용할 때 버퍼 크기에 제한이 있기 때문에 사전에 데이터를 섞는 것이 좋습니다(예를 들어 `dataset = dataset.shuffle(buffer_size)`).

검증 데이터는 셔플링되지 않습니다.


---

### `fit()`으로 훈련할 때 측정 값을 어떻게 모니터링하는 것이 좋을까요?

손실 값과 측정 지표 값은 `fit()` 메서드가 출력하는 기본적인 진행 막대를 통해 제공됩니다.
하지만 콘솔에서 바뀌는 숫자를 들여다 보는 것은 좋은 모니터링 방법이 아닙니다.
[텐서보드](https://www.tensorflow.org/tensorboard)를 사용하면 훈련 지표와 검증 지표를 훈련하는 동안 일정 간격으로 업데이트하여 멋진 그래프로 출력하고 브라우저에서 접속할 수 있습니다.

이를 위해 `fit()` 메서드에 [`TensorBoard` 콜백](/api/callbacks/tensorboard/)을 사용합니다.

---

### 어떻게 `fit()` 메서드를 커스터마이징할 수 있나요?

두 가지 방법이 있습니다:

**1) 저수준 사용자 정의 훈련 루프를 만듭니다**

상세한 모든 것을 제어하고 싶을 때 좋습니다. 대신 조금 번거롭습니다. 예를 들면:

```python
# 옵티마이저를 준비합니다.
optimizer = tf.keras.optimizers.Adam()
# 손실 함수를 준비합니다.
loss_fn = tf.keras.losses.kl_divergence

# 데이터셋의 배치를 순회합니다.
for inputs, targets in dataset:
    # GradientTape를 시작합니다.
    with tf.GradientTape() as tape:
        # 정방향 계산.
        predictions = model(inputs)
        # 배치에 대한 손실을 계산합니다.
        loss_value = loss_fn(targets, predictions)

    # 가중치에 대한 손실의 그레이디언트를 계산합니다.
    gradients = tape.gradient(loss_value, model.trainable_weights)
    # 모델의 가중치를 업데이트합니다.
    optimizer.apply_gradients(zip(gradients, model.trainable_weights))
```

이 예제는 진행 막대를 출력하거나, 콜백을 호출하고, 측정 지표를 업데이트하는 등의 기본적인 기능을 포함하고 있지 않습니다. 직접 만들어 보세요. 어렵지는 않지만 약간의 작업이 필요합니다.

**2) `Model` 클래스를 상속하고 `train_step`(그리고 `test_step`) 메서드를 오버라이딩합니다**

이 방법은 가중치 업데이트 규칙을 바꾸고 싶지만 `fit()` 메서에서 제공하는 콜백이나 효율적인 스텝 융합(step fusing) 같은 기능을 사용하고 싶을 때 좋습니다.

이 방법을 사용하면 함수형 API를 사용해 모델을 만들 수 있습니다(시퀀셜 모델도 사용할 수 있습니다).

다음은 사용자 정의 `train_step`과 함께 함수형 API를 사용하는 예입니다.

```python
from tensorflow import keras
import tensorflow as tf
import numpy as np

class MyCustomModel(keras.Model):

    def train_step(self, data):
        # 데이터를 언팩합니다. 데이터 구조는 모델에 따라 다르고
        # `fit()` 메서드에 전달한 값에 따라 다릅니다.
        x, y = data

        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)  # 정방향 계산
            # 손실을 계산합니다
            # (손실 함수는 `compile()`에서 설정합니다)
            loss = self.compiled_loss(y, y_pred,
                                      regularization_losses=self.losses)

        # 그레이디언트를 계산합니다
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        # 가중치를 업데이트합니다
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        # (손실을 포함하여) 측정 지표를 업데이트합니다
        self.compiled_metrics.update_state(y, y_pred)
        # 측정 지표와 현재 값을 매핑한 딕셔너리를 반환합니다
        return {m.name: m.result() for m in self.metrics}


# MyCustomModel의 객체를 만들고 컴파일합니다
inputs = keras.Input(shape=(32,))
outputs = keras.layers.Dense(1)(inputs)
model = MyCustomModel(inputs, outputs)
model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])

# 평상시처럼 `fit` 메서드를 사용합니다
x = np.random.random((1000, 32))
y = np.random.random((1000, 1))
model.fit(x, y, epochs=10)
```

샘플 가중치도 쉽게 적용할 수 있습니다:

```python
class MyCustomModel(keras.Model):

    def train_step(self, data):
        # 데이터를 언팩합니다. 데이터 구조는 모델에 따라 다르고
        # `fit()` 메서드에 전달한 값에 따라 다릅니다.
        if len(data) == 3:
            x, y, sample_weight = data
        else:
            x, y = data

        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)  # 정방향 계산
            # 손실을 계산합니다
            # (손실 함수는 `compile()`에서 설정합니다)
            loss = self.compiled_loss(y, y_pred,
                                      sample_weight=sample_weight,
                                      regularization_losses=self.losses)

        # 그레이디언트를 계산합니다
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)

        # 가중치를 업데이트합니다
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        # 측정 지표를 업데이트합니다
        # 측정 지표는 `compile()`에서 설정합니다
        self.compiled_metrics.update_state(
            y, y_pred, sample_weight=sample_weight)

        # 측정 지표와 현재 값을 매핑한 딕셔너리를 반환합니다
        # 여기에는 손실도 포함됩니다 (self.metrics에 기록되어 있습니다)
        return {m.name: m.result() for m in self.metrics}


# MyCustomModel의 객체를 만들고 컴파일합니다
inputs = keras.Input(shape=(32,))
outputs = keras.layers.Dense(1)(inputs)
model = MyCustomModel(inputs, outputs)
model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])

# 이제 sample_weight 매개변수를 사용할 수 있습니다
x = np.random.random((1000, 32))
y = np.random.random((1000, 1))
sw = np.random.random((1000, 1))
model.fit(x, y, sample_weight=sw, epochs=10)
```

비슷하게 `test_step` 오버라이딩하여 사용자 평가를 구현할 수 있습니다:

```python
class MyCustomModel(keras.Model):

    def test_step(self, data):
      # 데이터를 언팩합니다
      x, y = data
      # 예측을 만듭니다
      y_pred = self(x, training=False)
      # 손실 지표를 업데이트합니다
      self.compiled_loss(
          y, y_pred, regularization_losses=self.losses)
      # 측정 지표를 업데이트합니다
      self.compiled_metrics.update_state(y, y_pred)
      # 측정 지표와 현재 값을 매핑한 딕셔너리를 반환합니다
      # 여기에는 손실도 포함됩니다 (self.metrics에 기록되어 있습니다)
      return {m.name: m.result() for m in self.metrics}
```

---

### 어떻게 모델을 혼합 정밀도로 훈련할 수 있나요?

케라스는 기본적으로 GPU와 TPU에서 혼합 정밀도 훈련을 지원합니다. 자세한 내용은 [가이드 문서](https://www.tensorflow.org/guide/keras/mixed_precision)를 참고하세요.

---

## 모델링과 관련된 질문


### 어떻게 중간층의 출력(특성 추출)을 얻을 수 있나요?

함수형 API와 시퀀셜 API에서 정확히 층이 한 번 호출되면 `layer.output`으로 출력을 얻고 `layer.input`으로 입력을 얻을 수 있습니다. 이를 사용하면 다음처럼 특성 추출 모델을 간단히 만들 수 있습니다:

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

당연히 `Model`을 서브클래싱하고 `call` 메서드를 오버라이딩하는 모델로는 불가능합니다.

다음은 또 다른 예시입니다: 특정 층의 출력을 반환하는 `Model` 객체를 만듭니다:

```python
model = ...  # 원본 모델을 만듭니다

layer_name = 'my_layer'
intermediate_layer_model = keras.Model(inputs=model.input,
                                       outputs=model.get_layer(layer_name).output)
intermediate_output = intermediate_layer_model(data)
```

---

### 케라스에서 사전 훈련된 모델을 사용할 수 있나요?

[`keras.applications`에 있는 모델](/api/applications/)이나 [텐서플로 허브(Hub)](https://www.tensorflow.org/hub)에 있는 모델을 사용할 수 있습니다. 텐서플로 허브는 케라스와 잘 통합됩니다.

---

### 상태가 있는 RNN을 어떻게 사용하나요?

RNN이 상태를 가진다는 것은 각 배치의 샘플에 대한 상태가 다음 배치에 있는 샘플의 초기 상태로 재사용되는 것을 의미합니다.

따라서 상태가 있는 RNN을 사용할 때 다음을 가정합니다:

- 모든 배치는 샘플의 개수가 동일합니다.
- `x1`과 `x2`가 연속된 배치라면 모든 `i`에 대해 `x2[i]`는 `x1[i]`의 뒤를 잇습니다.

상태가 있는 RNN을 사용하려면 다음이 필요합니다:

- 모델의 첫 번째 층에 `batch_size` 매개변수로 사용할 배치 크기를 명시적으로 지정해야 합니다. 예를 들어 타임스텝마다 16개의 특성을 가지고 10개의 타임스텝으로 구성된 32개 샘플의 배치라면 `batch_size=32`라고 지정합니다.
- RNN 층을 `stateful=True`으로 지정합니다.
- `fit()` 메서드를 호출할 때 `shuffle=False`로 설정합니다.

누적된 상태를 초기화하려면:

- 모델에 있는 모든 층의 상태를 초기화하려면 `model.reset_states()`를 사용합니다.
- 특정 RNN 층의 상태를 초기화하려면 `layer.reset_states()`를 사용합니다.

예:

```python
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np

x = np.random.random((32, 21, 16))  # (32, 21, 16) 크기의 입력 데이터
# 길이가 10인 시퀀스로 모델에 주입합니다.

model = keras.Sequential()
model.add(layers.LSTM(32, input_shape=(10, 16), batch_size=32, stateful=True))
model.add(layers.Dense(16, activation='softmax'))

model.compile(optimizer='rmsprop', loss='categorical_crossentropy')

# 처음 10개의 타입스텝이 주어지면 11번째 타임스텝을 예측하는 모델을 훈련합니다:
model.train_on_batch(x[:, :10, :], np.reshape(x[:, 10, :], (32, 16)))

# 모델의 상태가 바뀌었습니다. 다음 시퀀스를 주입할 수 있습니다:
model.train_on_batch(x[:, 10:20, :], np.reshape(x[:, 20, :], (32, 16)))

# LSTM 층의 상태를 초기화해보죠:
model.reset_states()

# 초기화하는 다른 방법입니다:
model.layers[0].reset_states()
```

`predict`, `fit`, `train_on_batch`, `predict_classes`과 같은 메서드는 *모두* 상태가 있는 층의 상태를 업데이트합니다. 따라서 상태가 있는 훈련뿐만 아니라 상태가 있는 예측도 가능합니다.


---

