# 케라스 생태계

케라스 프로젝트는 신경망을 만들고 훈련하는 케라스의 핵심 API에 국한되지 않습니다.
다양한 프로젝트들이 머신러닝 워크플로의 모든 단계를 커버하고 있습니다.

---

## 케라스 튜너

[케라스 튜너 문서](https://keras-team.github.io/keras-tuner/) - [케라스 튜너 깃허브 저장소](https://github.com/keras-team/keras-tuner)


케라스 튜너(Tuner)는 번거로운 하이퍼파라미터 탐색 문제를 해결해 주며 사용하기 쉽고 확장이 용이한 하이퍼파라미터 최적화 프레임워크입니다.
실행 기반 정의(define-by-run) 방식으로 탐색 공간을 쉽게 설정하고 탐색 알고리즘 중 하나를 선택하여 모델에 맞는 최적의 하이퍼파라미터를 찾습니다.
케라스 튜너는 베이지안 최적화(Bayesian Optimization), 하이퍼밴드(Hyperband), 랜덤 탐색 알고리즘을 기본으로 제공합니다.
또한 연구자가 새로운 탐색 알고리즘을 실험하기 위해서 쉽게 확장할 수 있도록 설계되었습니다.

---

## 오토케라스

[오토케라스 문서](https://autokeras.com/) - [오토케라스 깃허브 저장소](https://github.com/keras-team/autokeras)

오토케라스(AutoKeras)는 케라스 기반의 AutoML 시스템입니다.
텍사스 A&M 대학교의 [데이터 랩(DATA Lab)](http://faculty.cs.tamu.edu/xiahu/index.html)에서 개발했습니다.
오토케라스의 목표는 모든 사람이 쉽게 머신러닝을 사용하도록 만드는 것입니다.
[`ImageClassifier`](https://autokeras.com/tutorial/image_classification/)이나
[`TextClassifier`](https://autokeras.com/tutorial/text_classification/)와 같은 고수준 엔드-투-엔드 API를 제공하여 몇 줄의 코드로 머신러닝 문제를 풀 수 있습니다.
또한 신경망 구조를 탐색하는데 [유연한 구성 요소](https://autokeras.com/tutorial/customized/)를 제공합니다.

---

## 텐서플로 클라우드

구글의 케라스 팀에서 관리되는 [텐서플로 클라우드]()는 아주 약간만의 설정으로 GCP에서 대규모 케라스 훈련을 실행하도록 도와주는 유틸리티 모음입니다.
클라우드에 있는 8개 이상의 GPU에서 모델을 실험하려면 `model.fit()`을 호출하기만 하면 됩니다.


---

## TensorFlow.js

[TensorFlow.js](https://www.tensorflow.org/js)는 텐서플로의 자바스크립트 런타임(runtime)입니다.
훈련과 추론을 위해 브라우저나 [Node.js](https://nodejs.org/en/) 서버에서 텐서플로 모델을 실행할 수 있습니다.
케라스 모델을 로드하고 브라우저에서 직접 케라스 모델을 미세 조정하거나 재훈련할 수 있습니다.


---

## 텐서플로 라이트

[텐서플로 라이트(Lite)](https://www.tensorflow.org/lite)는 효율적인 온-디바이스(on-device) 추론을 수행할 수 있는 런타임입니다.
텐서플로 라이트는 케라스 모델을 지원하고 안드로이드, iOS, 임베디드(embeded) 장치에 모델을 배포할 수 있습니다.


---

## 모델 최적화 툴킷

[텐서플로 모델 최적화 툴킷(Toolkit)](https://www.tensorflow.org/model_optimization)은 *훈련된 가중치 양자화*(post-training weight quantization)나 *가지치기를 고려한 훈련*(pruning-aware training)을 사용해 추론 모델을 더 빠르고, 더 메모리 효율적이고, 더 전력 효율적으로 만드는 유틸리티 모음입니다.


---

## TFX 통합

TFX는 머신러닝 제품 파이프라인(pipeline)을 배포하고 유지하기 위한 엔드-투-엔드 플랫폼입니다.
TFX는 [케라스 모델을 기본으로 지원합니다](https://www.tensorflow.org/tfx/guide/keras).


---



