YBIGTA 신입 기수 프로젝트 '신용카드 사기 거래 탐지 AI' 팀 Repository입니다.

Ensemble을 제외한 OneClassSVM 단일 알고리즘 적용 시도에 관한 내용입니다.

OneClassSVM은 비지도 학습 알고리즘으로 원점과 초평면 사이의 거리에 기반하여 이상치 데이터를 탐지해냅니다.

sklearn.svm.OneClassSVM(*, kernel='rbf', degree=3, gamma='scale', coef0=0.0, tol=0.001, nu=0.5, shrinking=True, cache_size=200, verbose=False, max_iter=- 1)

<주요 매개변수>

모델 성능 향상에 다소 영향력이 큰 매개변수 3개 정도만 조절하시면 문제가 없을 것입니다.

1. kernel은 알고리즘에 사용할 커널 유형을 지정합니다. 디폴트 값은 'rbf'로, 비선형 커널 중 일반적으로 성능이 가장 높게 나타납니다.
2. gamma는 'rbf', 'poly' 및 'sigmoid'에 대한 커널 계수입니다.
2. nu는 훈련 데이터의 오류 비율입니다. 모델을 진행할 때 validation set의 이상치 비율을 넣어 적용했습니다.

- kernel = {‘linear’, ‘poly’, ‘rbf’, ‘sigmoid’, ‘precomputed’}, default = ’rbf’
- gamma = {‘scale’, ‘auto’} or float, default = ’scale’

→ gamma = ‘scale’인 경우 감마 값으로 1 / (n_features * X.var())을 사용하고, 'auto'인 경우 1 / n_features를 사용

- nu = float, default = 0.5

모델링 순서:

1. 기본 디폴트 값으로 One class SVM 사용
2. StandardScaler로 스케일링 후 One class SVM 사용
3. MinMaxScaler로 스케일링 후 One class SVM 사용
4. tSNE로 차원 축소 후 One class SVM 사용
5. normalized tSNE로 차원 축소 후 One class SVM 사용
6. 상관계수가 큰 특성 5개를 추출하여 One class SVM 사용
7. PCA로 차원 축소 후 One class SVM 사용 (주성분 3개 경우가 가장 높은 점수)
8. One class SVM으로 첫 번째 예측값을 얻은 이후, one class SVM으로 train set에 임의의 label을 주어 LGBM으로 모델을 최적화하여 두 번째 예측값을 얻고 Ensemble
