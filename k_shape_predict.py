import pickle

MODEL_PATH = './output/kshape_model_k6.pkl'

def k_shape_predict(dataset):
    # 모델 로드
    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)

    # 클러스터 예측
    cluster_label = model.predict(dataset)  # 0~5 (클러스터 번호)

    return cluster_label