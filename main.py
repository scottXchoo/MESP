import numpy as np
import k_shape_predict

ORIGINAL_DATA_PATH = './dataset/preprocessed_training_dataset.npy'
MODEL_PATH = './output/kshape_model_k6.pkl'

X_train = np.load(ORIGINAL_DATA_PATH)

cluster_labels = k_shape_predict.k_shape_predict(X_train)

print("Cluster labels for the training dataset:")
print(cluster_labels)  # [1 0 3 ... 2 5 4] (출력값 예시)