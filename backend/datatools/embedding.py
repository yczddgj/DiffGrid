import numpy as np
import joblib
import os


def load_embedding_model(model_path: str):
    """
    加载保存的降维模型（如 UMAP、TSNE），支持 .joblib 格式
    """
    return joblib.load(model_path)


def reduce_features(embedding_model, features: np.ndarray):
    """
    使用降维模型对特征矩阵进行降维
    """
    return embedding_model.transform(features)


def save_embedding(embedding: np.ndarray, save_path: str):
    """
    保存降维结果
    """
    np.save(save_path, embedding)
