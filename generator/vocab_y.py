from typing import Any, Text, List, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, PowerTransformer, QuantileTransformer, FunctionTransformer


def identity_transformer(x):
    return x


class PropVocab:
    @classmethod
    def from_data(cls, df: pd.DataFrame,
                  labels: List[Text],
                  weights: Optional[np.ndarray] = None,
                  scaler_type: Optional[Text] = None):
        if scaler_type == 'standard':
            scaler = StandardScaler()
        elif scaler_type == 'power':
            scaler = PowerTransformer(method='yeo-johnson')
        elif scaler_type == 'quantile':
            scaler = QuantileTransformer()
        elif scaler_type == 'ln':
            scaler = StandardScaler()
        elif scaler_type == 'null':
            scaler = FunctionTransformer(identity_transformer, validate=False)
        else:
            raise ValueError(f'{scaler_type} not implemented!')
        scaler.fit(df[labels].values)
        return cls(scaler, scaler_type, labels, weights)

    def __init__(self, scaler: Any, scaler_type, labels: List[Text], weights: Optional[np.ndarray] = None):
        self.scaler_type = scaler_type
        self.scaler = scaler
        self.labels = labels
        if weights is None:
            # 默认生成权重全为1的列表
            weights = np.ones(len(labels)).astype(np.float32)
        self.weights = weights

    def transform(self, x: np.ndarray) -> np.ndarray:
        if self.scaler_type == 'power':
            epsilon = np.finfo(np.float32).eps
            x = x + epsilon
        if self.scaler_type == 'ln':
            epsilon = np.finfo(np.float32).eps
            x = x + epsilon
            x = np.log(x)
        return self.scaler.transform(x)

    def inverse_transform(self, x: np.ndarray) -> np.ndarray:
        if self.scaler_type == 'ln':
            return np.exp(self.scaler.inverse_transform(x))
        return self.scaler.inverse_transform(x)

    def df_to_y(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        return np.array(self.transform(df[self.labels].values)), np.ones(len(df), dtype=np.float32)
