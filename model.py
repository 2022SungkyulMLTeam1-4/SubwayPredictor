import numpy as np
import sklearn.model_selection


class Model:
    """
    학습용 모델입니다.
    """

    def __init__(self):
        self.x_train: np.ndarray = None
        self.y_train: np.ndarray = None
        self.x_test: np.ndarray = None
        self.y_test: np.ndarray = None
        self.model = None

    def load_data(self, x: np.ndarray, y: np.ndarray):
        self.x_train, self.x_test, self.y_train, self.y_test = sklearn.model_selection.train_test_split(x, y,
                                                                                                        test_size=0.2,
                                                                                                        random_state=42,
                                                                                                        shuffle=True)
