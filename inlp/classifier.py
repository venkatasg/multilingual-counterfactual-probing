import numpy as np
import ipdb

# an abstract class for linear classifiers

class Classifier(object):

    def __init__(self):

        pass

    def train(self, X_train: np.ndarray, Y_train: np.ndarray, X_dev: np.ndarray, Y_dev: np.ndarray) -> float:
        """

        :param X_train:
        :param Y_train:
        :param X_dev:
        :param Y_dev:
        :return: accuracy score on the dev set
        """
        raise NotImplementedError

    def get_weights(self) -> np.ndarray:
        """
        :return: final weights of the model, as np array
        """

        raise NotImplementedError




class SKlearnClassifier(Classifier):

    def __init__(self, m):

        self.model = m

    def train_network(self, X_train: np.ndarray, Y_train: np.ndarray, X_dev: np.ndarray, Y_dev: np.ndarray) -> float:

        """
        :param X_train:
        :param Y_train:
        :param X_dev:
        :param Y_dev:
        :return: accuracy score on the dev set / Person's R in the case of regression
        """
        
        self.model.fit(X_train, Y_train.ravel())#, coef_init = np.zeros(768))
        score = self.model.score(X_dev, Y_dev.ravel())
        return score
    
    def evaluate(self, X_dev: np.ndarray, Y_dev: np.ndarray) -> float:
        
        score = self.model.score(X_dev, Y_dev.ravel())
        
        return score

    def get_weights(self) -> np.ndarray:
        """
        :return: final weights of the model, as np array
        """

        w = self.model.coef_
        if len(w.shape) == 1:
                w = np.expand_dims(w, 0)

        return w
    
    def set_weights(self, w):
        """
        :return: set random attributes for model
        """

        self.model.coef_ = w
        self.model.intercept_ = 0
        self.model.classes_ = np.array([0,1])
