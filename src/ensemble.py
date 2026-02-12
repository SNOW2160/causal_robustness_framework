import numpy as np

class PSSWeightedEnsemble:
    def __init__(self, models, pss_scores, epsilon=0.1):
        self.models = models
        self.pss_scores = np.array(pss_scores)
        self.epsilon = epsilon
        # Weight = 1 / (Hallucination + epsilon)
        inv_scores = 1.0 / (self.pss_scores + self.epsilon)
        self.weights = inv_scores / np.sum(inv_scores)

    def predict_cate(self, X):
        predictions = np.array([model.predict_cate(X) for model in self.models])
        return np.average(predictions, axis=0, weights=self.weights)