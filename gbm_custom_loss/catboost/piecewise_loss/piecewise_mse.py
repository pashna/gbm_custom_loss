import numpy as np
from gbm_custom_loss.catboost.piecewise_loss.piecewise_objective import PiecewiseObjective


class PiecewiseMSE(PiecewiseObjective):
    """
    Class defines loss function according to CatBoost API
    https://catboost.ai/docs/concepts/python-usages-examples.html#user-defined-loss-function

    Interpretation of the loss: if the error for the object falls within predefined interval,
    we take it squared error and multiply by a corresponding to this error coefficient.

    The loss function is defined by mse_pieces:
    (lower_bound, upper_bound):  {"coef": mse_coefficient}

    So, the value of loss function is defined by:
        loss_i = coef_k*(y_true_i - y_pred_i)^2

        where coef_k: coef_k(y_true_i - y_pred_i) - step functions of residual,
                                                    defined in the configuration

    """

    def __init__(self, intervals_values, **kwargs):
        """

        :param intervals_values: dict, where key is a tuple that represents a left-open,
                                       right-closed interval.
                                       The value of the dict is a another dictionary where
                                       each item each has a parameter name as a key and the value is a number.
                                       For Piecewise MSE, the nested key is required to be "coef".
                                       Example:
                                       {
                                         (-1e20, 5): {"coef": 10},
                                         (5, 1e20): {"coef": 100}
                                       }
        :param kwargs:
        """
        super().__init__(intervals_values, **kwargs)
        self.name = "PiecewiseMSE"

    def _validate_input(self, approxes, targets, weights):
        assert len(approxes) == len(targets)
        if weights is not None:
            assert len(weights) == len(approxes)

    def calc_ders_range(self, approxes, targets, weights):
        """
        Returns a list of tuples with first and second derivative
        """
        self._validate_input(approxes, targets, weights)
        result = []
        for index in range(len(targets)):
            diff = targets[index] - approxes[index]
            coef = self._find_params(diff, ["coef"])[0]

            der1, der2 = diff * coef, -coef

            if weights is not None:
                der1 *= weights[index]
                der2 *= weights[index]
            result.append((der1, der2))
        return result

    def evaluate(self, approx, target, weight=None):

        loss = 0.0
        approx = approx[0]
        weight = np.ones_like(target) if weight is None else weight

        for i in range(len(approx)):
            diff = (target[i] - approx[i])
            coef = self._find_params(diff, ["coef"])[0]
            loss += weight[i] * coef * (diff ** 2)

        return loss / np.sum(weight), np.sum(weight)
