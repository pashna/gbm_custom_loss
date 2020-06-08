import numpy as np
from gbm_custom_loss.catboost.piecewise_loss.piecewise_mse import PiecewiseMSE


class BiasedPiecewiseRMSE(PiecewiseMSE):
    """
    Class defines loss function according to CatBoost API
    https://catboost.ai/docs/concepts/python-usages-examples.html#user-defined-loss-function

    Interpretation of the loss: if the error for the object falls within predefined interval,
    we take it squared error and multiply by a corresponding the this error coefficient and shift
    with a corresponding bias.

    The loss function is defined by mse_pieces:
    (lower_bound, upper_bound):  {"coef": mse_coefficient,
                                  "bias": mse_bias}

    So, the value of loss function is defined by:
        loss_i = coef_k*(y_true_i - y_pred_i - bias_k)^2

        where coef_k: coef_k(y_true_i - y_pred_i),
              bias_k: bias_k(y_true_i - y_pred_i)
                - step functions of residual
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.name = "BiasedPiecewiseMSE"

    def calc_ders_range(self, approxes, targets, weights):
        """
        Returns a list of tuples with first and second derivitave
        """
        self._validate_input(approxes, targets, weights)
        result = []
        for index in range(len(targets)):
            diff = targets[index] - approxes[index]
            coef, bias = self._find_params(diff, ["coef", "bias"])

            der1, der2 = (diff - bias) * coef, -coef

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
            coef, bias = self._find_params(diff, ["coef", "bias"])
            loss += weight[i] * coef * ((target[i] - approx[i] - bias) ** 2)

        return loss / np.sum(weight), np.sum(weight)
