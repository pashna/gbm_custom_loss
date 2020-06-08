import numpy as np


def calculate_mse(approx, target, weight=None):
    loss = 0.0
    weight = np.ones_like(target) if weight is None else weight
    for i in range(len(approx)):
        error = (target[i] - approx[i])
        loss += weight[i] * error * error

    return loss / len(approx), weight


def calculate_quantile(approx, target, alpha=0.5, weight=None):
    loss = 0.0

    weight = np.ones_like(target) if weight is None else weight
    for i in range(len(approx)):
        mult = 1 if target[i] <= approx[i] else 0
        loss += (alpha - mult) * (target[i] - approx[i]) * weight[i]

    return loss / len(approx), weight


def plot_loss(objective, values_range=(-40, 40)):
    """
    :param objective: str or object.
                      if str, should be 'mse', 'quantile:alpha=q', or 'mae'
                      if object, should have a method evaluate(approx, target, weight),
                      according to documentation of Catboost
                      https://catboost.ai/docs/concepts/python-usages-examples.html#user-defined-loss-function

    :param values_range: lin-space of evaluation
    :return:
    """
    if type(objective) == str:
        if objective.lower() == 'mse':
            calc_obj = calculate_mse
        elif 'quantile' in objective.lower():
            alpha = float(objective.split('=')[1])
            calc_obj = lambda approx, target: calculate_quantile(approx, target,
                                                                 alpha=alpha, weight=None)
        elif 'mae' in objective.lower():
            calc_obj = lambda approx, target: calculate_quantile(approx, target,
                                                                 alpha=0.5, weight=None)
        else:
            raise Exception("Visualization of {} is not supported".format(objective))
    else:
        calc_obj = objective.evaluate
    space = np.linspace(values_range[0], values_range[1], 5 * abs(int(values_range[1] - values_range[0])))

    objective_values = []
    for s in space:
        objective_values.append(calc_obj([[0]], [s])[0])

    return space, objective_values
