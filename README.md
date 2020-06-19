# GBM Custom Loss
There are many application where usage of default RMSE, MAE, Huber, Quintile often are not enough. Practically, it usually happens when objective of a project is non-symmetric or non-convex. 

For example, when you predict expected time of arrival for food delivery plaform, it is way more harmful to underpredict (set expectation that the food gonna arrive too soon) than overpredict. The same happens you predict when a plane will take off and based on this information passangers will take order taxi to an airport. In case you were too  oprimistic, the passangers might miss the flight, and the cost of overprediction is way higher than for underprediction. 

This repo has implementations of a few handy custom loss funcitons for GBM (at the moment, Catboost only), to tackle this problem

You could learn more of how we use this loss in production in this [medium post](https://towardsdatascience.com/byol-bring-your-own-loss-c5292cb8e9e3)


## Installation
```
pip install git+https://github.com/pashna/gbm_custom_loss.git
```

## Example of usage

```
from gbm_custom_loss.catboost.piecewise_loss.piecewise_mse import PiecewiseMSE
piecewise_mse = PiecewiseMSE({ 
                             (-1e20, -10): {"coef": 13}, 
                             (-10, 5): {"coef": 0.1},
                             (5, 1e20): {"coef": 25}
                            })
                            
cbr = CatBoostRegressor(loss_function=piecewise_mse,
                        eval_metric=piecewise_mse,
                        iterations=200,
                        silent=True,
                        use_best_model=True)
```

Visualization of a loss function:
```
from gbm_custom_loss.utils.visualization import plot_loss
x, y = plot_loss(piecewise_mse)
_, ax = plt.subplots(figsize=(17, 7))
ax.set_title(str(piecewise_mse))
ax.plot(x, y)
ax.set_xlabel("y_true - y_pred | y_true=0")
ax.set_ylabel("loss")
ax.grid()
```

## More examples
Please, check out the example directory to explore the usage of the [library](https://github.com/pashna/gbm_custom_loss/tree/master/example)

