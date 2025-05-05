from .logistic_regression import LogisticRegressionModel
from .simple_conv import SimpleConv


MODEL_REGISTRY = {
    'logistic_regression': LogisticRegressionModel,
    'simple_conv': SimpleConv
}

