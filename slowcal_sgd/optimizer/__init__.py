from torch.optim import SGD
from .anytime_sgd import AnyTimeSGD
from .mu2_sgd import Mu2SGD

OPTIMIZER_REGISTRY = {
    'sgd': SGD,
    'anytime_sgd': AnyTimeSGD,
    'mu2_sgd': Mu2SGD
}
