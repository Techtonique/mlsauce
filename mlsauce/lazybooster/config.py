import nnetsauce as ns 
from functools import partial
from sklearn.base import RegressorMixin
from sklearn.utils import all_estimators

removed_regressors = [
    "AdaBoostRegressor",
    "BaggingRegressor",
    "TheilSenRegressor",
    "ARDRegression",
    "ExtraTreesRegressor",
    "CCA",
    "GaussianProcessRegressor",
    "GradientBoostingRegressor",
    "HistGradientBoostingRegressor",
    "IsotonicRegression",
    "MLPRegressor",
    #"KernelRidge",
    "MultiOutputRegressor",
    "MultiTaskElasticNet",
    "MultiTaskElasticNetCV",
    "MultiTaskLasso",
    "MultiTaskLassoCV",
    #"NuSVR",
    "OrthogonalMatchingPursuit",
    "OrthogonalMatchingPursuitCV",
    "PLSCanonical",
    "PLSRegression",
    "RadiusNeighborsRegressor",
    "RandomForestRegressor",
    "RegressorChain",
    "StackingRegressor",
    #"SVR",
    "VotingRegressor",
]


REGRESSORS = [
    ("GenericBooster(" + est[0] + ")", est[1])
    for est in all_estimators()
    if (
        issubclass(est[1], RegressorMixin)
        and (est[0] not in removed_regressors)
    )
]
