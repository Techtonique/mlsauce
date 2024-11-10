import nnetsauce as ns
from functools import partial
from sklearn.base import RegressorMixin
from ..multitaskregressor.mtaskregr import MultiTaskRegressor
from sklearn.utils import all_estimators
from ..utils import is_multitask_estimator

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
    "KernelRidge",
    # "MultiOutputRegressor",
    # "MultiTaskElasticNet",
    "MultiTaskElasticNetCV",
    # "MultiTaskLasso",
    "MultiTaskLassoCV",
    "NuSVR",
    "OrthogonalMatchingPursuit",
    "OrthogonalMatchingPursuitCV",
    "PLSCanonical",
    "PLSRegression",
    "RadiusNeighborsRegressor",
    "RandomForestRegressor",
    "RANSACRegressor",
    "RegressorChain",
    "StackingRegressor",
    # "SVR",
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


MTSREGRESSORS = [
    ("MTS(GenericBooster(" + est[0] + "))", est[1])
    for est in all_estimators()
    if (
        issubclass(est[1], RegressorMixin)
        and (est[0] not in removed_regressors)
    )
]


MTASKREGRESSORS = [
    (
        "GenericBooster(MultiTask(" + est[0] + "))",
        partial(MultiTaskRegressor, regr=est[1]()),
    )
    for est in all_estimators()
    if (
        issubclass(est[1], RegressorMixin)
        and (est[0] not in removed_regressors)
        and (est[0][:5] != "Multi")
        and (est[0][-2:] != "CV")
        and (est[0] not in ("SVR", "HuberRegressor", "LassoLarsIC"))
        and is_multitask_estimator(est[1]()) == False
    )
]
