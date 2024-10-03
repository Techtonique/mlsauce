import mlsauce as ms 
import numpy as np
#from shap import KernelExplainer
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from time import time
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor

X, y = load_diabetes(as_frame=False, return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,
                                                    random_state=42)

lin_model = LinearRegression().fit(X_train, y_train);
#rfr_model = RandomForestRegressor().fit(X_train, y_train);
#gbt_model = GradientBoostingRegressor().fit(X_train, y_train);


from_scratch_explainer = ms.ShapExplainer(lin_model.predict, X_train[:100, :])
start = time()
from_scratch_shap_values = from_scratch_explainer.shap_values(X_train[:100, :])
end = time()
print(f"Time taken: {end - start}")
print(f"From scratch shap values: {from_scratch_shap_values}")

# def compare_methods(model, X_instances, X_background):

#     #library_explainer = KernelExplainer(model.predict, X_background)
#     #library_shap_values = library_explainer.shap_values(X_instances)
#     #print(f"Library shap values: {library_shap_values}")
#     from_scratch_explainer = ms.ShapExplainer(model.predict, X_background)
#     start = time()
#     from_scratch_shap_values = from_scratch_explainer.shap_values(X_instances)
#     end = time()
#     print(f"Time taken: {end - start}")
#     print(f"From scratch shap values: {from_scratch_shap_values}")
#     #return np.allclose(library_shap_values, from_scratch_shap_values)

# print(compare_methods(lin_model,
#                 X_background=X_train[:100, :],
#                 X_instances=X_test[:5, :]))

