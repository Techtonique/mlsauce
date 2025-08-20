import mlsauce as ms
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_iris

# Load dataset
data = load_iris()
X = data.data
y = data.target

# Initialize model
estimator = LogisticRegression(max_iter=200)

# Define hyperparameters to tune
param_dict = {'C': 1.0, 'solver': 'liblinear'}

# Compute penalized cross-validation score
penalized_score = ms.penalized_cross_val_score(
    estimator, X, y, param_dict, cv=5, penalty_strength=0.5, penalty_type='std'
)

print(f"Penalized CV score: {penalized_score}")
