import matplotlib.pyplot as plt 
import mlsauce as ms 
import numpy as np 
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from time import time 

X, y = load_diabetes(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=123
)

scaler = StandardScaler().fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

alpha = 0.10

print("=" * 70)
print(f"{'Model':25s} {'Coverage':>10s} {'Width':>10s} {'RMSE':>10s}")
print("-" * 70)

for name, model in [
    ("Plain Ridge", ms.RVFLJackknifePlus(n_hidden=0, lambda_=50.0)),
    ("RVFL (asymmetric)", ms.RVFLJackknifePlus(n_hidden=200, lambda_=50.0, symmetric=False)),
    ("RVFL (symmetric)", ms.RVFLJackknifePlus(n_hidden=200, lambda_=50.0, symmetric=True)),
]:
    start = time()
    model.fit(X_train, y_train)
    pred = model.predict(X_test, alpha=alpha, return_pi=True)
    print(f"Elapsed: {time() - start}")
    
    cov = np.mean((y_test >= pred.lower) & (y_test <= pred.upper))
    width = np.mean(pred.upper - pred.lower)
    rmse = np.sqrt(mean_squared_error(y_test, pred.mean))
    
    print(f"{name:25s} {cov:9.3f}  {width:9.2f}  {rmse:9.2f}")

# ---- Plot: RVFL out-of-sample jackknife+ band ----
# Use the asymmetric RVFL model for visualization
model_best = ms.RVFLJackknifePlus(n_hidden=200, lambda_=50.0, symmetric=False)
model_best.fit(X_train, y_train)
pred_rvfl = model_best.predict(X_test, alpha=alpha, return_pi=True)

order = np.argsort(pred_rvfl.mean)
x_axis = np.arange(len(order))

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(x_axis, pred_rvfl.mean[order], color="darkorange", lw=2, label="RVFL prediction")
ax.fill_between(
    x_axis, pred_rvfl.lower[order], pred_rvfl.upper[order],
    color="darkorange", alpha=0.20, label="Jackknife+ interval",
)
ax.scatter(
    x_axis, y_test[order], color="black", alpha=0.55, s=25,
    label="Held-out observations",
)
ax.set_xlabel("Test points (ordered by predicted value)")
ax.set_ylabel("Diabetes progression score")
ax.set_title("RVFL + ridge read-out: out-of-sample jackknife+ intervals")
ax.legend(loc="upper left", frameon=False)
fig.tight_layout()
plt.savefig("rvfl_jackknife_plus.png", dpi=150)
plt.show()
print("\nSaved plot to rvfl_jackknife_plus.png")