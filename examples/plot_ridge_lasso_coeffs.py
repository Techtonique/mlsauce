import subprocess
import sys
import os 

print(f"\n ----- Running: {os.path.basename(__file__)}... ----- \n")

subprocess.check_call([sys.executable, "-m", "pip", "install", "matplotlib"])

import matplotlib.pyplot as plt
import numpy as np
import mlsauce as ms
from numpy.linalg import norm
from sklearn.datasets import make_regression
from sklearn.metrics import mean_squared_error

clf1 = ms.RidgeRegressor()
clf2 = ms.LassoRegressor()

X, y, w = make_regression(n_samples=100, n_features=10, coef=True,
                          random_state=1, bias=3.5)

coefs1 = []
errors1 = []
coefs2 = []
errors2 = []

n_alphas = 200
alphas1 = 10**np.linspace(-6, 6, n_alphas)
alphas2 = 10**np.linspace(-6, 6, n_alphas)

# Ridge

# Train the model with different regularisation strengths
for i in range(n_alphas):

    clf1.set_params(reg_lambda=alphas1[i])
    clf1.fit(X, y)
    coefs1.append(clf1.beta)    
    errors1.append(norm(clf1.beta, ord=2)/norm(w, ord=2))

    clf2.set_params(reg_lambda=alphas2[i])
    clf2.fit(X, y)
    coefs2.append(clf2.beta)
    errors2.append(norm(clf2.beta, ord=1)/norm(w, ord=1))

# Display results
plt.figure(figsize=(20, 6))

plt.subplot(221)
ax = plt.gca()
ax.plot(alphas1, coefs1)
ax.set_xscale('log')
plt.xlabel('')
plt.ylabel('coefficients')
plt.title('Ridge coefficients = f(reg_lambda)')
plt.axis('tight')

plt.subplot(222)
ax = plt.gca()
ax.plot(errors1, coefs1)
ax.set_xscale('log')
plt.xlabel('')
plt.ylabel('coefficients')
plt.title('Ridge coefficients = f(norm_2(ridge)/norm_2(ls))')
plt.axis('tight')

plt.subplot(223)
ax = plt.gca()
ax.plot(alphas2, coefs2)
ax.set_xscale('log')
plt.xlabel('reg_lambda')
plt.ylabel('coefficients')
plt.title('Lasso coefficients = f(reg_lambda)')
plt.axis('tight')

plt.subplot(224)
ax = plt.gca()
ax.plot(errors2, coefs2)
ax.set_xscale('log')
plt.xlabel('coeffs norm ratio')
plt.ylabel('coefficients')
plt.title('Lasso coefficients = f(norm_1(lasso)/norm_1(ls))')
plt.axis('tight')

plt.show()
