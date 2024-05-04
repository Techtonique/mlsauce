import numpy as np
from numpy.linalg import lstsq, norm
from scipy.special import expit
from sklearn.cluster import MiniBatchKMeans


def one_hot_encode(y, n_classes):
    n_obs = len(y)
    res = np.zeros((n_obs, n_classes))
    for i in range(n_obs):
        res[i, y[i]] = 1
    return res


def update_proba(
    probs, misclass, w_new, w_prev, v_prev, eta, gamma, n_obs, n_classes
):
    new_probs = probs.copy()
    v_new = 0
    for idx in range(n_obs):
        if misclass[idx] != 1000:
            diff_proba = max(w_new[idx], 2.220446049250313e-16) - w_prev[idx]
            v_new = diff_proba
            new_probs[idx, misclass[idx]] -= (v_prev * gamma - v_new) / eta
    return new_probs, v_new


def fit_adaopt(
    X,
    y,
    n_iterations,
    n_classes,
    learning_rate,
    reg_lambda,
    reg_alpha,
    eta,
    gamma,
    tolerance,
):
    scaled_X = X / norm(X, ord=2, axis=1)[:, None]
    Y = one_hot_encode(y, n_classes)
    beta = lstsq(scaled_X, Y, rcond=None)[0]
    probs = expit(np.dot(scaled_X, beta))
    probs /= np.sum(probs, axis=1)[:, None]
    preds = np.argmax(probs, axis=1)
    w_prev = np.repeat(1.0 / len(X), len(X))
    w_new = np.repeat(0.0, len(X))
    misclass = np.repeat(1000, len(X))
    v_prev = 0
    alphas = np.zeros(n_iterations)
    err_bound = 1.0 - 1.0 / n_classes

    for m in range(n_iterations):
        for i in range(len(X)):
            if y[i] != preds[i]:
                misclass[i] = preds[i]
            else:
                misclass[i] = 1000

        err_m = np.sum(misclass != 1000) / len(X)
        err_m += reg_lambda * (
            reg_alpha * np.sum(np.abs(w_prev))
            + (1 - reg_alpha) * 0.5 * np.sum(np.power(w_prev, 2))
        )
        err_m = min(max(err_m, 2.220446049250313e-16), err_bound)

        alpha_m = learning_rate * np.log((n_classes - 1) * (1 / err_m - 1))
        alphas[m] = alpha_m
        w_prev *= np.exp(alpha_m * (misclass != 1000))
        w_prev /= np.sum(w_prev)
        w_new = w_prev

        probs, v_prev = update_proba(
            probs,
            misclass,
            w_new,
            w_prev,
            v_prev,
            eta,
            gamma,
            len(X),
            n_classes,
        )
        preds = np.argmax(probs, axis=1)

        if np.abs(np.max(alphas)) - np.abs(np.min(alphas)) <= tolerance:
            n_iterations = m
            break

    return {"probs": probs, "alphas": alphas, "n_iterations": n_iterations}


def predict_proba_adaopt(
    X_test,
    scaled_X_train,
    probs_train,
    k,
    n_clusters,
    seed,
    batch_size=100,
    type_dist="euclidean",
    cache=True,
):
    n_test, n_train = len(X_test), len(scaled_X_train)
    n_classes = probs_train.shape[1]
    out_probs = np.zeros((n_test, n_classes), dtype=np.double)
    probs_train_ = probs_train

    if n_clusters > 0:
        probs_train_ = np.zeros((n_clusters, n_classes), dtype=np.double)
        kmeans = MiniBatchKMeans(
            n_clusters=n_clusters, batch_size=batch_size, random_state=seed
        ).fit(scaled_X_train)
        scaled_X_train = kmeans.cluster_centers_

        for m in range(n_clusters):
            index_train = np.where(kmeans.labels_ == m)[0]
            avg_probs = np.average(probs_train[index_train], axis=0)
            probs_train_[m] = avg_probs

    dist_mat = (
        np.sum(X_test**2, axis=1)[:, None]
        + np.sum(scaled_X_train**2, axis=1)
        - 2 * np.dot(X_test, scaled_X_train.T)
    )

    for i in range(n_test):
        dists_test_i = dist_mat[i]
        kmin_test_i = np.argsort(dists_test_i)[:k]
        weights_test_i = 1 / np.maximum(
            dists_test_i[kmin_test_i], np.finfo(float).eps
        )
        weights_test_i /= np.sum(weights_test_i)
        probs_test_i = probs_train_[kmin_test_i]
        avg_probs_i = np.sum(probs_test_i * weights_test_i[:, None], axis=0)
        out_probs[i] = avg_probs_i

    out_probs_ = expit(out_probs)
    out_probs_ /= np.sum(out_probs_, axis=1)[:, None]
    return out_probs_


def predict_adaopt(
    X_test,
    scaled_X_train,
    probs_train,
    k,
    n_clusters,
    seed,
    batch_size=100,
    type_dist="euclidean",
    cache=True,
):
    return np.argmax(
        predict_proba_adaopt(
            X_test,
            scaled_X_train,
            probs_train,
            k,
            n_clusters,
            seed,
            batch_size,
            type_dist,
            cache,
        ),
        axis=1,
    )
