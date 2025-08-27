import numpy as np
from sklearn.datasets import make_classification
from scipy.stats import loguniform, dirichlet

def make_diverse_classification(n_datasets=100, random_state=None):
    rng = np.random.default_rng(random_state)

    for _ in range(n_datasets):
        # Sample parameters
        n_samples = int(loguniform(100, 10000).rvs(random_state=rng))
        n_features = int(rng.uniform(10, 50))

        # --- Step 1: Choose n_classes safely ---
        max_classes_by_sample = max(2, n_samples // 10)
        n_classes = rng.integers(2, min(100, max_classes_by_sample) + 1)
        n_classes = max(2, min(n_classes, n_samples // 2))

        # --- Step 2: Class weights with minimum samples ---
        alpha = [0.5] * n_classes
        weights = dirichlet.rvs(alpha, random_state=rng.integers(0, 2**32))[0]
        weights /= weights.sum()

        min_per_class = 2
        total_min = n_classes * min_per_class
        if total_min > n_samples:
            weights = np.ones(n_classes) / n_classes
        else:
            # Distribute at least min_per_class, then scale
            y_counts = np.maximum(np.round(weights * n_samples), min_per_class).astype(int)
            y_counts = (y_counts / y_counts.sum() * n_samples).astype(int)
            y_counts[-1] += n_samples - y_counts.sum()  # fix rounding
            y_counts = np.maximum(y_counts, min_per_class)
            y_counts[-1] += n_samples - y_counts.sum()
            weights = (y_counts / n_samples).tolist()

        # --- Step 3: Informative features ---
        # Must support n_classes * n_clusters_per_class <= 2 ** n_informative
        # So let's first pick n_informative large enough or cap n_classes
        n_informative = max(4, int(rng.uniform(4, min(10, n_features))))  # start higher
        n_informative = min(n_informative, n_features - 6, n_samples - 1)

        # Cap n_classes based on n_informative
        max_possible_classes = 2 ** n_informative
        if n_classes > max_possible_classes:
            n_classes = max_possible_classes
            # Recompute weights
            alpha = [0.5] * n_classes
            weights = dirichlet.rvs(alpha, random_state=rng.integers(0, 2**32))[0]
            weights = (weights / weights.sum()).tolist()

        # --- Step 4: Redundant, repeated, noise ---
        n_redundant = min(n_informative, int(rng.uniform(0, 0.5) * (n_features - n_informative)))
        available = n_features - n_informative - n_redundant
        n_repeated = int(rng.uniform(0, 0.2) * available) if available > 0 else 0
        n_noise = n_features - n_informative - n_redundant - n_repeated

        if n_noise < 0:
            continue  # should not happen

        # --- Step 5: Clusters per class ---
        max_clusters_total = 2 ** n_informative
        n_clusters_per_class = rng.integers(1, 4)
        n_clusters_per_class = min(n_clusters_per_class, max_clusters_total // n_classes)
        n_clusters_per_class = max(1, n_clusters_per_class)

        # --- Step 6: Other parameters ---
        class_sep = loguniform(0.1, 10).rvs(random_state=rng)
        flip_y = rng.uniform(0.0, 0.5)
        hypercube = rng.choice([True, False])
        shift = rng.uniform(-1, 1, n_features) if rng.random() < 0.5 else 0.0
        scale = rng.uniform(0.5, 5.0)

        # --- Final safety ---
        if n_informative + n_redundant + n_repeated > n_features:
            continue

        try:
            X, y = make_classification(
                n_samples=n_samples,
                n_features=n_features,
                n_informative=n_informative,
                n_redundant=n_redundant,
                n_repeated=n_repeated,
                n_classes=n_classes,
                n_clusters_per_class=n_clusters_per_class,
                weights=weights,
                flip_y=flip_y,
                class_sep=class_sep,
                hypercube=hypercube,
                shift=shift,
                scale=scale,
                shuffle=True,
                random_state=rng.integers(0, 2**32)
            )
        except Exception as e:
            print(f"Skipped due to error: {e}")
            continue

        metadata = {
            'n_samples': n_samples,
            'n_features': n_features,
            'n_classes': n_classes,
            'n_informative': n_informative,
            'n_redundant': n_redundant,
            'n_repeated': n_repeated,
            'n_noise': n_noise,
            'weights': weights,
            'flip_y': flip_y,
            'class_sep': class_sep,
            'n_clusters_per_class': n_clusters_per_class,
            'hypercube': hypercube,
            'scale': scale
        }

        yield X, y, metadata
