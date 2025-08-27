import mlsauce as ms 

gen = ms.make_diverse_classification(n_datasets=15, random_state=42)

for i, (X, y, meta) in enumerate(gen):
    print(f"Dataset {i+1}:")
    print(f"  Shape: {X.shape}, Classes: {len(np.unique(y))}")
    print(f"  Class distribution: {np.bincount(y)}")
    print(f"  Params: n_inf={meta['n_informative']}, "
          f"n_red={meta['n_redundant']}, "
          f"n_rep={meta['n_repeated']}, "
          f"flip_y={meta['flip_y']:.2f}")
    print("---")