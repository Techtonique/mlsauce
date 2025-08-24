import pandas as pd
import numpy as np
import mlsauce as ms


# Training data
train_data = pd.DataFrame({
    'city': ['NYC', 'LA', 'NYC', 'Chicago', 'LA', 'Chicago', 'NYC', 'LA'],
    'age': [5, 10, 3, 7, 12, 6, 4, 9],
    'rooms': [3, 2, 3, 1, 2, 1, 3, 2]
})
y_train = np.array([800, 400, 850, 300, 420, 310, 820, 390])  # house prices in thousands

# Test data (includes a new city: 'Boston')
test_data = pd.DataFrame({
    'city': ['Chicago', 'NYC', 'Boston'],
    'age': [8, 2, 5],
    'rooms': [2, 3, 1]
})

# Initialize encoder
encoder = ms.RankTargetEncoder(
    correlation_type='spearman',
    correlation_strength=0.5,
    shrinkage=5,
    n_folds=3,
    ensemble_size=3,
    random_state=42
)

# Step 1: Fit + Transform on training data
print("Step 1: fit_transform on training data")
print("train_data", train_data)
train_encoded = encoder.fit_transform(train_data, y_train)
print("train_encoded", train_encoded)

# Step 2: Transform on new (test) data
print("\nStep 2: transform on test data (unseen examples)")
print("test_data", test_data)
test_encoded = encoder.transform(test_data)
print("test_encoded", test_encoded)

# Bonus: see how categories were encoded
print("\nLearned encodings for 'city':")
print(encoder.get_category_mappings()['city'])

# Show which columns were identified as categorical
print(f"\nCategorical columns: {encoder.cat_columns_}")
print(f"Non-categorical columns: {encoder.non_cat_columns_}")

#print(encoder.validate_encoding(train_data, y_train, plot=False))

print(encoder.validate_encoding(train_data, y_train, plot=True))
