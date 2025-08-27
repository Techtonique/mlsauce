import mlsauce as ms 

# Initialize generator
generator = ms.HealthcareTimeSeriesGenerator(seed=42)

# Generate datasets
print("Generating patient demographics...")
patients = generator.generate_patient_demographics(n_patients=50)

print("Generating time series data...")
timeseries = generator.generate_time_series(patients, measurements_per_day=4)

print("Generating outcomes...")
outcomes = generator.generate_outcomes(patients, timeseries)

# Create visualizations
print("Creating visualizations...")
create_visualizations(patients, timeseries, outcomes)

# Display summary statistics
print("\n=== DATASET SUMMARY ===")
print(f"Total patients: {len(patients)}")
print(f"Total measurements: {len(timeseries)}")
print(f"Date range: {timeseries['timestamp'].min()} to {timeseries['timestamp'].max()}")

print("\nAge distribution:")
print(patients['age'].describe())

print("\nCondition frequency:")
all_conditions = [cond for conditions in patients['conditions'] for cond in conditions]
condition_counts = pd.Series(all_conditions).value_counts()
print(condition_counts)

print("\nVital signs summary:")
vital_cols = ['heart_rate', 'systolic_bp', 'diastolic_bp', 'temperature', 'respiratory_rate', 'oxygen_saturation']
print(timeseries[vital_cols].describe())

print("\nMissing data percentage:")
missing_pct = (timeseries.isnull().sum() / len(timeseries) * 100).round(2)
print(missing_pct[missing_pct > 0])

# Save datasets
patients.to_csv('healthcare_patients.csv', index=False)
timeseries.to_csv('healthcare_timeseries.csv', index=False)
outcomes.to_csv('healthcare_outcomes.csv', index=False)

print("\n=== FILES SAVED ===")
print("- healthcare_patients.csv")
print("- healthcare_timeseries.csv") 
print("- healthcare_outcomes.csv")
print("- healthcare_data_visualization.png")
print("- detailed_vitals_timeseries.png")
print("- correlation_heatmap.png")

# Sample data preview
print("\n=== SAMPLE DATA PREVIEW ===")
print("\nPatients (first 3 rows):")
print(patients.head(3))

print("\nTime Series (first 5 rows):")
print(timeseries.head(5))

print("\nOutcomes (first 3 rows):")
print(outcomes.head(3))

