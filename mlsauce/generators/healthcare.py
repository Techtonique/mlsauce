import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns

class HealthcareTimeSeriesGenerator:
    def __init__(self, seed=42):
        np.random.seed(seed)
        random.seed(seed)
        
        # Define realistic ranges for vital signs and lab values
        self.vital_ranges = {
            'heart_rate': (60, 100, 10),  # (min, max, std)
            'systolic_bp': (90, 140, 15),
            'diastolic_bp': (60, 90, 10),
            'temperature': (36.1, 37.2, 0.3),  # Celsius
            'respiratory_rate': (12, 20, 3),
            'oxygen_saturation': (95, 100, 2)
        }
        
        self.lab_ranges = {
            'glucose': (70, 110, 20),  # mg/dL
            'creatinine': (0.6, 1.2, 0.2),  # mg/dL
            'hemoglobin': (12, 16, 1.5),  # g/dL
            'white_blood_cells': (4000, 11000, 1500),  # cells/μL
            'sodium': (136, 145, 3),  # mEq/L
            'potassium': (3.5, 5.0, 0.4)  # mEq/L
        }
        
        # Medical conditions that affect vital signs
        self.conditions = [
            'hypertension', 'diabetes', 'copd', 'heart_failure', 
            'kidney_disease', 'anemia', 'infection', 'healthy'
        ]
        
    def generate_patient_demographics(self, n_patients=100):
        """Generate realistic patient demographics"""
        patients = []
        
        for i in range(n_patients):
            age = np.random.normal(65, 15)  # Average hospital patient age
            age = max(18, min(95, int(age)))  # Clamp between 18-95
            
            gender = random.choice(['M', 'F'])
            
            # Assign conditions based on age and gender
            conditions = self._assign_conditions(age, gender)
            
            patient = {
                'patient_id': f'P{i+1:04d}',
                'age': age,
                'gender': gender,
                'conditions': conditions,
                'admission_date': self._random_date(),
                'length_of_stay': random.randint(1, 30)
            }
            patients.append(patient)
            
        return pd.DataFrame(patients)
    
    def _assign_conditions(self, age, gender):
        """Assign medical conditions based on demographics"""
        conditions = []
        
        # Age-related condition probabilities
        if age > 50:
            if random.random() < 0.3: conditions.append('hypertension')
            if random.random() < 0.15: conditions.append('diabetes')
            if random.random() < 0.1: conditions.append('heart_failure')
            
        if age > 60:
            if random.random() < 0.08: conditions.append('copd')
            if random.random() < 0.12: conditions.append('kidney_disease')
            
        if gender == 'F' and random.random() < 0.1:
            conditions.append('anemia')
            
        if random.random() < 0.05:
            conditions.append('infection')
            
        if not conditions:
            conditions.append('healthy')
            
        return conditions
    
    def _random_date(self):
        """Generate random date within last 2 years"""
        start_date = datetime.now() - timedelta(days=730)
        random_days = random.randint(0, 730)
        return start_date + timedelta(days=random_days)
    
    def generate_time_series(self, patients_df, measurements_per_day=4, include_missing=True):
        """Generate time series data for all patients"""
        all_measurements = []
        
        for _, patient in patients_df.iterrows():
            patient_measurements = self._generate_patient_timeseries(
                patient, measurements_per_day, include_missing
            )
            all_measurements.extend(patient_measurements)
            
        return pd.DataFrame(all_measurements)
    
    def _generate_patient_timeseries(self, patient, measurements_per_day, include_missing):
        """Generate time series for a single patient"""
        measurements = []
        
        start_date = patient['admission_date']
        length_of_stay = patient['length_of_stay']
        conditions = patient['conditions']
        
        # Generate measurements for each day
        for day in range(length_of_stay):
            current_date = start_date + timedelta(days=day)
            
            # Generate multiple measurements per day
            for measurement_num in range(measurements_per_day):
                timestamp = current_date + timedelta(
                    hours=measurement_num * (24 / measurements_per_day)
                )
                
                measurement = {
                    'patient_id': patient['patient_id'],
                    'timestamp': timestamp,
                    'day_of_stay': day + 1
                }
                
                # Generate vital signs
                vitals = self._generate_vitals(conditions, day, patient['age'])
                measurement.update(vitals)
                
                # Generate lab values (less frequent)
                if measurement_num == 0 or random.random() < 0.1:  # Morning labs or random
                    labs = self._generate_labs(conditions, day)
                    measurement.update(labs)
                else:
                    # Add NaN for missing lab values
                    for lab in self.lab_ranges.keys():
                        measurement[lab] = np.nan
                
                # Add some random missing values if requested
                if include_missing:
                    measurement = self._add_missing_values(measurement)
                
                measurements.append(measurement)
                
        return measurements
    
    def _generate_vitals(self, conditions, day, age):
        """Generate vital signs based on patient conditions and progression"""
        vitals = {}
        
        for vital, (base_min, base_max, base_std) in self.vital_ranges.items():
            base_mean = (base_min + base_max) / 2
            
            # Adjust based on conditions
            mean_adjustment = 0
            std_adjustment = 1
            
            if 'hypertension' in conditions:
                if 'systolic' in vital:
                    mean_adjustment += 20
                elif 'diastolic' in vital:
                    mean_adjustment += 10
                    
            if 'heart_failure' in conditions:
                if vital == 'heart_rate':
                    mean_adjustment += 15
                elif vital == 'respiratory_rate':
                    mean_adjustment += 5
                elif vital == 'oxygen_saturation':
                    mean_adjustment -= 3
                    
            if 'copd' in conditions:
                if vital == 'respiratory_rate':
                    mean_adjustment += 8
                elif vital == 'oxygen_saturation':
                    mean_adjustment -= 5
                    
            if 'infection' in conditions:
                if vital == 'temperature':
                    mean_adjustment += np.random.normal(1.5, 0.5)
                elif vital == 'heart_rate':
                    mean_adjustment += 20
                    
            # Age adjustments
            if age > 70:
                if vital == 'systolic_bp':
                    mean_adjustment += 10
                elif vital == 'heart_rate':
                    mean_adjustment -= 5
                    
            # Day progression (recovery/deterioration)
            day_effect = np.sin(day * 0.2) * 2  # Subtle oscillation
            
            # Generate value
            adjusted_mean = base_mean + mean_adjustment + day_effect
            adjusted_std = base_std * std_adjustment
            
            value = np.random.normal(adjusted_mean, adjusted_std)
            
            # Apply realistic bounds
            if vital == 'temperature':
                value = max(35.0, min(42.0, value))
            elif vital == 'oxygen_saturation':
                value = max(70, min(100, value))
            elif vital == 'heart_rate':
                value = max(40, min(180, value))
            elif 'bp' in vital:
                value = max(40, min(200, value))
            elif vital == 'respiratory_rate':
                value = max(8, min(40, value))
                
            vitals[vital] = round(value, 1)
            
        return vitals
    
    def _generate_labs(self, conditions, day):
        """Generate lab values based on conditions"""
        labs = {}
        
        for lab, (base_min, base_max, base_std) in self.lab_ranges.items():
            base_mean = (base_min + base_max) / 2
            
            # Condition-based adjustments
            mean_adjustment = 0
            
            if 'diabetes' in conditions and lab == 'glucose':
                mean_adjustment += np.random.normal(50, 20)
                
            if 'kidney_disease' in conditions:
                if lab == 'creatinine':
                    mean_adjustment += np.random.normal(1.0, 0.5)
                elif lab == 'potassium':
                    mean_adjustment += np.random.normal(0.5, 0.2)
                    
            if 'anemia' in conditions and lab == 'hemoglobin':
                mean_adjustment -= np.random.normal(3, 1)
                
            if 'infection' in conditions and lab == 'white_blood_cells':
                mean_adjustment += np.random.normal(5000, 2000)
                
            # Generate value
            adjusted_mean = base_mean + mean_adjustment
            value = np.random.normal(adjusted_mean, base_std)
            
            # Apply bounds
            if lab == 'glucose':
                value = max(30, min(500, value))
            elif lab == 'creatinine':
                value = max(0.3, min(10.0, value))
            elif lab == 'hemoglobin':
                value = max(5.0, min(20.0, value))
            elif lab == 'white_blood_cells':
                value = max(1000, min(50000, value))
            elif lab == 'sodium':
                value = max(120, min(160, value))
            elif lab == 'potassium':
                value = max(2.0, min(7.0, value))
                
            labs[lab] = round(value, 2)
            
        return labs
    
    def _add_missing_values(self, measurement, missing_prob=0.05):
        """Randomly add missing values to simulate real-world data"""
        for key, value in measurement.items():
            if key not in ['patient_id', 'timestamp', 'day_of_stay'] and not pd.isna(value):
                if random.random() < missing_prob:
                    measurement[key] = np.nan
        return measurement
    
    def generate_outcomes(self, patients_df, timeseries_df):
        """Generate patient outcomes based on their data"""
        outcomes = []
        
        for _, patient in patients_df.iterrows():
            patient_data = timeseries_df[
                timeseries_df['patient_id'] == patient['patient_id']
            ]
            
            # Calculate outcome probability based on conditions and vital trends
            readmission_prob = self._calculate_readmission_risk(patient, patient_data)
            mortality_risk = self._calculate_mortality_risk(patient, patient_data)
            
            outcome = {
                'patient_id': patient['patient_id'],
                'readmitted_30_days': random.random() < readmission_prob,
                'mortality_risk_score': round(mortality_risk, 3),
                'length_of_stay_actual': patient['length_of_stay'],
                'discharge_disposition': self._assign_discharge_disposition(patient, mortality_risk)
            }
            outcomes.append(outcome)
            
        return pd.DataFrame(outcomes)
    
    def _calculate_readmission_risk(self, patient, patient_data):
        """Calculate 30-day readmission risk"""
        base_risk = 0.1  # 10% base readmission rate
        
        # Condition-based risk
        if 'heart_failure' in patient['conditions']:
            base_risk += 0.15
        if 'diabetes' in patient['conditions']:
            base_risk += 0.08
        if 'kidney_disease' in patient['conditions']:
            base_risk += 0.12
            
        # Age-based risk
        if patient['age'] > 75:
            base_risk += 0.1
            
        # Vital signs instability
        if len(patient_data) > 0:
            hr_std = patient_data['heart_rate'].std()
            if hr_std > 15:
                base_risk += 0.05
                
        return min(0.8, base_risk)
    
    def _calculate_mortality_risk(self, patient, patient_data):
        """Calculate mortality risk score"""
        risk_score = 0
        
        # Age component
        risk_score += patient['age'] * 0.02
        
        # Condition components
        condition_weights = {
            'heart_failure': 0.3,
            'kidney_disease': 0.25,
            'copd': 0.2,
            'infection': 0.15,
            'diabetes': 0.1,
            'hypertension': 0.05
        }
        
        for condition in patient['conditions']:
            if condition in condition_weights:
                risk_score += condition_weights[condition]
                
        # Vital signs component
        if len(patient_data) > 0:
            # Abnormal vital signs increase risk
            avg_o2_sat = patient_data['oxygen_saturation'].mean()
            if avg_o2_sat < 92:
                risk_score += 0.2
                
            avg_temp = patient_data['temperature'].mean()
            if avg_temp > 38.5:
                risk_score += 0.15
                
        return min(1.0, risk_score)
    
    def _assign_discharge_disposition(self, patient, mortality_risk):
        """Assign discharge disposition"""
        if mortality_risk > 0.7:
            return random.choice(['ICU Transfer', 'Deceased'])
        elif mortality_risk > 0.4:
            return random.choice(['Skilled Nursing Facility', 'Home with Services'])
        else:
            return random.choice(['Home', 'Home with Services', 'Rehabilitation'])

    def create_visualizations(self, patients, timeseries, outcomes):
        """Create comprehensive visualizations of the healthcare data"""
        
        # Set up the plotting style
        plt.style.use('default')
        sns.set_palette("husl")
        
        # Create figure with subplots
        fig = plt.figure(figsize=(20, 16))
        
        # 1. Patient Demographics
        plt.subplot(3, 4, 1)
        patients['age'].hist(bins=15, alpha=0.7, color='skyblue', edgecolor='black')
        plt.title('Age Distribution', fontsize=12, fontweight='bold')
        plt.xlabel('Age')
        plt.ylabel('Frequency')
        
        # 2. Gender Distribution
        plt.subplot(3, 4, 2)
        gender_counts = patients['gender'].value_counts()
        plt.pie(gender_counts.values, labels=gender_counts.index, autopct='%1.1f%%', 
                colors=['lightcoral', 'lightblue'])
        plt.title('Gender Distribution', fontsize=12, fontweight='bold')
        
        # 3. Medical Conditions Frequency
        plt.subplot(3, 4, 3)
        all_conditions = [cond for conditions in patients['conditions'] for cond in conditions]
        condition_counts = pd.Series(all_conditions).value_counts()
        condition_counts.plot(kind='bar', color='lightgreen', alpha=0.8)
        plt.title('Medical Conditions Frequency', fontsize=12, fontweight='bold')
        plt.xticks(rotation=45)
        plt.ylabel('Count')
        
        # 4. Length of Stay Distribution
        plt.subplot(3, 4, 4)
        patients['length_of_stay'].hist(bins=15, alpha=0.7, color='orange', edgecolor='black')
        plt.title('Length of Stay Distribution', fontsize=12, fontweight='bold')
        plt.xlabel('Days')
        plt.ylabel('Frequency')
        
        # 5. Heart Rate Time Series for Sample Patients
        plt.subplot(3, 4, 5)
        sample_patients = patients['patient_id'].head(5)
        for pid in sample_patients:
            patient_data = timeseries[timeseries['patient_id'] == pid].copy()
            if len(patient_data) > 0:
                patient_data = patient_data.sort_values('timestamp')
                plt.plot(patient_data['day_of_stay'], patient_data['heart_rate'], 
                        marker='o', markersize=3, alpha=0.7, label=pid)
        plt.title('Heart Rate Over Time (Sample Patients)', fontsize=12, fontweight='bold')
        plt.xlabel('Day of Stay')
        plt.ylabel('Heart Rate (bpm)')
        plt.legend(fontsize=8)
        
        # 6. Blood Pressure Correlation
        plt.subplot(3, 4, 6)
        clean_bp = timeseries.dropna(subset=['systolic_bp', 'diastolic_bp'])
        plt.scatter(clean_bp['systolic_bp'], clean_bp['diastolic_bp'], 
                alpha=0.5, s=10, color='red')
        plt.title('Blood Pressure Correlation', fontsize=12, fontweight='bold')
        plt.xlabel('Systolic BP')
        plt.ylabel('Diastolic BP')
        
        # 7. Temperature vs Heart Rate
        plt.subplot(3, 4, 7)
        clean_temp_hr = timeseries.dropna(subset=['temperature', 'heart_rate'])
        plt.scatter(clean_temp_hr['temperature'], clean_temp_hr['heart_rate'], 
                alpha=0.5, s=10, color='purple')
        plt.title('Temperature vs Heart Rate', fontsize=12, fontweight='bold')
        plt.xlabel('Temperature (°C)')
        plt.ylabel('Heart Rate (bpm)')
        
        # 8. Vital Signs Distribution
        plt.subplot(3, 4, 8)
        vital_cols = ['heart_rate', 'respiratory_rate', 'oxygen_saturation']
        timeseries[vital_cols].boxplot()
        plt.title('Vital Signs Distribution', fontsize=12, fontweight='bold')
        plt.xticks(rotation=45)
        
        # 9. Lab Values Over Time
        plt.subplot(3, 4, 9)
        sample_patient = timeseries[timeseries['patient_id'] == sample_patients.iloc[0]].copy()
        sample_patient = sample_patient.sort_values('timestamp')
        
        # Plot glucose if available
        glucose_data = sample_patient.dropna(subset=['glucose'])
        if len(glucose_data) > 0:
            plt.plot(glucose_data['day_of_stay'], glucose_data['glucose'], 
                    'o-', color='green', label='Glucose')
        
        # Plot creatinine if available
        creat_data = sample_patient.dropna(subset=['creatinine'])
        if len(creat_data) > 0:
            plt.twinx()
            plt.plot(creat_data['day_of_stay'], creat_data['creatinine'], 
                    'o-', color='blue', label='Creatinine')
            plt.ylabel('Creatinine (mg/dL)', color='blue')
        
        plt.title(f'Lab Values - {sample_patients.iloc[0]}', fontsize=12, fontweight='bold')
        plt.xlabel('Day of Stay')
        plt.ylabel('Glucose (mg/dL)', color='green')
        
        # 10. Readmission Risk by Age Group
        plt.subplot(3, 4, 10)
        merged_data = patients.merge(outcomes, on='patient_id')
        merged_data['age_group'] = pd.cut(merged_data['age'], 
                                        bins=[0, 40, 60, 80, 100], 
                                        labels=['<40', '40-60', '60-80', '80+'])
        readmission_by_age = merged_data.groupby('age_group')['readmitted_30_days'].mean()
        readmission_by_age.plot(kind='bar', color='salmon', alpha=0.8)
        plt.title('30-Day Readmission Rate by Age', fontsize=12, fontweight='bold')
        plt.ylabel('Readmission Rate')
        plt.xticks(rotation=0)
        
        # 11. Mortality Risk Distribution
        plt.subplot(3, 4, 11)
        outcomes['mortality_risk_score'].hist(bins=20, alpha=0.7, color='darkred', edgecolor='black')
        plt.title('Mortality Risk Score Distribution', fontsize=12, fontweight='bold')
        plt.xlabel('Risk Score')
        plt.ylabel('Frequency')
        
        # 12. Missing Data Heatmap
        plt.subplot(3, 4, 12)
        # Calculate missing data percentage for each column
        missing_data = timeseries.isnull().sum() / len(timeseries) * 100
        missing_data = missing_data[missing_data > 0].sort_values(ascending=False)
        
        if len(missing_data) > 0:
            missing_data.plot(kind='bar', color='gray', alpha=0.8)
            plt.title('Missing Data Percentage', fontsize=12, fontweight='bold')
            plt.ylabel('Missing %')
            plt.xticks(rotation=45)
        else:
            plt.text(0.5, 0.5, 'No Missing Data', ha='center', va='center', 
                    transform=plt.gca().transAxes, fontsize=14)
            plt.title('Missing Data Percentage', fontsize=12, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig('healthcare_data_visualization.png', dpi=300, bbox_inches='tight')
        print("Visualization saved as: healthcare_data_visualization.png")
        plt.show()        
        # Create additional detailed plots
        self.create_detailed_plots(patients, timeseries)

    def create_detailed_plots(self, patients, timeseries):
        """Create additional detailed visualizations"""
        
        # Time Series Plot for Multiple Vital Signs
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Select a patient with longer stay for better visualization
        long_stay_patients = patients[patients['length_of_stay'] >= 7]['patient_id'].head(3)
        
        vital_signs = ['heart_rate', 'systolic_bp', 'temperature', 'oxygen_saturation']
        colors = ['red', 'blue', 'orange', 'green']
        
        for i, vital in enumerate(vital_signs):
            ax = axes[i//2, i%2]
            
            for j, pid in enumerate(long_stay_patients):
                patient_data = timeseries[timeseries['patient_id'] == pid].copy()
                patient_data = patient_data.sort_values('timestamp')
                clean_data = patient_data.dropna(subset=[vital])
                
                if len(clean_data) > 0:
                    ax.plot(clean_data['day_of_stay'], clean_data[vital], 
                        marker='o', label=pid, alpha=0.7, linewidth=2)
            
            ax.set_title(f'{vital.replace("_", " ").title()} Over Time', fontweight='bold')
            ax.set_xlabel('Day of Stay')
            ax.set_ylabel(vital.replace('_', ' ').title())
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('detailed_vitals_timeseries.png', dpi=300, bbox_inches='tight')
        print("Detailed vital signs plot saved as: detailed_vitals_timeseries.png")
        plt.show()
        
        # Correlation Heatmap
        plt.figure(figsize=(12, 10))
        numeric_cols = ['heart_rate', 'systolic_bp', 'diastolic_bp', 'temperature', 
                    'respiratory_rate', 'oxygen_saturation', 'glucose', 'creatinine', 
                    'hemoglobin', 'white_blood_cells', 'sodium', 'potassium']
        
        correlation_matrix = timeseries[numeric_cols].corr()
        
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0,
                    square=True, fmt='.2f', cbar_kws={'shrink': 0.8})
        plt.title('Healthcare Parameters Correlation Matrix', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig('correlation_heatmap.png', dpi=300, bbox_inches='tight')
        print("Correlation heatmap saved as: correlation_heatmap.png")
        plt.show()
        