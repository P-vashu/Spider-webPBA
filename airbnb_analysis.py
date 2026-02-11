import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import shap
import matplotlib.pyplot as plt

# Load the three datasets
print("Loading datasets...")
brighton = pd.read_csv('BrightonPerformanceData.csv')
joshua = pd.read_csv('JoshuaPerformanceData.csv')
malibu = pd.read_csv('MalibuPerformanceData.csv')

print(f"Brighton shape: {brighton.shape}")
print(f"Joshua shape: {joshua.shape}")
print(f"Malibu shape: {malibu.shape}")

# Convert last_seen to datetime formatted as YYYY-MM-DD
print("\nConverting last_seen to datetime...")
brighton['last_seen'] = pd.to_datetime(brighton['last_seen'], errors='coerce').dt.strftime('%Y-%m-%d')
joshua['last_seen'] = pd.to_datetime(joshua['last_seen'], errors='coerce').dt.strftime('%Y-%m-%d')
malibu['last_seen'] = pd.to_datetime(malibu['last_seen'], errors='coerce').dt.strftime('%Y-%m-%d')

# Get weekly periods that appear across all three locations
print("\nIdentifying common weekly periods...")
brighton_periods = set(brighton['last_seen'].dropna())
joshua_periods = set(joshua['last_seen'].dropna())
malibu_periods = set(malibu['last_seen'].dropna())

common_periods = brighton_periods & joshua_periods & malibu_periods
print(f"Common periods across all locations: {len(common_periods)}")

# Filter to only retain common weekly periods
brighton = brighton[brighton['last_seen'].isin(common_periods)]
joshua = joshua[joshua['last_seen'].isin(common_periods)]
malibu = malibu[malibu['last_seen'].isin(common_periods)]

print(f"\nAfter filtering to common periods:")
print(f"Brighton shape: {brighton.shape}")
print(f"Joshua shape: {joshua.shape}")
print(f"Malibu shape: {malibu.shape}")

# Ensure required columns are numeric
numeric_columns = ['ADR (USD)', 'Occupancy Rate', 'cleaning_fee', 'Bedrooms', 'Revenue (USD)']

print("\nConverting numeric columns and dropping invalid rows...")
for df in [brighton, joshua, malibu]:
    for col in numeric_columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

# Drop rows with missing or non-numeric values in these columns
brighton = brighton.dropna(subset=numeric_columns)
joshua = joshua.dropna(subset=numeric_columns)
malibu = malibu.dropna(subset=numeric_columns)

print(f"\nAfter dropping invalid rows:")
print(f"Brighton shape: {brighton.shape}")
print(f"Joshua shape: {joshua.shape}")
print(f"Malibu shape: {malibu.shape}")

# Standardize property type and city to lowercase
print("\nStandardizing Property Type and City to lowercase...")
brighton['Property Type'] = brighton['Property Type'].str.lower()
brighton['City'] = brighton['City'].str.lower()

joshua['Property Type'] = joshua['Property Type'].str.lower()
joshua['City'] = joshua['City'].str.lower()

malibu['Property Type'] = malibu['Property Type'].str.lower()
malibu['City'] = malibu['City'].str.lower()

# Merge the three datasets
print("\nMerging datasets...")
combined = pd.concat([brighton, joshua, malibu], ignore_index=True)
print(f"Combined dataset shape: {combined.shape}")

# Create Reporting Month as integer (1-12)
print("\nCreating Reporting Month as integer...")
combined['Reporting Month Integer'] = pd.to_datetime(combined['Reporting Month'], format='%Y-%m').dt.month

# Prepare features for modeling
print("\nPreparing features for modeling...")
feature_columns = ['ADR (USD)', 'Occupancy Rate', 'cleaning_fee', 'Bedrooms',
                   'Property Type', 'City', 'Reporting Month Integer']

# Create feature matrix with one-hot encoding
X = combined[feature_columns].copy()
y = combined['Revenue (USD)'].copy()

# One-hot encode Property Type and City with drop_first=True
X_encoded = pd.get_dummies(X, columns=['Property Type', 'City'], drop_first=True)

print(f"Feature matrix shape after encoding: {X_encoded.shape}")
print(f"Features: {list(X_encoded.columns)}")

# Split into training and test sets (80-20 with random_state=42)
print("\nSplitting data into train and test sets...")
X_train, X_test, y_train, y_test = train_test_split(
    X_encoded, y, test_size=0.2, random_state=42
)

print(f"Training set size: {X_train.shape}")
print(f"Test set size: {X_test.shape}")

# Fit Linear Regression on training set
print("\nFitting Linear Regression model on training set...")
model = LinearRegression()
model.fit(X_train, y_train)

# Evaluate on test set
y_pred_test = model.predict(X_test)
test_r2 = r2_score(y_test, y_pred_test)
print(f"Test-set R²: {test_r2:.4f}")

# Fit final model on full dataset
print("\nFitting final Linear Regression model on full dataset...")
final_model = LinearRegression()
final_model.fit(X_encoded, y)

# Compute SHAP values using LinearExplainer with feature_perturbation="interventional"
print("\nComputing SHAP values...")
explainer = shap.LinearExplainer(final_model, X_encoded, feature_perturbation="interventional")
shap_values = explainer.shap_values(X_encoded)

print(f"SHAP values shape: {shap_values.shape}")

# Calculate mean absolute SHAP values for each feature
mean_abs_shap = np.mean(np.abs(shap_values), axis=0)

# Create a mapping from encoded feature names to original features
feature_names = list(X_encoded.columns)

# Calculate SHAP importance for each original feature
shap_importance = {}

# Continuous features
continuous_features = ['ADR (USD)', 'Occupancy Rate', 'cleaning_fee', 'Bedrooms', 'Reporting Month Integer']
for feat in continuous_features:
    if feat in feature_names:
        idx = feature_names.index(feat)
        shap_importance[feat] = mean_abs_shap[idx]

# Property Type (sum of all one-hot encoded property type SHAP means)
property_type_shap = []
for i, name in enumerate(feature_names):
    if name.startswith('Property Type_'):
        property_type_shap.append(mean_abs_shap[i])
shap_importance['Property Type'] = sum(property_type_shap) if property_type_shap else 0.0

# City (sum of all one-hot encoded city SHAP means)
city_shap = []
for i, name in enumerate(feature_names):
    if name.startswith('City_'):
        city_shap.append(mean_abs_shap[i])
shap_importance['City'] = sum(city_shap) if city_shap else 0.0

# Find maximum SHAP value and its index
shap_values_flat = shap_values.flatten()
max_shap_value = np.max(np.abs(shap_values_flat))
max_shap_idx_flat = np.argmax(np.abs(shap_values_flat))

# Convert flat index to row index
max_shap_row_idx = max_shap_idx_flat // shap_values.shape[1]

print("\n" + "="*60)
print("RESULTS")
print("="*60)
print(f"\nTest-set R²: {test_r2:.4f}")
print(f"\nMean absolute SHAP value for ADR (USD): {shap_importance['ADR (USD)']:.4f}")
print(f"Mean absolute SHAP value for Occupancy Rate: {shap_importance['Occupancy Rate']:.4f}")
print(f"Mean absolute SHAP value for cleaning_fee: {shap_importance['cleaning_fee']:.4f}")
print(f"Mean absolute SHAP value for Bedrooms: {shap_importance['Bedrooms']:.4f}")
print(f"Mean absolute SHAP value for Property Type: {shap_importance['Property Type']:.4f}")
print(f"Mean absolute SHAP value for City: {shap_importance['City']:.4f}")
print(f"Mean absolute SHAP value for Reporting Month: {shap_importance['Reporting Month Integer']:.4f}")
print(f"\nMaximum single SHAP value: {max_shap_value:.4f}")
print(f"Index (row identifier) where maximum SHAP value occurs: {max_shap_row_idx}")
print("="*60)

# Create scatter plot: ADR vs Revenue colored by Occupancy Rate
print("\nCreating scatter plot...")
plt.figure(figsize=(10, 6))
scatter = plt.scatter(combined['ADR (USD)'], combined['Revenue (USD)'],
                     c=combined['Occupancy Rate'], cmap='viridis', alpha=0.6)
plt.xlabel('ADR (USD)', fontsize=12)
plt.ylabel('Revenue (USD)', fontsize=12)
plt.title('ADR vs Revenue (colored by Occupancy Rate)', fontsize=14)
plt.colorbar(scatter, label='Occupancy Rate')
plt.tight_layout()
plt.savefig('adr_revenue_scatter.png', dpi=300)
print("Scatter plot saved as 'adr_revenue_scatter.png'")

print("\nAnalysis complete!")
