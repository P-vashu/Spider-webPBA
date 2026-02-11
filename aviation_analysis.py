"""
Aviation Infrastructure Causal Analysis
Comprehensive analysis of airlines, airplanes, and airports datasets
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import genpareto
from sklearn.cluster import DBSCAN
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.regression.quantile_regression import QuantReg
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)

print("="*80)
print("AVIATION INFRASTRUCTURE CAUSAL ANALYSIS")
print("="*80)

# ============================================================================
# STEP 1: DATA LOADING AND PREPROCESSING
# ============================================================================
print("\n[STEP 1] Loading and preprocessing datasets...")

# Load datasets
airlines = pd.read_csv('airlines.csv')
airplanes = pd.read_csv('airplanes.csv')
airports = pd.read_csv('airports.csv')

print(f"Original airlines shape: {airlines.shape}")
print(f"Original airplanes shape: {airplanes.shape}")
print(f"Original airports shape: {airports.shape}")

# Filter airlines where Active == 'Y'
airlines_active = airlines[airlines['Active'] == 'Y'].copy()
print(f"\nActive airlines shape: {airlines_active.shape}")

# Inner join filtered airlines with airports on Country
merged = pd.merge(airlines_active, airports, on='Country', how='inner')
print(f"Merged dataset shape (before altitude cleaning): {merged.shape}")

# Remove rows with missing or non-numeric Altitude values
merged['Altitude'] = pd.to_numeric(merged['Altitude'], errors='coerce')
merged = merged.dropna(subset=['Altitude'])
print(f"Merged dataset shape (after altitude cleaning): {merged.shape}")

# Standardize Altitude using z-score normalization
altitude_mean = merged['Altitude'].mean()
altitude_std = merged['Altitude'].std()
merged['Altitude_standardized'] = (merged['Altitude'] - altitude_mean) / altitude_std

print(f"\nAltitude statistics:")
print(f"  Mean: {altitude_mean:.2f}")
print(f"  Std: {altitude_std:.2f}")

# ============================================================================
# STEP 2: EXTREME VALUE THEORY - GENERALIZED PARETO DISTRIBUTION
# ============================================================================
print("\n[STEP 2] Applying Extreme Value Theory...")

# Calculate 95th percentile threshold
threshold = np.percentile(merged['Altitude_standardized'], 95)
print(f"95th percentile threshold: {threshold:.6f}")

# Get exceedances
exceedances = merged['Altitude_standardized'][merged['Altitude_standardized'] > threshold] - threshold

# Fit GPD using maximum likelihood estimation
gpd_params = genpareto.fit(exceedances, floc=0)
shape_param = gpd_params[0]
scale_param = gpd_params[2]

print(f"\nGeneralized Pareto Distribution Parameters:")
print(f"  Shape parameter (ξ): {shape_param:.6f}")
print(f"  Scale parameter (σ): {scale_param:.6f}")

# ============================================================================
# STEP 3: QUANTILE REGRESSION
# ============================================================================
print("\n[STEP 3] Performing Quantile Regression...")

# Prepare data for quantile regression (remove any NaN in Longitude)
qr_data = merged[['Longitude', 'Altitude']].dropna()

# Fit quantile regression for 10th, 50th, and 90th percentiles
quantiles = [0.10, 0.50, 0.90]
qr_results = {}

for q in quantiles:
    model = QuantReg(qr_data['Altitude'], sm.add_constant(qr_data['Longitude']))
    result = model.fit(q=q)
    qr_results[q] = result
    print(f"\n{int(q*100)}th percentile quantile regression:")
    print(f"  Intercept: {result.params['const']:.5f}")
    print(f"  Slope (Longitude): {result.params['Longitude']:.5f}")

# Extract specific coefficients as requested
slope_90th = qr_results[0.90].params['Longitude']
intercept_10th = qr_results[0.10].params['const']

print(f"\nRequested coefficients:")
print(f"  90th percentile slope: {slope_90th:.5f}")
print(f"  10th percentile intercept: {intercept_10th:.5f}")

# ============================================================================
# STEP 4: PROPENSITY SCORE MATCHING - CAUSAL INFERENCE
# ============================================================================
print("\n[STEP 4] Implementing Propensity Score Matching...")

# Count active airlines per country
airline_counts = airlines_active.groupby('Country').size().reset_index(name='airline_count')
print(f"\nAirline counts per country calculated: {len(airline_counts)} countries")

# Create treatment indicator (1 if > 50 active airlines, 0 if <= 50)
airline_counts['treatment'] = (airline_counts['airline_count'] > 50).astype(int)
print(f"Countries with treatment=1 (>50 airlines): {airline_counts['treatment'].sum()}")
print(f"Countries with treatment=0 (<=50 airlines): {len(airline_counts) - airline_counts['treatment'].sum()}")

# Merge treatment indicator with merged dataset
merged_psm = pd.merge(merged, airline_counts[['Country', 'treatment']], on='Country', how='left')
merged_psm = merged_psm.dropna(subset=['Latitude', 'Longitude', 'Altitude_standardized'])

print(f"\nDataset for PSM: {merged_psm.shape}")

# Fit logistic regression for propensity scores
X = merged_psm[['Latitude', 'Longitude']]
y = merged_psm['treatment']

logit_model = LogisticRegression(random_state=42, max_iter=1000)
logit_model.fit(X, y)

# Calculate propensity scores
propensity_scores = logit_model.predict_proba(X)[:, 1]
merged_psm['propensity_score'] = propensity_scores

# Calculate logit of propensity score
merged_psm['logit_ps'] = np.log(propensity_scores / (1 - propensity_scores + 1e-10))

# Calculate caliper (0.2 standard deviations of logit of propensity score)
caliper = 0.2 * merged_psm['logit_ps'].std()
print(f"Caliper width: {caliper:.6f}")

# Perform 1-to-1 nearest neighbor matching without replacement using optimized approach
treated = merged_psm[merged_psm['treatment'] == 1].copy()
control = merged_psm[merged_psm['treatment'] == 0].copy()

print(f"Treated units: {len(treated)}")
print(f"Control units: {len(control)}")

# Use sampling to make the matching more manageable for large datasets
# Sample treated units to reduce computational burden
if len(treated) > 10000:
    print(f"Sampling treated units for efficient matching...")
    treated = treated.sample(n=10000, random_state=42)
    print(f"Sampled treated units: {len(treated)}")

# Match treated to control using vectorized operations
matched_pairs = []
control_logits = control['logit_ps'].values
control_indices = control.index.values
control_outcomes = control['Altitude_standardized'].values
used_control_mask = np.zeros(len(control), dtype=bool)

for idx, treated_row in treated.iterrows():
    treated_logit = treated_row['logit_ps']

    # Find available controls
    available_mask = ~used_control_mask
    if not np.any(available_mask):
        break

    # Calculate distances to all available controls
    distances = np.abs(control_logits[available_mask] - treated_logit)

    # Find nearest within caliper
    if len(distances) > 0 and distances.min() <= caliper:
        # Get the index in the available array
        nearest_idx_in_available = distances.argmin()
        # Map back to original control index
        available_indices = np.where(available_mask)[0]
        nearest_idx = available_indices[nearest_idx_in_available]

        matched_pairs.append({
            'treated_idx': idx,
            'control_idx': control_indices[nearest_idx],
            'treated_outcome': treated_row['Altitude_standardized'],
            'control_outcome': control_outcomes[nearest_idx]
        })
        used_control_mask[nearest_idx] = True

print(f"Successfully matched pairs: {len(matched_pairs)}")

# Calculate Average Treatment Effect (ATE)
if len(matched_pairs) > 0:
    matched_df = pd.DataFrame(matched_pairs)
    ate = (matched_df['treated_outcome'] - matched_df['control_outcome']).mean()
    print(f"\nAverage Treatment Effect on standardized Altitude: {ate:.4f}")
else:
    ate = np.nan
    print("WARNING: No matched pairs found!")

# ============================================================================
# STEP 5: KOLMOGOROV-SMIRNOV TEST
# ============================================================================
print("\n[STEP 5] Conducting Kolmogorov-Smirnov Test...")

# Get countries with at least one active airline
countries_with_active = set(airlines_active['Country'].unique())

# Group A: airports from countries with at least 1 active airline
group_a = airports[airports['Country'].isin(countries_with_active)]['Timezone'].dropna()

# Group B: airports from countries with 0 active airlines
group_b = airports[~airports['Country'].isin(countries_with_active)]['Timezone'].dropna()

print(f"Group A size (countries with active airlines): {len(group_a)}")
print(f"Group B size (countries without active airlines): {len(group_b)}")

# Perform KS test
ks_statistic, ks_pvalue = stats.ks_2samp(group_a, group_b)

print(f"\nKolmogorov-Smirnov Test Results:")
print(f"  KS statistic: {ks_statistic:.5f}")
print(f"  KS p-value: {ks_pvalue:.8f}")

# ============================================================================
# STEP 6: AIRCRAFT TYPE DIVERSITY ANALYSIS
# ============================================================================
print("\n[STEP 6] Analyzing Aircraft Type Diversity...")

# Get unique aircraft names and sort alphabetically
unique_aircraft = airplanes['Name'].unique()
unique_aircraft_sorted = sorted(unique_aircraft)

# Assign integer identifiers starting from 1
aircraft_id_map = {name: idx + 1 for idx, name in enumerate(unique_aircraft_sorted)}

# Total count of unique aircraft types
total_aircraft_types = len(unique_aircraft_sorted)

# Find identifier for Boeing 737-800
boeing_737_800_id = aircraft_id_map.get('Boeing 737-800', None)

print(f"\nAircraft Type Analysis:")
print(f"  Total unique aircraft types: {total_aircraft_types}")
print(f"  Boeing 737-800 identifier: {boeing_737_800_id if boeing_737_800_id else 'Not found'}")

# ============================================================================
# STEP 7: DBSCAN CLUSTERING
# ============================================================================
print("\n[STEP 7] Applying DBSCAN Clustering...")

# Prepare data for clustering (remove NaN values)
clustering_data = airports[['Latitude', 'Longitude']].dropna()

# Apply DBSCAN
dbscan = DBSCAN(eps=0.5, min_samples=10, metric='euclidean')
clusters = dbscan.fit_predict(clustering_data)

# Count distinct clusters (excluding noise points labeled as -1)
unique_clusters = set(clusters)
num_clusters = len([c for c in unique_clusters if c != -1])

print(f"\nDBSCAN Clustering Results:")
print(f"  Total data points: {len(clusters)}")
print(f"  Unique cluster labels: {unique_clusters}")
print(f"  Number of distinct clusters (excluding noise): {num_clusters}")
print(f"  Noise points (cluster -1): {sum(clusters == -1)}")

# ============================================================================
# STEP 8: HEXBIN VISUALIZATION
# ============================================================================
print("\n[STEP 8] Generating Hexbin Visualization...")

# Create hexbin plot
plt.figure(figsize=(14, 8))
plt.hexbin(airports['Longitude'], airports['Latitude'],
           gridsize=50, cmap='YlOrRd', mincnt=1)
plt.colorbar(label='Count of Airports')
plt.xlabel('Longitude (degrees)', fontsize=12)
plt.ylabel('Latitude (degrees)', fontsize=12)
plt.title('Global Airport Distribution - Hexbin Plot', fontsize=14, fontweight='bold')
plt.xlim(-180, 180)
plt.ylim(-90, 90)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('airport_hexbin_plot.png', dpi=300, bbox_inches='tight')
print("Hexbin plot saved as 'airport_hexbin_plot.png'")

# ============================================================================
# FINAL SUMMARY REPORT
# ============================================================================
print("\n" + "="*80)
print("FINAL SUMMARY REPORT")
print("="*80)

summary_report = f"""
1. EXTREME VALUE THEORY (GPD):
   - Shape parameter (ξ): {shape_param:.6f}
   - Scale parameter (σ): {scale_param:.6f}

2. QUANTILE REGRESSION:
   - 90th percentile slope coefficient: {slope_90th:.5f}
   - 10th percentile intercept coefficient: {intercept_10th:.5f}

3. PROPENSITY SCORE MATCHING:
   - Average Treatment Effect (standardized Altitude): {ate:.4f}
   - Matched pairs: {len(matched_pairs)}

4. KOLMOGOROV-SMIRNOV TEST:
   - KS statistic: {ks_statistic:.5f}
   - KS p-value: {ks_pvalue:.8f}

5. AIRCRAFT TYPE DIVERSITY:
   - Total unique aircraft types: {total_aircraft_types}
   - Boeing 737-800 identifier: {boeing_737_800_id if boeing_737_800_id else 'Not found'}

6. DBSCAN CLUSTERING:
   - Number of distinct clusters (excluding noise): {num_clusters}

7. VISUALIZATION:
   - Hexbin plot generated and saved
"""

print(summary_report)

# Save results to file
with open('aviation_analysis_results.txt', 'w') as f:
    f.write("AVIATION INFRASTRUCTURE CAUSAL ANALYSIS - RESULTS\n")
    f.write("="*80 + "\n")
    f.write(summary_report)

print("\nResults saved to 'aviation_analysis_results.txt'")
print("Analysis complete!")
