"""
Google Stock Time Series Analysis
Advanced multi-granularity volatility assessment and statistical testing
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller, grangercausalitytests
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima.model import ARIMA
from scipy.stats import pearsonr

# Set random seed for reproducibility
np.random.seed(42)

print("=" * 80)
print("GOOGLE STOCK TIME SERIES ANALYSIS")
print("=" * 80)

# ============================================================================
# TASK 1: Load datasets and convert Date columns to datetime
# ============================================================================
print("\n[TASK 1] Loading datasets and converting Date columns...")

# Load the three datasets
df_daily = pd.read_csv('google-stock-dataset-Daily.csv')
df_weekly = pd.read_csv('google-stock-dataset-Weekly.csv')
df_monthly = pd.read_csv('google-stock-dataset-Monthly.csv')

# Convert Date columns to datetime with automatic format detection
df_daily['Date'] = pd.to_datetime(df_daily['Date'])
df_weekly['Date'] = pd.to_datetime(df_weekly['Date'])
df_monthly['Date'] = pd.to_datetime(df_monthly['Date'])

print(f"Daily dataset shape: {df_daily.shape}")
print(f"Weekly dataset shape: {df_weekly.shape}")
print(f"Monthly dataset shape: {df_monthly.shape}")

# ============================================================================
# TASK 2: Subset daily and weekly datasets (2015-01-01 to 2017-12-31)
# ============================================================================
print("\n[TASK 2] Subsetting daily and weekly datasets...")

# Subset daily data
df_daily = df_daily[(df_daily['Date'] >= '2015-01-01') &
                    (df_daily['Date'] <= '2017-12-31')].reset_index(drop=True)

# Subset weekly data
df_weekly = df_weekly[(df_weekly['Date'] >= '2015-01-01') &
                      (df_weekly['Date'] <= '2017-12-31')].reset_index(drop=True)

# Monthly data: retain all records (no subsetting)
print(f"Subsetted daily dataset shape: {df_daily.shape}")
print(f"Subsetted weekly dataset shape: {df_weekly.shape}")
print(f"Monthly dataset shape (no subsetting): {df_monthly.shape}")

# ============================================================================
# TASK 3: Augmented Dickey-Fuller test on daily Close column
# ============================================================================
print("\n[TASK 3] Augmented Dickey-Fuller test on daily Close column...")

adf_result = adfuller(df_daily['Close'], maxlag=12)
adf_statistic = adf_result[0]
adf_pvalue = adf_result[1]

print(f"ADF Test Statistic: {adf_statistic:.4f}")
print(f"ADF p-value: {adf_pvalue:.4f}")

# ============================================================================
# TASK 4: Time-series decomposition on weekly Close column
# ============================================================================
print("\n[TASK 4] Time-series decomposition on weekly Close column...")

# Set Date as index for seasonal decomposition
df_weekly_indexed = df_weekly.set_index('Date')

# Perform additive decomposition with period=52 weeks
decomposition = seasonal_decompose(df_weekly_indexed['Close'],
                                   model='additive',
                                   period=52)

# Calculate mean of trend component (excluding NaN)
trend_mean = decomposition.trend.dropna().mean()

# Calculate standard deviation of seasonal component (excluding NaN)
seasonal_std = decomposition.seasonal.dropna().std()

print(f"Trend component mean (excluding NaN): {trend_mean:.4f}")
print(f"Seasonal component std dev (excluding NaN): {seasonal_std:.4f}")

# ============================================================================
# TASK 5: ARIMA modeling on monthly Close column
# ============================================================================
print("\n[TASK 5] ARIMA modeling on monthly Close column...")

# Fit ARIMA model with parameters (p=2, d=1, q=2)
# Using default method (which implements maximum likelihood estimation)
arima_model = ARIMA(df_monthly['Close'], order=(2, 1, 2))
arima_fit = arima_model.fit()

# Get AIC value
aic_value = arima_fit.aic

print(f"ARIMA(2,1,2) AIC value: {aic_value:.4f}")

# ============================================================================
# TASK 6: Merge daily and weekly datasets on Date
# ============================================================================
print("\n[TASK 6] Merging daily and weekly datasets...")

# Inner join on Date column with suffixes
df_merged = pd.merge(df_daily, df_weekly, on='Date', how='inner',
                     suffixes=('_x', '_y'))

print(f"Merged dataset shape: {df_merged.shape}")
print(f"Number of observations in merged dataset: {len(df_merged)}")

# Verify exactly 140 observations
if len(df_merged) == 140:
    print("✓ Verification successful: Merged dataset contains exactly 140 observations")
else:
    print(f"✗ Warning: Merged dataset contains {len(df_merged)} observations (expected 140)")

# ============================================================================
# TASK 7: Granger causality test
# ============================================================================
print("\n[TASK 7] Granger causality test...")

# Construct 2D array: first column = Close_y, second column = Close_x
granger_array = df_merged[['Close_y', 'Close_x']].values

# Perform Granger causality test with max lag = 4
granger_results = grangercausalitytests(granger_array, maxlag=4, verbose=False)

# Extract F-statistic and p-value for lag 4
lag4_result = granger_results[4][0]['ssr_ftest']
f_statistic_lag4 = lag4_result[0]
p_value_lag4 = lag4_result[1]

print(f"Granger causality F-statistic (lag 4): {f_statistic_lag4:.4f}")
print(f"Granger causality p-value (lag 4): {p_value_lag4:.4f}")

# ============================================================================
# TASK 8: Bootstrap correlation between Volume_x and Volume_y
# ============================================================================
print("\n[TASK 8] Bootstrap correlation between Volume_x and Volume_y...")

# Reset random seed before bootstrap loop
np.random.seed(42)

# Number of bootstrap iterations
n_iterations = 1000
n_samples = 140

# Store bootstrap correlation coefficients
bootstrap_correlations = []

# Perform bootstrap
for i in range(n_iterations):
    # Randomly sample 140 row indices with replacement
    sample_indices = np.random.choice(n_samples, size=n_samples, replace=True)

    # Get bootstrap sample (keeping Volume_x and Volume_y pairs together)
    volume_x_sample = df_merged.iloc[sample_indices]['Volume_x'].values
    volume_y_sample = df_merged.iloc[sample_indices]['Volume_y'].values

    # Calculate Pearson correlation coefficient
    correlation, _ = pearsonr(volume_x_sample, volume_y_sample)
    bootstrap_correlations.append(correlation)

# Convert to numpy array
bootstrap_correlations = np.array(bootstrap_correlations)

# Calculate 95% confidence interval using percentile method
ci_lower = np.percentile(bootstrap_correlations, 2.5)
ci_upper = np.percentile(bootstrap_correlations, 97.5)

print(f"Bootstrap 95% CI lower bound: {ci_lower:.4f}")
print(f"Bootstrap 95% CI upper bound: {ci_upper:.4f}")

# ============================================================================
# TASK 9: Generate hexbin plot visualization
# ============================================================================
print("\n[TASK 9] Generating hexbin plot visualization...")

plt.figure(figsize=(10, 8))
plt.hexbin(df_merged['Close_x'], df_merged['Close_y'],
           gridsize=20, cmap='viridis', mincnt=1)
plt.colorbar(label='Count')
plt.xlabel('Close_x (Daily)', fontsize=12)
plt.ylabel('Close_y (Weekly)', fontsize=12)
plt.title('Hexbin Plot: Daily vs Weekly Close Prices', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('hexbin_plot_daily_vs_weekly.png', dpi=300, bbox_inches='tight')
print("✓ Hexbin plot saved as 'hexbin_plot_daily_vs_weekly.png'")

# ============================================================================
# SUMMARY OF RESULTS
# ============================================================================
print("\n" + "=" * 80)
print("SUMMARY OF RESULTS")
print("=" * 80)
print(f"\n1. Augmented Dickey-Fuller Test (Daily Close):")
print(f"   - Test Statistic: {adf_statistic:.4f}")
print(f"   - p-value: {adf_pvalue:.4f}")

print(f"\n2. Time-Series Decomposition (Weekly Close, period=52):")
print(f"   - Trend Mean: {trend_mean:.4f}")
print(f"   - Seasonal Std Dev: {seasonal_std:.4f}")

print(f"\n3. ARIMA(2,1,2) Model (Monthly Close):")
print(f"   - AIC: {aic_value:.4f}")

print(f"\n4. Merged Dataset:")
print(f"   - Number of observations: {len(df_merged)}")

print(f"\n5. Granger Causality Test (lag 4):")
print(f"   - F-statistic: {f_statistic_lag4:.4f}")
print(f"   - p-value: {p_value_lag4:.4f}")

print(f"\n6. Bootstrap Correlation (Volume_x vs Volume_y):")
print(f"   - 95% CI Lower Bound: {ci_lower:.4f}")
print(f"   - 95% CI Upper Bound: {ci_upper:.4f}")

print("\n" + "=" * 80)
print("ANALYSIS COMPLETE")
print("=" * 80)
