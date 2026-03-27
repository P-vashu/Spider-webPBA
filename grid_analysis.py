import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.stattools import adfuller, grangercausalitytests, kpss
from statsmodels.tsa.arima.model import ARIMA
from itertools import product
import warnings
warnings.filterwarnings('ignore')

# ===============================
# 1. Load and Transform Data
# ===============================

def load_and_transform(filename, operator_name):
    """Load CSV and transform to long format"""
    print(f"\nLoading {filename}...")

    # Read CSV file
    df = pd.read_csv(filename, encoding='utf-8-sig')

    # Convert Date column to datetime with DD/MM/YYYY format
    df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%Y')

    # Get time columns (all except Date)
    time_cols = [col for col in df.columns if col != 'Date']

    # Melt to long format
    df_long = df.melt(id_vars=['Date'], value_vars=time_cols,
                      var_name='TimeOfDay', value_name='Value')

    # Combine Date and TimeOfDay into DateTime
    # TimeOfDay is in format HH:MM:SS, Date is DD/MM/YYYY
    df_long['DateTime'] = pd.to_datetime(
        df_long['Date'].dt.strftime('%d/%m/%Y') + ' ' + df_long['TimeOfDay'],
        format='%d/%m/%Y %H:%M:%S'
    )

    # Add operator column
    df_long['Operator'] = operator_name

    # Keep only DateTime, Operator, Value
    df_long = df_long[['DateTime', 'Operator', 'Value']].copy()

    # Sort by DateTime
    df_long = df_long.sort_values('DateTime').reset_index(drop=True)

    print(f"  Shape: {df_long.shape}")
    print(f"  Date range: {df_long['DateTime'].min()} to {df_long['DateTime'].max()}")

    return df_long

# Load all three datasets
df_50hertz = load_and_transform('50Hertz.csv', '50Hertz')
df_amprion = load_and_transform('Amprion.csv', 'Amprion')
df_tennet = load_and_transform('TenneTTSO.csv', 'TenneTTSO')

# Merge all datasets
df_combined = pd.concat([df_50hertz, df_amprion, df_tennet], ignore_index=True)
df_combined = df_combined.sort_values(['DateTime', 'Operator']).reset_index(drop=True)

print(f"\nCombined dataset shape: {df_combined.shape}")
print(f"Operators: {df_combined['Operator'].unique()}")
print(f"Value range: {df_combined['Value'].min()} to {df_combined['Value'].max()}")

# ===============================
# 2. Extract First 100 Dates for Analysis
# ===============================

# Get unique dates
unique_dates = sorted(df_combined['DateTime'].dt.date.unique())
first_100_dates = unique_dates[:100]

print(f"\nFirst 100 dates: {first_100_dates[0]} to {first_100_dates[99]}")

# Filter data for first 100 dates
df_100 = df_combined[df_combined['DateTime'].dt.date.isin(first_100_dates)].copy()

# Create separate dataframes for each operator (first 100 dates)
df_50hertz_100 = df_100[df_100['Operator'] == '50Hertz'].sort_values('DateTime').reset_index(drop=True)
df_amprion_100 = df_100[df_100['Operator'] == 'Amprion'].sort_values('DateTime').reset_index(drop=True)
df_tennet_100 = df_100[df_100['Operator'] == 'TenneTTSO'].sort_values('DateTime').reset_index(drop=True)

print(f"\n50Hertz (100 dates): {len(df_50hertz_100)} observations")
print(f"Amprion (100 dates): {len(df_amprion_100)} observations")
print(f"TenneTTSO (100 dates): {len(df_tennet_100)} observations")

# ===============================
# 3. STL Decomposition
# ===============================

print("\n" + "="*60)
print("STL DECOMPOSITION (Seasonal Period = 96, First 100 Dates)")
print("="*60)

# STL for 50Hertz
stl_50hertz = STL(df_50hertz_100['Value'], seasonal=97, period=96).fit()
seasonal_50hertz = stl_50hertz.seasonal
max_abs_seasonal_50hertz = np.abs(seasonal_50hertz).max()

print(f"\n50Hertz - Maximum Absolute Seasonal Component: {max_abs_seasonal_50hertz:.2f}")

# STL for TenneTTSO
stl_tennet = STL(df_tennet_100['Value'], seasonal=97, period=96).fit()
seasonal_tennet = stl_tennet.seasonal
residual_tennet = stl_tennet.resid

# Calculate variance ratio for TenneTTSO
var_seasonal = np.var(seasonal_tennet)
var_original = np.var(df_tennet_100['Value'])
variance_ratio_pct = (var_seasonal / var_original) * 100

print(f"\nTenneTTSO - Seasonal Variance Ratio: {variance_ratio_pct:.2f}%")

# STL for Amprion (we'll need this for later)
stl_amprion = STL(df_amprion_100['Value'], seasonal=97, period=96).fit()

# ===============================
# 4. Augmented Dickey-Fuller Test (Amprion)
# ===============================

print("\n" + "="*60)
print("AUGMENTED DICKEY-FULLER TEST (Amprion, First 100 Dates)")
print("="*60)

adf_result = adfuller(df_amprion_100['Value'])
adf_pvalue = adf_result[1]

print(f"\nADF Test p-value: {adf_pvalue:.4f}")

# ===============================
# 5. Global Peak Load
# ===============================

print("\n" + "="*60)
print("GLOBAL PEAK LOAD (First 100 Dates)")
print("="*60)

global_peak_value = df_100['Value'].max()
peak_row = df_100[df_100['Value'] == global_peak_value].iloc[0]

print(f"\nGlobal Peak Load Value: {global_peak_value:.2f}")
print(f"Operator: {peak_row['Operator']}")
print(f"DateTime: {peak_row['DateTime']}")

# ===============================
# 6. Granger Causality Test
# ===============================

print("\n" + "="*60)
print("GRANGER CAUSALITY TEST (50Hertz -> TenneTTSO, Lag=4)")
print("="*60)

# Create dataframe with both series
granger_df = pd.DataFrame({
    '50Hertz': df_50hertz_100['Value'].values,
    'TenneTTSO': df_tennet_100['Value'].values
})

# Perform Granger causality test
# Test if 50Hertz Granger-causes TenneTTSO
granger_result = grangercausalitytests(granger_df[['TenneTTSO', '50Hertz']], maxlag=4, verbose=False)

# Extract p-value for lag 4
granger_pvalue = granger_result[4][0]['ssr_ftest'][1]

print(f"\nGranger Causality p-value (lag=4): {granger_pvalue:.4f}")

# ===============================
# 7. Cross-Correlation
# ===============================

print("\n" + "="*60)
print("CROSS-CORRELATION (50Hertz vs Amprion, Lags -48 to +48)")
print("="*60)

# Get the value series
series_50hertz = df_50hertz_100['Value'].values
series_amprion = df_amprion_100['Value'].values

# Calculate cross-correlation for lags -48 to +48
lags = np.arange(-48, 49)
cross_corr = []

for lag in lags:
    if lag < 0:
        # Negative lag: shift series_50hertz backward (or series_amprion forward)
        corr = np.corrcoef(series_50hertz[:lag], series_amprion[-lag:])[0, 1]
    elif lag > 0:
        # Positive lag: shift series_50hertz forward (or series_amprion backward)
        corr = np.corrcoef(series_50hertz[lag:], series_amprion[:-lag])[0, 1]
    else:
        # Zero lag
        corr = np.corrcoef(series_50hertz, series_amprion)[0, 1]
    cross_corr.append(corr)

cross_corr = np.array(cross_corr)

# Find lag with maximum absolute cross-correlation
max_corr_idx = np.argmax(np.abs(cross_corr))
max_corr_lag = lags[max_corr_idx]
max_corr_value = cross_corr[max_corr_idx]

print(f"\nLag with Maximum Absolute Cross-Correlation: {max_corr_lag} (15-minute intervals)")
print(f"Cross-Correlation Coefficient at this lag: {max_corr_value:.4f}")

# ===============================
# 8. KPSS Test on TenneTTSO Residuals
# ===============================

print("\n" + "="*60)
print("KPSS STATIONARITY TEST (TenneTTSO STL Residuals)")
print("="*60)

# Perform KPSS test with constant regression (trend='c')
kpss_result = kpss(residual_tennet, regression='c', nlags='auto')
kpss_statistic = kpss_result[0]

print(f"\nKPSS Statistic: {kpss_statistic:.4f}")

# ===============================
# 9. Auto ARIMA and MAPE
# ===============================

print("\n" + "="*60)
print("AUTO ARIMA (50Hertz, Train: Dates 1-80, Test: Dates 81-100)")
print("="*60)

# Get first 80 and 81-100 dates
first_80_dates = unique_dates[:80]
dates_81_to_100 = unique_dates[80:100]

# Filter data
df_50hertz_train = df_50hertz[df_50hertz['DateTime'].dt.date.isin(first_80_dates)].copy()
df_50hertz_test = df_50hertz[df_50hertz['DateTime'].dt.date.isin(dates_81_to_100)].copy()

train_values = df_50hertz_train['Value'].values
test_values = df_50hertz_test['Value'].values

print(f"\nTraining observations: {len(train_values)}")
print(f"Test observations: {len(test_values)}")

# Fit Auto ARIMA using grid search
print("\nFitting Auto ARIMA model...")

# Define parameter ranges for grid search
p_range = range(0, 3)
d_range = range(0, 2)
q_range = range(0, 3)

best_aic = np.inf
best_order = None
best_model = None

# Grid search
for p, d, q in product(p_range, d_range, q_range):
    try:
        model = ARIMA(train_values, order=(p, d, q))
        fitted = model.fit()
        if fitted.aic < best_aic:
            best_aic = fitted.aic
            best_order = (p, d, q)
            best_model = fitted
    except:
        continue

print(f"Best model: ARIMA{best_order}, AIC: {best_aic:.2f}")

# Forecast for dates 81-100
forecast = best_model.forecast(steps=len(test_values))

# Calculate MAPE excluding zeros
non_zero_mask = test_values != 0
test_values_nonzero = test_values[non_zero_mask]
forecast_nonzero = forecast[non_zero_mask]

if len(test_values_nonzero) > 0:
    mape = np.mean(np.abs((test_values_nonzero - forecast_nonzero) / test_values_nonzero)) * 100
    print(f"\nMAPE (excluding zeros): {mape:.2f}%")
else:
    print("\nNo non-zero values in test set")

# ===============================
# 10. Hexbin Plot
# ===============================

print("\n" + "="*60)
print("GENERATING HEXBIN PLOT")
print("="*60)

# Get 50Hertz and TenneTTSO values for first 100 dates
x_values = df_50hertz_100['Value'].values
y_values = df_tennet_100['Value'].values

# Create hexbin plot
plt.figure(figsize=(10, 8))
hexbin = plt.hexbin(x_values, y_values, gridsize=30, cmap='viridis', mincnt=1)
plt.colorbar(hexbin, label='Observation Count')
plt.xlabel('50Hertz Value', fontsize=12)
plt.ylabel('TenneTTSO Value', fontsize=12)
plt.title('Hexbin Plot: 50Hertz vs TenneTTSO (First 100 Dates)', fontsize=14)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('hexbin_plot.png', dpi=300, bbox_inches='tight')
print("\nHexbin plot saved as 'hexbin_plot.png'")

# ===============================
# SUMMARY OF RESULTS
# ===============================

print("\n" + "="*60)
print("SUMMARY OF RESULTS")
print("="*60)

print(f"""
1. STL Decomposition (Seasonal Period=96, First 100 Dates):
   - 50Hertz Max Absolute Seasonal Component: {max_abs_seasonal_50hertz:.2f}
   - TenneTTSO Seasonal Variance Ratio: {variance_ratio_pct:.2f}%

2. Augmented Dickey-Fuller Test (Amprion, First 100 Dates):
   - p-value: {adf_pvalue:.4f}

3. Global Peak Load (First 100 Dates):
   - Value: {global_peak_value:.2f}
   - Operator: {peak_row['Operator']}
   - DateTime: {peak_row['DateTime']}

4. Granger Causality Test (50Hertz -> TenneTTSO, Lag=4):
   - p-value: {granger_pvalue:.4f}

5. Cross-Correlation (50Hertz vs Amprion, Lags -48 to +48):
   - Lag with Max Absolute Cross-Correlation: {max_corr_lag}
   - Cross-Correlation Coefficient: {max_corr_value:.4f}

6. KPSS Test (TenneTTSO STL Residuals):
   - KPSS Statistic: {kpss_statistic:.4f}

7. Auto ARIMA (50Hertz, Train: 1-80, Test: 81-100):
   - MAPE (excluding zeros): {mape:.2f}%

8. Hexbin Plot:
   - Saved as 'hexbin_plot.png'
""")

print("="*60)
print("ANALYSIS COMPLETE")
print("="*60)
