import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from statsmodels.tsa.stattools import grangercausalitytests
from scipy import stats
from scipy.fft import fft
import statsmodels.api as sm
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("CRYPTOCURRENCY MARKET DYNAMICS ANALYSIS")
print("="*80)

# Step 1: Load and preprocess data
print("\n[1] LOADING AND PREPROCESSING DATA")
print("-" * 80)

# Load datasets
bitcoin = pd.read_csv('coin_Bitcoin.csv')
binance = pd.read_csv('coin_BinanceCoin.csv')
aave = pd.read_csv('coin_Aave.csv')

# Convert Date columns to datetime
bitcoin['Date'] = pd.to_datetime(bitcoin['Date'])
binance['Date'] = pd.to_datetime(binance['Date'])
aave['Date'] = pd.to_datetime(aave['Date'])

# Filter for October 5, 2020 to July 6, 2021
start_date = pd.to_datetime('2020-10-05')
end_date = pd.to_datetime('2021-07-06')

bitcoin_filtered = bitcoin[(bitcoin['Date'] >= start_date) & (bitcoin['Date'] <= end_date)].copy()
binance_filtered = binance[(binance['Date'] >= start_date) & (binance['Date'] <= end_date)].copy()
aave_filtered = aave[(aave['Date'] >= start_date) & (aave['Date'] <= end_date)].copy()

print(f"Bitcoin records: {len(bitcoin_filtered)}")
print(f"BinanceCoin records: {len(binance_filtered)}")
print(f"Aave records: {len(aave_filtered)}")

# Create Daily_Return
bitcoin_filtered['Daily_Return'] = (bitcoin_filtered['Close'] - bitcoin_filtered['Close'].shift(1)) / bitcoin_filtered['Close'].shift(1) * 100
binance_filtered['Daily_Return'] = (binance_filtered['Close'] - binance_filtered['Close'].shift(1)) / binance_filtered['Close'].shift(1) * 100
aave_filtered['Daily_Return'] = (aave_filtered['Close'] - aave_filtered['Close'].shift(1)) / aave_filtered['Close'].shift(1) * 100

# Remove first row where Daily_Return is undefined
bitcoin_filtered = bitcoin_filtered.iloc[1:].reset_index(drop=True)
binance_filtered = binance_filtered.iloc[1:].reset_index(drop=True)
aave_filtered = aave_filtered.iloc[1:].reset_index(drop=True)

print(f"\nAfter removing first row:")
print(f"Bitcoin records: {len(bitcoin_filtered)}")
print(f"BinanceCoin records: {len(binance_filtered)}")
print(f"Aave records: {len(aave_filtered)}")

# Step 2: Seasonal Decomposition
print("\n[2] SEASONAL DECOMPOSITION - BITCOIN CLOSE PRICES")
print("-" * 80)

# Apply seasonal decomposition
decomposition = seasonal_decompose(
    bitcoin_filtered['Close'],
    model='multiplicative',
    period=30,
    extrapolate_trend='freq'
)

mean_trend = np.mean(decomposition.trend)
std_residual = np.std(decomposition.resid)

print(f"Mean of trend component: {mean_trend:.4f}")
print(f"Standard deviation of residual component: {std_residual:.4f}")

# Step 3: PCA Analysis
print("\n[3] PRINCIPAL COMPONENT ANALYSIS")
print("-" * 80)

# Add dataset identifier
bitcoin_pca = bitcoin_filtered.copy()
binance_pca = binance_filtered.copy()
aave_pca = aave_filtered.copy()

bitcoin_pca['Dataset'] = 'Bitcoin'
binance_pca['Dataset'] = 'BinanceCoin'
aave_pca['Dataset'] = 'Aave'

# Vertically concatenate
combined = pd.concat([bitcoin_pca, binance_pca, aave_pca], axis=0, ignore_index=True)

print(f"Combined dataset size: {len(combined)}")

# Z-score normalization
columns_to_normalize = ['High', 'Low', 'Open', 'Close', 'Volume']
scaler = StandardScaler()
combined[columns_to_normalize] = scaler.fit_transform(combined[columns_to_normalize])

# Perform PCA
pca = PCA()
pca_features = pca.fit_transform(combined[columns_to_normalize])

explained_variance_pc1 = pca.explained_variance_ratio_[0]
cumulative_variance_pc3 = np.sum(pca.explained_variance_ratio_[:3])

print(f"Explained variance ratio of PC1: {explained_variance_pc1:.4f}")
print(f"Cumulative explained variance (PC1-PC3): {cumulative_variance_pc3:.4f}")

# Step 4: Granger Causality Test
print("\n[4] GRANGER CAUSALITY TEST - BITCOIN vs BINANCECOIN")
print("-" * 80)

# Inner join on Date
merged = pd.merge(
    bitcoin_filtered[['Date', 'Daily_Return']],
    binance_filtered[['Date', 'Daily_Return']],
    on='Date',
    how='inner',
    suffixes=('_BTC', '_BNB')
)

print(f"Merged records: {len(merged)}")

# Prepare data for Granger causality test
data = merged[['Daily_Return_BTC', 'Daily_Return_BNB']].dropna()

# Conduct Granger causality test
# Testing if Bitcoin Granger-causes BinanceCoin
gc_result = grangercausalitytests(data[['Daily_Return_BNB', 'Daily_Return_BTC']], maxlag=5, verbose=False)

# Extract F-statistic and p-value for lag 5
f_stat = gc_result[5][0]['ssr_ftest'][0]
p_value = gc_result[5][0]['ssr_ftest'][1]

print(f"F-statistic (lag 5): {f_stat:.3f}")
print(f"p-value: {p_value:.6f}")

# Step 5: Generalized Pareto Distribution
print("\n[5] GENERALIZED PARETO DISTRIBUTION - BITCOIN VOLUME")
print("-" * 80)

# Calculate 95th percentile
threshold = np.percentile(bitcoin_filtered['Volume'], 95)
print(f"95th percentile threshold: {threshold:.2f}")

# Get exceedances
exceedances = bitcoin_filtered['Volume'][bitcoin_filtered['Volume'] > threshold] - threshold

print(f"Number of exceedances: {len(exceedances)}")

# Fit GPD with location=0
shape, loc, scale = stats.genpareto.fit(exceedances, floc=0)

print(f"Shape parameter: {shape:.4f}")
print(f"Scale parameter: {scale:.4f}")

# Step 6: Quantile Regression
print("\n[6] QUANTILE REGRESSION - BINANCECOIN")
print("-" * 80)

# Prepare data
X = sm.add_constant(binance_filtered['Volume'])
y = binance_filtered['Close']

# Fit quantile regression at different quantiles
quantiles = [0.25, 0.50, 0.75]
results = {}

for q in quantiles:
    model = sm.QuantReg(y, X)
    result = model.fit(q=q)
    slope = result.params['Volume']
    results[q] = slope
    print(f"Quantile {q}: Slope = {slope:.6f}")

print(f"\nSlope coefficient at 0.50 quantile: {results[0.50]:.5f}")

# Step 7: Spectral Analysis
print("\n[7] SPECTRAL ANALYSIS - AAVE CLOSE PRICES")
print("-" * 80)

# Subtract mean
close_series = aave_filtered['Close'].values
close_demeaned = close_series - np.mean(close_series)
N = len(close_demeaned)

print(f"Number of data points: {N}")

# Compute FFT
fft_result = fft(close_demeaned)

# Compute periodogram
periodogram = (1/N) * np.abs(fft_result)**2

# Get frequencies (excluding DC component)
freqs = np.fft.fftfreq(N, d=1)  # d=1 for daily data

# Consider only positive frequencies up to Nyquist frequency (0.5)
positive_freq_mask = (freqs > 0) & (freqs <= 0.5)
positive_freqs = freqs[positive_freq_mask]
positive_periodogram = periodogram[positive_freq_mask]

# Find frequency with highest PSD
max_idx = np.argmax(positive_periodogram)
max_freq = positive_freqs[max_idx]
max_psd = positive_periodogram[max_idx]

# Calculate periodicity in days
periodicity = 1 / max_freq if max_freq > 0 else np.inf

print(f"Frequency with highest PSD: {max_freq:.6f} cycles/day")
print(f"Periodicity: {periodicity:.2f} days")

# Step 8: Hexbin Plot
print("\n[8] GENERATING HEXBIN PLOT")
print("-" * 80)

# Use already merged data from Granger causality test
plt.figure(figsize=(10, 8))
plt.hexbin(merged['Daily_Return_BTC'], merged['Daily_Return_BNB'],
           gridsize=30, cmap='YlOrRd', mincnt=1)
plt.colorbar(label='Count')
plt.xlabel('Bitcoin Daily Return (%)')
plt.ylabel('BinanceCoin Daily Return (%)')
plt.title('Hexbin Plot: Bitcoin vs BinanceCoin Daily Returns')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('hexbin_plot.png', dpi=300, bbox_inches='tight')
print("Hexbin plot saved as 'hexbin_plot.png'")

print("\n" + "="*80)
print("ANALYSIS COMPLETE")
print("="*80)

# Summary Report
print("\n" + "="*80)
print("SUMMARY OF RESULTS")
print("="*80)

print("\n[SEASONAL DECOMPOSITION - Bitcoin]")
print(f"  Mean of trend component: {mean_trend:.4f}")
print(f"  Std dev of residual component: {std_residual:.4f}")

print("\n[PRINCIPAL COMPONENT ANALYSIS]")
print(f"  Explained variance ratio (PC1): {explained_variance_pc1:.4f}")
print(f"  Cumulative explained variance (PC1-PC3): {cumulative_variance_pc3:.4f}")

print("\n[GRANGER CAUSALITY TEST]")
print(f"  F-statistic (lag 5): {f_stat:.3f}")
print(f"  p-value: {p_value:.6f}")

print("\n[GENERALIZED PARETO DISTRIBUTION - Bitcoin Volume]")
print(f"  Shape parameter: {shape:.4f}")
print(f"  Scale parameter: {scale:.4f}")

print("\n[QUANTILE REGRESSION - BinanceCoin]")
print(f"  Slope coefficient at 0.50 quantile: {results[0.50]:.5f}")

print("\n[SPECTRAL ANALYSIS - Aave]")
print(f"  Frequency with highest PSD: {max_freq:.6f} cycles/day")
print(f"  Periodicity: {periodicity:.2f} days")

print("\n" + "="*80)
