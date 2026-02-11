# Cryptocurrency Market Dynamics Analysis

## Executive Summary

This report presents a comprehensive quantitative analysis of cryptocurrency market dynamics using Bitcoin (BTC), Binance Coin (BNB), and Aave (AAVE) datasets from October 5, 2020, to July 6, 2021.

## Data Processing

- **Date Range**: October 5, 2020 - July 6, 2021
- **Records per Dataset**: 273 (after removing first row with undefined Daily_Return)
- **Daily Return Calculation**: (Close - Previous Close) / Previous Close × 100

## Analysis Results

### 1. Seasonal Decomposition (Bitcoin Close Prices)

**Method**: Multiplicative model with period=30, classical decomposition with extrapolate_trend='freq'

**Results**:
- **Mean of Trend Component**: 36378.1594
- **Standard Deviation of Residual Component**: 0.0640

**Interpretation**: The trend component shows Bitcoin's average price around $36,378 during the period. The very low residual standard deviation (0.0640) indicates strong seasonal patterns with minimal unexplained variation.

### 2. Principal Component Analysis

**Method**: Z-score normalization applied to High, Low, Open, Close, and Volume columns across all three concatenated datasets (819 total records)

**Results**:
- **Explained Variance Ratio (PC1)**: 0.9535
- **Cumulative Explained Variance (PC1-PC3)**: 0.9998

**Interpretation**: The first principal component captures 95.35% of the total variance, indicating extremely high correlation among the normalized price and volume features. The first three components explain virtually all variance (99.98%), suggesting the data lies in a low-dimensional manifold.

### 3. Granger Causality Test (Bitcoin → BinanceCoin)

**Method**: SSR F-test with lag=5, testing whether Bitcoin Daily_Return Granger-causes BinanceCoin Daily_Return

**Results**:
- **F-statistic (lag 5)**: 1.501
- **p-value**: 0.189942

**Interpretation**: At conventional significance levels (α = 0.05), we fail to reject the null hypothesis. This suggests that Bitcoin's past returns do not have statistically significant predictive power for BinanceCoin's current returns when testing lags 1-5 jointly, indicating no clear lead-lag relationship in the specified period.

### 4. Generalized Pareto Distribution (Bitcoin Volume)

**Method**: GPD fitted to exceedances above 95th percentile using Maximum Likelihood Estimation with location parameter fixed at 0

**Results**:
- **95th Percentile Threshold**: 85,524,751,547.05
- **Number of Exceedances**: 14
- **Shape Parameter (ξ)**: 0.6827
- **Scale Parameter (σ)**: 12,232,871,190.0979

**Interpretation**: The positive shape parameter (ξ = 0.6827) indicates a heavy-tailed distribution, characteristic of extreme trading volumes. This suggests that extreme volume events have fatter tails than exponential distribution, with substantial probability mass in the tail region - typical of financial market data with occasional large trading spikes.

### 5. Quantile Regression (BinanceCoin)

**Method**: Quantile regression with Volume as independent variable, Close as dependent variable

**Results**:
- **Slope Coefficient at τ=0.25**: 0.000000
- **Slope Coefficient at τ=0.50**: 0.00000
- **Slope Coefficient at τ=0.75**: 0.000000

**Interpretation**: The extremely small coefficients (< 0.000005) reflect the scale disparity between Volume (hundreds of thousands to millions) and Close price (< $1 in early period). While statistically small, this suggests minimal direct linear relationship between trading volume and price at different quantile levels in the BinanceCoin market during this period.

### 6. Spectral Analysis (Aave Close Prices)

**Method**: Periodogram computed from FFT of demeaned series, analyzing positive frequencies up to Nyquist frequency (0.5 cycles/day)

**Results**:
- **Frequency with Highest Power Spectral Density**: 0.003663 cycles/day
- **Periodicity**: 273.00 days

**Interpretation**: The dominant frequency corresponds to a periodicity of 273 days, which essentially spans the entire analysis period (273 data points). This suggests the most significant cyclical pattern in Aave prices occurs at the scale of the full dataset, possibly reflecting the overall bull/bear cycle of the crypto market during this timeframe rather than a shorter-term trading cycle.

### 7. Cross-Asset Dependency Visualization

**Method**: Hexbin plot of Bitcoin vs BinanceCoin Daily Returns (inner join on Date)

**Output**: `hexbin_plot.png`

**Interpretation**: The hexbin plot visualizes the joint distribution of daily returns between Bitcoin and BinanceCoin, revealing concentration patterns and potential correlation structure in the return dynamics.

## Key Findings

1. **Strong Feature Correlation**: Over 95% of variance explained by first principal component indicates cryptocurrencies move together
2. **Heavy-Tailed Volume Distribution**: GPD shape parameter of 0.68 confirms extreme volume events are more likely than normal distribution predicts
3. **No Clear Lead-Lag**: Granger causality test shows Bitcoin doesn't significantly predict BinanceCoin at 5-day lag
4. **Low Residual Variation**: Seasonal decomposition shows strong predictable patterns in Bitcoin prices
5. **Macro-Scale Cycles**: Spectral analysis identifies period-spanning cycles in Aave, suggesting market-wide trends dominate

## Technical Implementation

All analyses were implemented in Python using:
- `pandas` for data manipulation
- `statsmodels` for time series decomposition, Granger causality, and quantile regression
- `scipy` for FFT, statistical distributions, and GPD fitting
- `scikit-learn` for PCA and standardization
- `matplotlib` for hexbin visualization

## Files Generated

- `crypto_analysis.py` - Complete analysis script
- `hexbin_plot.png` - Bitcoin vs BinanceCoin daily returns visualization
- `ANALYSIS_RESULTS.md` - This comprehensive report

## Conclusion

The analysis reveals strong interdependencies across cryptocurrency markets with pronounced heavy-tailed behavior in extreme events. The combination of seasonal patterns, extreme value characteristics, and high-dimensional correlation provides insights into the complex dynamics of digital asset markets during the analyzed period.
