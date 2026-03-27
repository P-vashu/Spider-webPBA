# German Grid Time Series Analysis Results

## Analysis Overview
This analysis examined electricity transmission data from three German transmission system operators (TSOs): 50Hertz, Amprion, and TenneTTSO. The data covers hourly energy transmission and consumption recorded at 15-minute intervals.

## Data Transformation
- **Date Format**: Converted to datetime using DD/MM/YYYY pattern
- **Structure**: Transformed from wide to long format
- **Time Period**: First 100 dates analyzed (2019-08-23 to 2019-11-30)
- **Observations per Operator**: 9,600 (100 days × 96 fifteen-minute intervals)

---

## Key Findings

### 1. STL Seasonal Decomposition (Seasonal Period = 96)

**50Hertz - Maximum Absolute Seasonal Component**: `41.13`

This represents the maximum deviation from the trend due to seasonal (daily) patterns in the 50Hertz transmission network.

**TenneTTSO - Seasonal Variance Ratio**: `0.94%`

The seasonal component accounts for only 0.94% of the total variance in TenneTTSO's transmission values, indicating that seasonal patterns contribute minimally to overall variability compared to trend and residual components.

---

### 2. Augmented Dickey-Fuller Test (Amprion)

**p-value**: `0.0000`

The extremely low p-value (< 0.05) strongly rejects the null hypothesis of a unit root, indicating that the Amprion Value series is **stationary**. This suggests the time series does not exhibit trending behavior and has constant statistical properties over time.

---

### 3. Global Peak Load

**Value**: `665.00` MW
**Operator**: 50Hertz
**DateTime**: 2019-09-30 08:45:00

The highest electricity transmission value across all three operators during the first 100 dates occurred in the 50Hertz network on September 30, 2019, at 8:45 AM, likely corresponding to morning peak demand.

---

### 4. Granger Causality Test

**Test**: Does 50Hertz Granger-cause TenneTTSO?
**Lag**: 4 (15-minute intervals = 1 hour)
**p-value**: `0.0000`

The highly significant p-value (< 0.05) indicates that **50Hertz does Granger-cause TenneTTSO** at lag 4. This means past values of 50Hertz transmission are statistically significant predictors of current TenneTTSO values, suggesting interdependence between these transmission regions with approximately a 1-hour lead time.

---

### 5. Cross-Correlation Analysis (50Hertz vs Amprion)

**Lag with Maximum Absolute Cross-Correlation**: `23` (15-minute intervals)
**Cross-Correlation Coefficient**: `0.7362`

The strongest correlation between 50Hertz and Amprion occurs when 50Hertz leads by 23 fifteen-minute intervals (approximately 5 hours and 45 minutes). The correlation coefficient of 0.7362 indicates a strong positive relationship, suggesting that transmission patterns in 50Hertz are followed by similar patterns in Amprion with this time delay.

---

### 6. KPSS Stationarity Test (TenneTTSO STL Residuals)

**KPSS Statistic**: `0.0025`

The very low KPSS statistic (far below critical values of ~0.35-0.74) fails to reject the null hypothesis of stationarity. This confirms that after removing trend and seasonal components via STL decomposition, the **residuals are stationary**, indicating the decomposition successfully captured non-stationary patterns.

---

### 7. Auto ARIMA Forecasting (50Hertz)

**Model**: ARIMA(2, 0, 2)
**Training Period**: Dates 1-80
**Test Period**: Dates 81-100
**MAPE (excluding zeros)**: `240.27%`

The Auto ARIMA model selected an ARIMA(2,0,2) specification, indicating:
- AR(2): Two autoregressive terms
- I(0): No differencing needed (stationary series)
- MA(2): Two moving average terms

The high MAPE of 240.27% indicates poor forecasting performance. This is likely due to:
- High volatility in electricity transmission data
- Complex seasonal patterns not fully captured by the ARIMA model
- External factors (weather, economic activity) not included in the univariate model

**Note**: A more sophisticated approach using SARIMAX, external regressors, or machine learning methods would likely improve forecast accuracy.

---

### 8. Hexbin Visualization

**File**: `hexbin_plot.png`
**Description**: Hexagonal binning plot showing the relationship between 50Hertz and TenneTTSO transmission values

The hexbin plot visualizes the joint distribution of 50Hertz and TenneTTSO values using hexagonal bins colored by observation count. This reveals concentration patterns and correlation structure between the two transmission networks.

---

## Summary Statistics

| Metric | Value |
|--------|-------|
| Total Observations | 114,336 (across 3 operators) |
| Date Range | 2019-08-23 to 2020-09-22 |
| Analysis Period | First 100 dates |
| Time Resolution | 15-minute intervals |
| Seasonal Period | 96 (1 day) |
| Value Range | 0.00 - 727.44 MW |

---

## Technical Implementation

### Libraries Used
- `pandas`: Data manipulation and transformation
- `numpy`: Numerical computations
- `statsmodels`: Time series analysis (STL, ADF, KPSS, Granger, ARIMA)
- `matplotlib`: Visualization
- `scipy`: Statistical functions

### Key Methodologies
1. **STL Decomposition**: Seasonal and Trend decomposition using Loess
2. **Stationarity Tests**: ADF (unit root) and KPSS (trend stationarity)
3. **Causality Analysis**: Granger causality testing
4. **Cross-Correlation**: Time-lagged correlation analysis
5. **Forecasting**: ARIMA modeling with grid search parameter optimization

---

## Analysis Date
Generated: 2025-11-12

## Files
- `grid_analysis.py`: Complete analysis script
- `hexbin_plot.png`: Visualization output
- `ANALYSIS_RESULTS.md`: This summary document
