import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from statsmodels.regression.quantile_regression import QuantReg
from statsmodels.tsa.stattools import grangercausalitytests
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("RETAIL SALES STATISTICAL ANALYSIS")
print("=" * 80)

# Step 1: Load datasets
print("\n[1] Loading datasets...")
order_details = pd.read_csv('Order Details.csv')
list_orders = pd.read_csv('List of Orders.csv')
sales_target = pd.read_csv('Sales target.csv')

print(f"Order Details shape: {order_details.shape}")
print(f"List of Orders shape: {list_orders.shape}")
print(f"Sales Target shape: {sales_target.shape}")

# Step 2: Inner merge between Order-Details and List-of-Orders
print("\n[2] Performing inner merge on Order ID...")
merged_data = pd.merge(order_details, list_orders, on='Order ID', how='inner')
print(f"Merged data shape: {merged_data.shape}")

# Step 3: Parse Order Date and create Month of Order Date
print("\n[3] Creating Month of Order Date column...")
merged_data['Order Date'] = pd.to_datetime(merged_data['Order Date'], format='%d-%m-%Y')
merged_data['Month of Order Date'] = merged_data['Order Date'].dt.strftime('%b-%y')
print(f"Sample Month of Order Date: {merged_data['Month of Order Date'].head(3).tolist()}")

# Step 4: Left merge with Sales-target
print("\n[4] Performing left merge with Sales target...")
final_data = pd.merge(merged_data, sales_target,
                      on=['Month of Order Date', 'Category'],
                      how='left')
print(f"Final merged data shape: {final_data.shape}")

# ============================================================================
# REGRESSION DISCONTINUITY ANALYSIS - ELECTRONICS
# ============================================================================
print("\n" + "=" * 80)
print("REGRESSION DISCONTINUITY ANALYSIS - ELECTRONICS CATEGORY")
print("=" * 80)

electronics_data = final_data[final_data['Category'] == 'Electronics'].copy()
print(f"\n[5] Electronics records: {len(electronics_data)}")

# Subset A: Quantity < 5
subset_a = electronics_data[electronics_data['Quantity'] < 5].copy()
print(f"Subset A (Quantity < 5): {len(subset_a)} records")

# Subset B: Quantity >= 5
subset_b = electronics_data[electronics_data['Quantity'] >= 5].copy()
print(f"Subset B (Quantity >= 5): {len(subset_b)} records")

# Fit polynomial regression degree 2 for Subset A
X_a = subset_a['Quantity'].values
y_a = subset_a['Profit'].values
# Create polynomial features: 1, X, X^2
X_a_poly = np.column_stack([np.ones(len(X_a)), X_a, X_a**2])
# Ordinary Least Squares
beta_a = np.linalg.lstsq(X_a_poly, y_a, rcond=None)[0]
print(f"\nSubset A Model: Profit = {beta_a[0]:.4f} + {beta_a[1]:.4f}*Q + {beta_a[2]:.4f}*Q^2")

# Fit polynomial regression degree 2 for Subset B
X_b = subset_b['Quantity'].values
y_b = subset_b['Profit'].values
X_b_poly = np.column_stack([np.ones(len(X_b)), X_b, X_b**2])
beta_b = np.linalg.lstsq(X_b_poly, y_b, rcond=None)[0]
print(f"Subset B Model: Profit = {beta_b[0]:.4f} + {beta_b[1]:.4f}*Q + {beta_b[2]:.4f}*Q^2")

# Evaluate both models at Quantity = 5
Q_eval = 5
profit_a_at_5 = beta_a[0] + beta_a[1] * Q_eval + beta_a[2] * Q_eval**2
profit_b_at_5 = beta_b[0] + beta_b[1] * Q_eval + beta_b[2] * Q_eval**2

discontinuity_effect = profit_b_at_5 - profit_a_at_5
print(f"\nProfit from Subset A at Q=5: {profit_a_at_5:.4f}")
print(f"Profit from Subset B at Q=5: {profit_b_at_5:.4f}")
print(f"\n>>> DISCONTINUITY EFFECT: {discontinuity_effect:.2f}")

# Calculate R-squared for Subset B model
y_b_pred = X_b_poly @ beta_b
residual_sum_squares = np.sum((y_b - y_b_pred)**2)
total_sum_squares = np.sum((y_b - np.mean(y_b))**2)
r_squared_b = 1 - (residual_sum_squares / total_sum_squares)
print(f">>> SUBSET B R-SQUARED: {r_squared_b:.4f}")

# ============================================================================
# QUANTILE REGRESSION - CLOTHING
# ============================================================================
print("\n" + "=" * 80)
print("QUANTILE REGRESSION - CLOTHING CATEGORY")
print("=" * 80)

clothing_data = final_data[final_data['Category'] == 'Clothing'].copy()
print(f"\n[6] Clothing records: {len(clothing_data)}")

# Remove any missing values
clothing_clean = clothing_data[['Amount', 'Profit']].dropna()
print(f"Clean clothing records: {len(clothing_clean)}")

# Perform quantile regression for 25th, 50th, and 75th percentiles
quantiles = [0.25, 0.50, 0.75]
quantile_results = {}

for q in quantiles:
    model = QuantReg(clothing_clean['Profit'],
                     np.column_stack([np.ones(len(clothing_clean)),
                                     clothing_clean['Amount']]))
    result = model.fit(q=q)
    quantile_results[q] = {
        'intercept': result.params[0],
        'slope': result.params[1]
    }
    print(f"\nQuantile {int(q*100)}%: Profit = {result.params[0]:.4f} + {result.params[1]:.4f} * Amount")

print(f"\n>>> SLOPE AT 75TH PERCENTILE: {quantile_results[0.75]['slope']:.4f}")
print(f">>> INTERCEPT AT 25TH PERCENTILE: {quantile_results[0.25]['intercept']:.2f}")
print(f">>> SLOPE AT 50TH PERCENTILE: {quantile_results[0.50]['slope']:.4f}")

# ============================================================================
# PRINCIPAL COMPONENT ANALYSIS
# ============================================================================
print("\n" + "=" * 80)
print("PRINCIPAL COMPONENT ANALYSIS")
print("=" * 80)

print("\n[7] Performing PCA on Amount, Profit, and Quantity...")
# Extract columns and remove missing values
pca_data = final_data[['Amount', 'Profit', 'Quantity']].dropna()
print(f"Records for PCA: {len(pca_data)}")

# Standardize using z-score normalization (N-1 in denominator)
scaler = StandardScaler()  # Uses N-1 by default
pca_standardized = scaler.fit_transform(pca_data)

print(f"\nMeans after standardization: {pca_standardized.mean(axis=0)}")
print(f"Std devs after standardization: {pca_standardized.std(axis=0, ddof=1)}")

# Perform PCA with 2 components
pca = PCA(n_components=2)
pca_transformed = pca.fit_transform(pca_standardized)

variance_explained = pca.explained_variance_ratio_
cumulative_variance = np.cumsum(variance_explained)

print(f"\nVariance explained by PC1: {variance_explained[0]:.4f}")
print(f"Variance explained by PC2: {variance_explained[1]:.4f}")

print(f"\n>>> PROPORTION OF VARIANCE EXPLAINED BY PC2: {variance_explained[1]:.4f}")
print(f">>> CUMULATIVE PROPORTION (PC1 + PC2): {cumulative_variance[1]:.4f}")

# ============================================================================
# BOOTSTRAP CONFIDENCE INTERVAL
# ============================================================================
print("\n" + "=" * 80)
print("BOOTSTRAP CONFIDENCE INTERVAL FOR PROFIT MEDIAN")
print("=" * 80)

print("\n[8] Performing bootstrap with 1000 iterations...")
profit_data = final_data['Profit'].dropna().values
print(f"Profit records: {len(profit_data)}")

# Set random seed
np.random.seed(42)

# Bootstrap
n_iterations = 1000
bootstrap_medians = []

for i in range(n_iterations):
    # Resample with replacement
    sample = np.random.choice(profit_data, size=len(profit_data), replace=True)
    bootstrap_medians.append(np.median(sample))

bootstrap_medians = np.array(bootstrap_medians)

# 95% confidence interval using percentile method
lower_bound = np.percentile(bootstrap_medians, 2.5)
upper_bound = np.percentile(bootstrap_medians, 97.5)

print(f"Original median: {np.median(profit_data):.2f}")
print(f"\n>>> 95% CI LOWER BOUND: {lower_bound:.2f}")
print(f">>> 95% CI UPPER BOUND: {upper_bound:.2f}")

# ============================================================================
# GRANGER CAUSALITY TEST
# ============================================================================
print("\n" + "=" * 80)
print("GRANGER CAUSALITY TEST")
print("=" * 80)

print("\n[9] Creating monthly time series...")
# Aggregate by Month of Order Date
monthly_data = final_data.groupby('Month of Order Date').agg({
    'Amount': 'sum',
    'Profit': 'sum'
}).reset_index()

# Sort by date
monthly_data['Date'] = pd.to_datetime(monthly_data['Month of Order Date'], format='%b-%y')
monthly_data = monthly_data.sort_values('Date')

# Filter April 2018 to March 2019
start_date = pd.to_datetime('2018-04-01')
end_date = pd.to_datetime('2019-03-31')
monthly_data = monthly_data[(monthly_data['Date'] >= start_date) &
                            (monthly_data['Date'] <= end_date)]

print(f"Monthly records: {len(monthly_data)}")
print(f"\nMonthly data:")
print(monthly_data[['Month of Order Date', 'Amount', 'Profit']].to_string(index=False))

# Prepare data for Granger causality test
# Test if Amount Granger-causes Profit
granger_data = monthly_data[['Amount', 'Profit']].values

print("\n[10] Performing Granger causality test (lag=2)...")
print("Testing: Does Amount Granger-cause Profit?")

# Granger causality test
granger_result = grangercausalitytests(granger_data, maxlag=2, verbose=False)

# Extract F-statistic and p-value for lag 2
lag_2_results = granger_result[2][0]
f_statistic = lag_2_results['ssr_ftest'][0]
p_value = lag_2_results['ssr_ftest'][1]

print(f"\n>>> F-STATISTIC: {f_statistic:.4f}")
print(f">>> P-VALUE: {p_value:.4f}")

# ============================================================================
# HEXBIN PLOT
# ============================================================================
print("\n" + "=" * 80)
print("HEXBIN PLOT VISUALIZATION")
print("=" * 80)

print("\n[11] Creating hexbin plot...")
plot_data = final_data[['Amount', 'Profit']].dropna()

plt.figure(figsize=(10, 6))
plt.hexbin(plot_data['Amount'], plot_data['Profit'],
           gridsize=30, cmap='YlOrRd', mincnt=1)
plt.colorbar(label='Count')
plt.xlabel('Amount', fontsize=12)
plt.ylabel('Profit', fontsize=12)
plt.title('Hexbin Plot: Amount vs Profit', fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('hexbin_amount_vs_profit.png', dpi=300, bbox_inches='tight')
print("Hexbin plot saved as 'hexbin_amount_vs_profit.png'")

# ============================================================================
# SUMMARY REPORT
# ============================================================================
print("\n" + "=" * 80)
print("SUMMARY OF KEY FINDINGS")
print("=" * 80)

print("\n1. REGRESSION DISCONTINUITY (Electronics):")
print(f"   - Discontinuity Effect at Q=5: {discontinuity_effect:.2f}")
print(f"   - Subset B R-squared: {r_squared_b:.4f}")

print("\n2. QUANTILE REGRESSION (Clothing):")
print(f"   - Slope at 75th percentile: {quantile_results[0.75]['slope']:.4f}")
print(f"   - Intercept at 25th percentile: {quantile_results[0.25]['intercept']:.2f}")
print(f"   - Slope at 50th percentile: {quantile_results[0.50]['slope']:.4f}")

print("\n3. PRINCIPAL COMPONENT ANALYSIS:")
print(f"   - Variance explained by PC2: {variance_explained[1]:.4f}")
print(f"   - Cumulative variance (PC1+PC2): {cumulative_variance[1]:.4f}")

print("\n4. BOOTSTRAP CONFIDENCE INTERVAL:")
print(f"   - 95% CI for Profit Median: [{lower_bound:.2f}, {upper_bound:.2f}]")

print("\n5. GRANGER CAUSALITY:")
print(f"   - F-statistic: {f_statistic:.4f}")
print(f"   - P-value: {p_value:.4f}")

print("\n" + "=" * 80)
print("ANALYSIS COMPLETE")
print("=" * 80)
