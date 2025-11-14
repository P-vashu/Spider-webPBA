import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import kstest, ks_2samp
import statsmodels.formula.api as smf
import pymc as pm
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("BANK MARKETING CAUSAL INFERENCE ANALYSIS")
print("=" * 80)

# ============================================================================
# STEP 1: DATA LOADING AND PREPROCESSING
# ============================================================================
print("\n[1] Loading data...")
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

print(f"Train shape: {train.shape}")
print(f"Test shape: {test.shape}")

# ============================================================================
# STEP 2: REPLACE NEGATIVE BALANCE WITH MEDIAN
# ============================================================================
print("\n[2] Replacing negative balance values...")

def replace_negative_balance(df):
    """Replace negative balance with median of non-negative balance"""
    df = df.copy()

    # Calculate overall non-negative median as fallback
    overall_median = df[df['balance'] >= 0]['balance'].median()

    # Find rows with negative balance
    negative_mask = df['balance'] < 0

    for idx in df[negative_mask].index:
        job_val = df.loc[idx, 'job']
        marital_val = df.loc[idx, 'marital']

        # Get non-negative balance for same job and marital
        mask = (df['job'] == job_val) & (df['marital'] == marital_val) & (df['balance'] >= 0)
        group_median = df.loc[mask, 'balance'].median()

        # Use group median if available, otherwise use overall median
        if pd.notna(group_median):
            df.loc[idx, 'balance'] = group_median
        else:
            df.loc[idx, 'balance'] = overall_median

    return df

train = replace_negative_balance(train)
test = replace_negative_balance(test)

print(f"Negative balance in train after replacement: {(train['balance'] < 0).sum()}")
print(f"Negative balance in test after replacement: {(test['balance'] < 0).sum()}")

# ============================================================================
# STEP 3: ENCODE CATEGORICAL VARIABLES
# ============================================================================
print("\n[3] Encoding categorical variables...")

# Define categorical columns to encode alphabetically
cat_cols = ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'poutcome']

# Create encoding dictionaries by sorting unique values alphabetically
encoding_dicts = {}
for col in cat_cols:
    unique_vals = sorted(pd.concat([train[col], test[col]]).unique())
    encoding_dicts[col] = {val: idx for idx, val in enumerate(unique_vals)}

# Apply encoding to train and test
for col in cat_cols:
    train[col] = train[col].map(encoding_dicts[col])
    test[col] = test[col].map(encoding_dicts[col])

# Encode y as 0/1 (no=0, yes=1)
train['y'] = train['y'].map({'no': 0, 'yes': 1})

print("Categorical encoding completed")

# ============================================================================
# STEP 4: MERGE TRAIN AND TEST WITH SOURCE INDICATOR
# ============================================================================
print("\n[4] Merging train and test datasets...")

# Add source column (1 for train, 0 for test)
train['source'] = 1
test['source'] = 0
test['y'] = np.nan  # Test has null y

# Merge vertically
merged = pd.concat([train, test], axis=0, ignore_index=True)

print(f"Merged shape: {merged.shape}")
print(f"Source distribution:\n{merged['source'].value_counts()}")

# ============================================================================
# STEP 5: PROPENSITY SCORE MATCHING
# ============================================================================
print("\n[5] Fitting logistic regression for propensity scores...")

# Filter to source == 1 (train data only)
train_data = merged[merged['source'] == 1].copy()

# Fit logistic regression: y ~ age + balance + campaign + previous
X_prop = train_data[['age', 'balance', 'campaign', 'previous']]
y_prop = train_data['y']

# Use L-BFGS solver
lr = LogisticRegression(solver='lbfgs', max_iter=1000, random_state=42)
lr.fit(X_prop, y_prop)

# Calculate propensity scores
train_data['propensity'] = lr.predict_proba(X_prop)[:, 1]

# Calculate caliper as 0.05 * std of propensity scores
caliper = 0.05 * train_data['propensity'].std()
print(f"Caliper: {caliper:.6f}")

# ============================================================================
# STEP 6: PERFORM 1-TO-1 NEAREST NEIGHBOR MATCHING
# ============================================================================
print("\n[6] Performing propensity score matching...")

# Split into treatment (y=1) and control (y=0)
treated = train_data[train_data['y'] == 1].copy().sort_values('ID')
control = train_data[train_data['y'] == 0].copy().sort_values('ID')

print(f"Treated units: {len(treated)}")
print(f"Control units: {len(control)}")

# Perform matching
matched_pairs = []
used_controls = set()

for idx, treated_row in treated.iterrows():
    treated_ps = treated_row['propensity']
    treated_id = treated_row['ID']

    # Find eligible controls (not used, within caliper)
    eligible_control = control[
        (~control['ID'].isin(used_controls)) &
        (np.abs(control['propensity'] - treated_ps) <= caliper)
    ]

    if len(eligible_control) > 0:
        # Calculate distances
        distances = np.abs(eligible_control['propensity'] - treated_ps)

        # Find minimum distance
        min_dist = distances.min()

        # Get all controls with minimum distance
        min_controls = eligible_control[distances == min_dist]

        # If tie, use smallest ID
        matched_control_idx = min_controls['ID'].idxmin()
        matched_control_id = min_controls.loc[matched_control_idx, 'ID']

        # Record match
        matched_pairs.append({
            'treated_id': treated_id,
            'control_id': matched_control_id,
            'treated_idx': idx,
            'control_idx': matched_control_idx
        })

        used_controls.add(matched_control_id)

matched_count = len(matched_pairs)
print(f"\n✓ Matched pair count: {matched_count}")

# ============================================================================
# STEP 7: QUANTILE REGRESSION ON MATCHED PAIRS
# ============================================================================
print("\n[7] Fitting quantile regression on matched pairs...")

# Extract treated units from matched pairs
matched_treated_indices = [pair['treated_idx'] for pair in matched_pairs]
matched_treated_data = train_data.loc[matched_treated_indices].copy()

# Fit quantile regression: balance ~ age + campaign + previous (tau=0.75)
qr_data = matched_treated_data[['balance', 'age', 'campaign', 'previous']].copy()
qr_model = smf.quantreg('balance ~ age + campaign + previous', qr_data)
qr_result = qr_model.fit(q=0.75, method='interior-point')

age_coef = qr_result.params['age']
campaign_coef = qr_result.params['campaign']

print(f"\n✓ Quantile Regression (τ=0.75) Results:")
print(f"  Age coefficient: {age_coef:.4f}")
print(f"  Campaign coefficient: {campaign_coef:.4f}")

# ============================================================================
# STEP 8: PLS REGRESSION WITH CROSS-VALIDATION
# ============================================================================
print("\n[8] Applying PLS Regression with StratifiedKFold...")

# Use source == 1 data
pls_data = train_data.copy()
X_pls = pls_data[['age', 'balance', 'housing', 'loan', 'campaign', 'pdays', 'previous']]
y_pls = pls_data['y']

# PLSRegression with n_components=3, scale=True
pls = PLSRegression(n_components=3, scale=True)

# StratifiedKFold with n_splits=5, shuffle=False (random_state not needed when shuffle=False)
skf = StratifiedKFold(n_splits=5, shuffle=False)

# Calculate R² scores for each fold
r2_scores = cross_val_score(pls, X_pls, y_pls, cv=skf, scoring='r2')
mean_r2 = r2_scores.mean()

print(f"\n✓ PLS Regression Cross-Validation Results:")
print(f"  Mean R² score: {mean_r2:.4f}")

# ============================================================================
# STEP 9: GAUSSIAN PROCESS REGRESSION
# ============================================================================
print("\n[9] Training Gaussian Process Regressor...")

# Filter source == 1 where y == 1, sort by ID
gp_data = train_data[train_data['y'] == 1].copy().sort_values('ID')

print(f"Total y=1 samples: {len(gp_data)}")

# Use first 500 for training
gp_train = gp_data.iloc[:500]
gp_test = gp_data.iloc[500:]

print(f"Training samples: {len(gp_train)}")
print(f"Test samples: {len(gp_test)}")

# Prepare data
X_gp_train = gp_train[['age', 'campaign']].values
y_gp_train = gp_train['balance'].values

X_gp_test = gp_test[['age', 'campaign']].values
y_gp_test = gp_test['balance'].values

# Define kernel: RBF with length_scale=1.0
kernel = RBF(length_scale=1.0)

# GaussianProcessRegressor with specified parameters
gpr = GaussianProcessRegressor(
    kernel=kernel,
    alpha=1e-10,
    n_restarts_optimizer=0,
    normalize_y=False,
    random_state=42
)

# Fit on first 500
gpr.fit(X_gp_train, y_gp_train)

# Predict on remaining
y_gp_pred = gpr.predict(X_gp_test)

# Calculate MSE
gp_mse = mean_squared_error(y_gp_test, y_gp_pred)

print(f"\n✓ Gaussian Process Regression Results:")
print(f"  MSE on remaining samples: {gp_mse:.2f}")

# ============================================================================
# STEP 10: BAYESIAN MODELING WITH PYMC
# ============================================================================
print("\n[10] Performing Bayesian modeling with PyMC...")

# Filter source == 1 where y == 1
bayes_data = train_data[train_data['y'] == 1]['campaign'].values

print(f"Samples for Bayesian modeling: {len(bayes_data)}")

# Build PyMC model
with pm.Model() as model:
    # Priors
    mu = pm.Normal('mu', mu=0, sigma=100)
    sigma = pm.HalfNormal('sigma', sigma=10)

    # Likelihood
    campaign_obs = pm.Normal('campaign_obs', mu=mu, sigma=sigma, observed=bayes_data)

    # Sample with NUTS
    trace = pm.sample(
        draws=2000,
        tune=500,
        chains=1,
        random_seed=42,
        return_inferencedata=True,
        progressbar=False
    )

# Extract posterior statistics
mu_posterior = trace.posterior['mu'].values.flatten()
mu_mean = mu_posterior.mean()
mu_ci_lower = np.percentile(mu_posterior, 2.5)
mu_ci_upper = np.percentile(mu_posterior, 97.5)

print(f"\n✓ Bayesian Modeling Results:")
print(f"  Posterior mean μ: {mu_mean:.3f}")
print(f"  95% CI: [{mu_ci_lower:.3f}, {mu_ci_upper:.3f}]")

# ============================================================================
# STEP 11: KOLMOGOROV-SMIRNOV TEST
# ============================================================================
print("\n[11] Performing Kolmogorov-Smirnov test...")

# Filter source == 1
ks_data = train_data.copy()

# Get balance for education == 2 and education == 1
balance_edu2 = ks_data[ks_data['education'] == 2]['balance'].values
balance_edu1 = ks_data[ks_data['education'] == 1]['balance'].values

print(f"Samples with education=2: {len(balance_edu2)}")
print(f"Samples with education=1: {len(balance_edu1)}")

# Two-sided KS test
ks_stat, ks_pval = ks_2samp(balance_edu2, balance_edu1)

print(f"\n✓ Kolmogorov-Smirnov Test Results:")
print(f"  KS statistic: {ks_stat:.4f}")
print(f"  p-value: {ks_pval:.5f}")

# ============================================================================
# STEP 12: VISUALIZATION
# ============================================================================
print("\n[12] Creating visualization: Age vs Balance...")

# Filter source == 1
viz_data = train_data.copy()

plt.figure(figsize=(10, 6))
plt.scatter(viz_data['age'], viz_data['balance'], alpha=0.5, s=10)
plt.xlabel('Age', fontsize=12)
plt.ylabel('Balance', fontsize=12)
plt.title('Age vs Balance (Training Data)', fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('age_vs_balance.png', dpi=300, bbox_inches='tight')
print("✓ Visualization saved as 'age_vs_balance.png'")

# ============================================================================
# SUMMARY OF RESULTS
# ============================================================================
print("\n" + "=" * 80)
print("ANALYSIS SUMMARY")
print("=" * 80)
print(f"\n1. Matched pair count: {matched_count}")
print(f"\n2. Quantile Regression (τ=0.75):")
print(f"   - Age coefficient: {age_coef:.4f}")
print(f"   - Campaign coefficient: {campaign_coef:.4f}")
print(f"\n3. PLS Regression Mean R²: {mean_r2:.4f}")
print(f"\n4. Gaussian Process MSE: {gp_mse:.2f}")
print(f"\n5. Bayesian Posterior Mean μ: {mu_mean:.3f}")
print(f"   95% CI: [{mu_ci_lower:.3f}, {mu_ci_upper:.3f}]")
print(f"\n6. Kolmogorov-Smirnov Test:")
print(f"   - KS statistic: {ks_stat:.4f}")
print(f"   - p-value: {ks_pval:.5f}")
print("\n" + "=" * 80)

# ============================================================================
# INSIGHTS FROM MATCHING ANALYSIS
# ============================================================================
print("\nKEY INSIGHTS FROM MATCHING ANALYSIS:")
print("=" * 80)

print("""
1. MATCHING SUCCESS & SAMPLE SIZE:
   - Successfully matched {0} pairs out of {1} treatment units
   - Match rate: {2:.1f}%
   - This indicates that we could find comparable control units for most
     treated individuals, suggesting good overlap in propensity scores

2. FACTORS INFLUENCING SUBSCRIPTION (from Propensity Score Model):
   - The logistic regression used age, balance, campaign, and previous contacts
   - These variables capture customer demographics, financial status, and
     marketing exposure
   - Propensity scores help identify customers with similar baseline
     characteristics but different outcomes

3. QUANTILE REGRESSION INSIGHTS (75th percentile):
   - Age coefficient: {3:.4f} - Shows relationship between age and balance
     at upper quartile
   - Campaign coefficient: {4:.4f} - Indicates impact of contact frequency
     on balance for successful customers
   - The 75th percentile focuses on customers with higher balances among
     those who subscribed

4. OVERALL PATTERNS:
   - The matching approach controls for selection bias by comparing similar
     customers
   - Allows causal inference about subscription factors while holding other
     characteristics constant
   - The matched sample provides a more balanced comparison for understanding
     what drives subscription success
""".format(matched_count, len(treated), 100 * matched_count / len(treated),
           age_coef, campaign_coef))

print("=" * 80)
print("Analysis completed successfully!")
print("=" * 80)
