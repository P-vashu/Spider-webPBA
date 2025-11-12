import pandas as pd
import numpy as np
from statsmodels.regression.linear_model import OLS
from statsmodels.tools import add_constant
import statsmodels.formula.api as smf
from statsmodels.regression.quantile_regression import QuantReg
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)

print("=" * 80)
print("PANEL DATA ECONOMETRIC ANALYSIS - WORLD HAPPINESS REPORT")
print("=" * 80)

# ============================================================================
# STEP 1: Load and standardize datasets
# ============================================================================
print("\n1. LOADING AND STANDARDIZING DATASETS")
print("-" * 80)

# Load 2017 data
df_2017 = pd.read_csv('2017.csv')
df_2017 = df_2017[['Country', 'Happiness.Score', 'Economy..GDP.per.Capita.', 'Freedom', 'Generosity']]
df_2017.columns = ['Country', 'Happiness_Score', 'GDP_per_capita', 'Freedom', 'Generosity']
df_2017['Year'] = 2017

# Load 2018 data
df_2018 = pd.read_csv('2018.csv')
df_2018 = df_2018[['Country or region', 'Score', 'GDP per capita', 'Freedom to make life choices', 'Generosity']]
df_2018.columns = ['Country', 'Happiness_Score', 'GDP_per_capita', 'Freedom', 'Generosity']
df_2018['Year'] = 2018

# Load 2019 data
df_2019 = pd.read_csv('2019.csv')
df_2019 = df_2019[['Country or region', 'Score', 'GDP per capita', 'Freedom to make life choices', 'Generosity']]
df_2019.columns = ['Country', 'Happiness_Score', 'GDP_per_capita', 'Freedom', 'Generosity']
df_2019['Year'] = 2019

print(f"2017 dataset: {len(df_2017)} countries")
print(f"2018 dataset: {len(df_2018)} countries")
print(f"2019 dataset: {len(df_2019)} countries")

# ============================================================================
# STEP 2: Identify common countries and create panel
# ============================================================================
print("\n2. CREATING PANEL DATASET")
print("-" * 80)

# Find countries present in all three years
countries_2017 = set(df_2017['Country'].unique())
countries_2018 = set(df_2018['Country'].unique())
countries_2019 = set(df_2019['Country'].unique())

common_countries = countries_2017 & countries_2018 & countries_2019
print(f"Countries present in all three years: {len(common_countries)}")

# Filter to keep only common countries
df_2017 = df_2017[df_2017['Country'].isin(common_countries)]
df_2018 = df_2018[df_2018['Country'].isin(common_countries)]
df_2019 = df_2019[df_2019['Country'].isin(common_countries)]

# Remove missing values
df_2017 = df_2017.dropna()
df_2018 = df_2018.dropna()
df_2019 = df_2019.dropna()

print(f"After removing missing values:")
print(f"  2017: {len(df_2017)} countries")
print(f"  2018: {len(df_2018)} countries")
print(f"  2019: {len(df_2019)} countries")

# Update common countries after removing missing values
common_countries = set(df_2017['Country'].unique()) & set(df_2018['Country'].unique()) & set(df_2019['Country'].unique())
df_2017 = df_2017[df_2017['Country'].isin(common_countries)]
df_2018 = df_2018[df_2018['Country'].isin(common_countries)]
df_2019 = df_2019[df_2019['Country'].isin(common_countries)]

# Stack the three datasets to create panel
panel_data = pd.concat([df_2017, df_2018, df_2019], ignore_index=True)
panel_data = panel_data.sort_values(['Country', 'Year']).reset_index(drop=True)

print(f"\nPanel dataset created: {len(panel_data)} observations ({len(common_countries)} countries × 3 years)")

# ============================================================================
# STEP 3: Fixed Effects Regression
# ============================================================================
print("\n3. FIXED EFFECTS REGRESSION MODEL")
print("-" * 80)

# Demean the data for fixed effects (within transformation)
panel_data['Country_code'] = pd.Categorical(panel_data['Country']).codes

# Calculate country means
country_means = panel_data.groupby('Country')[['Happiness_Score', 'GDP_per_capita', 'Freedom', 'Generosity']].mean()

# Create demeaned variables
panel_data_demeaned = panel_data.copy()
for var in ['Happiness_Score', 'GDP_per_capita', 'Freedom', 'Generosity']:
    panel_data_demeaned[f'{var}_demeaned'] = panel_data_demeaned.apply(
        lambda row: row[var] - country_means.loc[row['Country'], var], axis=1
    )

# Fit fixed effects model (using demeaned data, no intercept needed for within estimator)
y_demeaned = panel_data_demeaned['Happiness_Score_demeaned']
X_demeaned = panel_data_demeaned[['GDP_per_capita_demeaned', 'Freedom_demeaned', 'Generosity_demeaned']]

fe_model = OLS(y_demeaned, X_demeaned).fit()

# Calculate within R-squared
# Within R-squared = 1 - SSR/TSS (where TSS is total sum of squares of demeaned y)
within_r2 = fe_model.rsquared

print("Fixed Effects Model Results:")
print(f"  GDP_per_capita coefficient: {fe_model.params['GDP_per_capita_demeaned']:.4f}")
print(f"  Freedom coefficient: {fe_model.params['Freedom_demeaned']:.4f}")
print(f"  Generosity coefficient: {fe_model.params['Generosity_demeaned']:.4f}")
print(f"  Within R-squared: {within_r2:.4f}")

# ============================================================================
# STEP 4: Bootstrap Procedure
# ============================================================================
print("\n4. BOOTSTRAP PROCEDURE (500 iterations)")
print("-" * 80)

# Reset random seed
np.random.seed(42)

bootstrap_gdp_coefs = []
n_bootstrap = 500
country_list = list(common_countries)

for i in range(n_bootstrap):
    # Resample countries with replacement
    resampled_countries = np.random.choice(country_list, size=len(country_list), replace=True)

    # Keep all three years for each resampled country
    bootstrap_sample = pd.concat([
        panel_data[panel_data['Country'] == country] for country in resampled_countries
    ], ignore_index=True)

    # Calculate country means for bootstrap sample
    boot_country_means = bootstrap_sample.groupby('Country')[['Happiness_Score', 'GDP_per_capita', 'Freedom', 'Generosity']].mean()

    # Demean variables
    bootstrap_demeaned = bootstrap_sample.copy()
    for var in ['Happiness_Score', 'GDP_per_capita', 'Freedom', 'Generosity']:
        bootstrap_demeaned[f'{var}_demeaned'] = bootstrap_demeaned.apply(
            lambda row: row[var] - boot_country_means.loc[row['Country'], var], axis=1
        )

    # Fit fixed effects model
    y_boot = bootstrap_demeaned['Happiness_Score_demeaned']
    X_boot = bootstrap_demeaned[['GDP_per_capita_demeaned', 'Freedom_demeaned', 'Generosity_demeaned']]

    boot_model = OLS(y_boot, X_boot).fit()
    bootstrap_gdp_coefs.append(boot_model.params['GDP_per_capita_demeaned'])

bootstrap_gdp_coefs = np.array(bootstrap_gdp_coefs)

# Calculate confidence intervals
ci_lower = np.percentile(bootstrap_gdp_coefs, 2.5)
ci_upper = np.percentile(bootstrap_gdp_coefs, 97.5)

print(f"Bootstrap results (500 iterations):")
print(f"  2.5th percentile of GDP_per_capita coefficient: {ci_lower:.4f}")
print(f"  97.5th percentile of GDP_per_capita coefficient: {ci_upper:.4f}")
print(f"  Bootstrap 95% CI: [{ci_lower:.4f}, {ci_upper:.4f}]")

# ============================================================================
# STEP 5: Quantile Regression (2019 data only)
# ============================================================================
print("\n5. QUANTILE REGRESSION MODELS (2019 data)")
print("-" * 80)

# Use only 2019 data
data_2019 = panel_data[panel_data['Year'] == 2019].copy()

y_2019 = data_2019['Happiness_Score']
X_2019 = data_2019[['GDP_per_capita', 'Freedom', 'Generosity']]
X_2019_const = add_constant(X_2019)

# Fit quantile regressions at 25th, 50th, and 90th percentiles
qr_25 = QuantReg(y_2019, X_2019_const).fit(q=0.25)
qr_50 = QuantReg(y_2019, X_2019_const).fit(q=0.50)
qr_90 = QuantReg(y_2019, X_2019_const).fit(q=0.90)

print("Quantile Regression Results:")
print(f"\n  25th percentile:")
print(f"    GDP_per_capita coefficient: {qr_25.params['GDP_per_capita']:.4f}")
print(f"    Freedom coefficient: {qr_25.params['Freedom']:.4f}")
print(f"    Generosity coefficient: {qr_25.params['Generosity']:.4f}")

print(f"\n  50th percentile (median):")
print(f"    GDP_per_capita coefficient: {qr_50.params['GDP_per_capita']:.4f}")
print(f"    Freedom coefficient: {qr_50.params['Freedom']:.4f}")
print(f"    Generosity coefficient: {qr_50.params['Generosity']:.4f}")

print(f"\n  90th percentile:")
print(f"    GDP_per_capita coefficient: {qr_90.params['GDP_per_capita']:.4f}")
print(f"    Freedom coefficient: {qr_90.params['Freedom']:.4f}")
print(f"    Generosity coefficient: {qr_90.params['Generosity']:.4f}")

# ============================================================================
# STEP 6: Coefficient of Variation Analysis
# ============================================================================
print("\n6. COEFFICIENT OF VARIATION ANALYSIS")
print("-" * 80)

# Calculate coefficient of variation for each country
cv_results = []
for country in common_countries:
    country_data = panel_data[panel_data['Country'] == country]['Happiness_Score']
    mean_happiness = country_data.mean()
    std_happiness = country_data.std()
    cv = std_happiness / mean_happiness
    cv_results.append({'Country': country, 'CV': cv, 'Mean': mean_happiness, 'Std': std_happiness})

cv_df = pd.DataFrame(cv_results).sort_values('CV')

min_cv_country = cv_df.iloc[0]
max_cv_country = cv_df.iloc[-1]

print(f"Country with minimum coefficient of variation:")
print(f"  Country: {min_cv_country['Country']}")
print(f"  Coefficient of Variation: {min_cv_country['CV']:.4f}")
print(f"  (Mean: {min_cv_country['Mean']:.4f}, Std: {min_cv_country['Std']:.4f})")

print(f"\nCountry with maximum coefficient of variation:")
print(f"  Country: {max_cv_country['Country']}")
print(f"  Coefficient of Variation: {max_cv_country['CV']:.4f}")
print(f"  (Mean: {max_cv_country['Mean']:.4f}, Std: {max_cv_country['Std']:.4f})")

# ============================================================================
# STEP 7: Scatter Plot (2019 data)
# ============================================================================
print("\n7. CREATING SCATTER PLOT")
print("-" * 80)

plt.figure(figsize=(10, 6))
plt.scatter(data_2019['GDP_per_capita'], data_2019['Happiness_Score'], alpha=0.6, s=50)
plt.xlabel('GDP per capita', fontsize=12)
plt.ylabel('Happiness Score', fontsize=12)
plt.title('Relationship between GDP per capita and Happiness Score (2019)', fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('happiness_gdp_scatter_2019.png', dpi=300, bbox_inches='tight')
print("Scatter plot saved as 'happiness_gdp_scatter_2019.png'")

# ============================================================================
# STEP 8: Interpretation of Quantile Regression Results
# ============================================================================
print("\n8. QUANTILE REGRESSION INTERPRETATION")
print("-" * 80)

gdp_coef_25 = qr_25.params['GDP_per_capita']
gdp_coef_50 = qr_50.params['GDP_per_capita']
gdp_coef_90 = qr_90.params['GDP_per_capita']

print(f"GDP_per_capita coefficients across quantiles:")
print(f"  25th percentile: {gdp_coef_25:.4f}")
print(f"  50th percentile: {gdp_coef_50:.4f}")
print(f"  90th percentile: {gdp_coef_90:.4f}")

if gdp_coef_90 > gdp_coef_50 > gdp_coef_25:
    interpretation = "INCREASE"
    explanation = "The marginal effect of GDP_per_capita on Happiness_Score INCREASES as we move from lower to higher quantiles of the happiness distribution. This suggests that GDP has a stronger positive effect on happiness for already-happy countries."
elif gdp_coef_90 < gdp_coef_50 < gdp_coef_25:
    interpretation = "DECREASE"
    explanation = "The marginal effect of GDP_per_capita on Happiness_Score DECREASES as we move from lower to higher quantiles of the happiness distribution. This suggests that GDP has a stronger positive effect on happiness for less-happy countries."
else:
    # Check overall trend
    if gdp_coef_90 > gdp_coef_25:
        interpretation = "INCREASE"
        explanation = "The marginal effect of GDP_per_capita on Happiness_Score generally INCREASES as we move from lower to higher quantiles of the happiness distribution, though not monotonically."
    elif gdp_coef_90 < gdp_coef_25:
        interpretation = "DECREASE"
        explanation = "The marginal effect of GDP_per_capita on Happiness_Score generally DECREASES as we move from lower to higher quantiles of the happiness distribution, though not monotonically."
    else:
        interpretation = "REMAIN CONSTANT"
        explanation = "The marginal effect of GDP_per_capita on Happiness_Score remains relatively CONSTANT across quantiles of the happiness distribution."

print(f"\nInterpretation: {interpretation}")
print(f"{explanation}")

# ============================================================================
# SUMMARY OF KEY RESULTS
# ============================================================================
print("\n" + "=" * 80)
print("SUMMARY OF KEY RESULTS")
print("=" * 80)

print("\nFIXED EFFECTS REGRESSION:")
print(f"  • GDP_per_capita coefficient: {fe_model.params['GDP_per_capita_demeaned']:.4f}")
print(f"  • Freedom coefficient: {fe_model.params['Freedom_demeaned']:.4f}")
print(f"  • Generosity coefficient: {fe_model.params['Generosity_demeaned']:.4f}")
print(f"  • Within R-squared: {within_r2:.4f}")

print("\nBOOTSTRAP CONFIDENCE INTERVAL (GDP_per_capita):")
print(f"  • 2.5th percentile: {ci_lower:.4f}")
print(f"  • 97.5th percentile: {ci_upper:.4f}")

print("\nQUANTILE REGRESSION (2019 data):")
print(f"  • GDP_per_capita at 25th percentile: {gdp_coef_25:.4f}")
print(f"  • GDP_per_capita at 50th percentile: {gdp_coef_50:.4f}")
print(f"  • GDP_per_capita at 90th percentile: {gdp_coef_90:.4f}")
print(f"  • Freedom at 90th percentile: {qr_90.params['Freedom']:.4f}")

print("\nCOEFFICIENT OF VARIATION:")
print(f"  • Minimum: {min_cv_country['Country']} ({min_cv_country['CV']:.4f})")
print(f"  • Maximum: {max_cv_country['Country']} ({max_cv_country['CV']:.4f})")

print(f"\nQUANTILE REGRESSION INTERPRETATION:")
print(f"  • Marginal effect of GDP_per_capita: {interpretation}")

print("\n" + "=" * 80)
print("ANALYSIS COMPLETE")
print("=" * 80)
