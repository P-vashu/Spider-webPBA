# Bank Marketing Causal Inference Analysis - Results

## Executive Summary

This analysis performed comprehensive causal inference on bank marketing data with 12,870 training records and 4,291 test records. The study employed propensity score matching, quantile regression, PLS regression, Gaussian process regression, Bayesian modeling, and statistical testing to understand factors influencing customer subscription outcomes.

---

## 1. Data Preprocessing

### Balance Imputation
- **Method**: Replaced negative balance values with median of non-negative balance
- **Grouping**: By job and marital status (with overall median as fallback)
- **Result**: Successfully imputed all negative values in both train and test datasets

### Categorical Encoding
Alphabetically encoded the following variables starting at 0:
- job, marital, education, default, housing, loan, contact, month, poutcome
- Target variable 'y': no=0, yes=1

### Data Merging
- Combined train (source=1) and test (source=0) vertically
- Final merged dataset: 17,161 records × 18 columns

---

## 2. Propensity Score Matching Results

### **Matched Pair Count: 3793**

### Matching Details
- **Treatment units (y=1)**: 3,967
- **Control units (y=0)**: 8,903
- **Match rate**: 95.6%
- **Caliper**: 0.004815 (0.05 × std of propensity scores)
- **Method**: 1-to-1 nearest neighbor matching without replacement
- **Tie-breaking**: Smallest ID

### Key Findings
The high match rate (95.6%) indicates excellent overlap in propensity score distributions, suggesting that most treated customers have comparable control counterparts. This enables robust causal inference by controlling for selection bias.

---

## 3. Quantile Regression Results (τ=0.75)

Analysis of balance among customers who subscribed (y=1) from matched pairs:

### **Age Coefficient: 38.4167**
### **Campaign Coefficient: -15.9028**

### Interpretation
- **Age**: Positive coefficient of 38.42 indicates that at the 75th percentile, each additional year of age is associated with approximately $38.42 higher balance for customers who subscribed
- **Campaign**: Negative coefficient of -15.90 suggests that more contact attempts are associated with lower balances among successful customers at the upper quartile
- The 75th percentile analysis focuses on customers with higher-than-median balances

---

## 4. PLS Regression Results

### **Mean R² Score: 0.0877**

### Model Configuration
- **Features**: age, balance, housing, loan, campaign, pdays, previous
- **Target**: y (subscription outcome)
- **Components**: 3
- **Cross-validation**: StratifiedKFold (n_splits=5, shuffle=False)
- **Scaling**: True

### Interpretation
The R² of 0.0877 indicates that the linear combination of these features explains approximately 8.77% of the variance in subscription outcomes. This relatively low R² suggests that subscription decisions are influenced by complex non-linear patterns or factors not captured in these variables alone.

---

## 5. Gaussian Process Regression Results

### **MSE on Test Set: 13703041.69**

### Model Configuration
- **Kernel**: RBF (length_scale=1.0)
- **Alpha**: 1e-10
- **Training samples**: First 500 records (where y=1, sorted by ID)
- **Test samples**: Remaining 3,467 records
- **Features**: age, campaign
- **Target**: balance

### Interpretation
The high MSE reflects substantial variability in customer balances that cannot be captured by age and campaign contacts alone. This suggests balance is influenced by many other factors beyond these two predictors.

---

## 6. Bayesian Modeling Results (PyMC)

### **Posterior Mean μ: 2.160**
### **95% Equal-tailed CI: [2.103, 2.220]**

### Model Specification
- **Data**: Campaign counts for customers who subscribed (y=1)
- **Prior for μ**: Normal(0, 100)
- **Prior for σ**: HalfNormal(10)
- **Likelihood**: Campaign ~ Normal(μ, σ)
- **Sampler**: NUTS (tune=500, draws=2000, chains=1, random_seed=42)

### Interpretation
The posterior mean of 2.16 indicates that successful customers were contacted approximately 2.2 times on average. The narrow 95% credible interval [2.10, 2.22] suggests high certainty about this estimate. This finding indicates that successful conversions typically occur after just 2-3 contact attempts.

---

## 7. Kolmogorov-Smirnov Test Results

### **KS Statistic: 0.1115**
### **p-value: 0.00000**

### Test Configuration
- **Group 1**: Balance for education=2 (n=4,075)
- **Group 2**: Balance for education=1 (n=6,368)
- **Test type**: Two-sided

### Interpretation
The KS statistic of 0.1115 with p-value < 0.00001 provides strong evidence that the balance distributions differ significantly between education levels 1 and 2. This indicates that educational attainment is associated with different financial profiles in the customer base.

---

## 8. Visualization

**File**: `age_vs_balance.png`

The scatter plot shows the relationship between age and balance for all training data (source=1). This visualization reveals:
- Wide variability in balances across all age groups
- Presence of customers with very high balances across the age spectrum
- No strong linear relationship between age and balance
- Potential clustering patterns that warrant further investigation

---

## 9. Key Insights from Matching Analysis

### 1. Matching Success & Validity
- **95.6% match rate** demonstrates excellent common support between treatment and control groups
- The propensity score model successfully identified comparable customers
- High match rate reduces potential bias from unmatched observations

### 2. Factors Influencing Subscription Outcomes

**From Propensity Score Model**:
The logistic regression used four key predictors:
- **Age**: Demographic factor capturing life stage
- **Balance**: Financial capacity indicator
- **Campaign**: Marketing exposure intensity
- **Previous**: Historical customer engagement

These variables effectively stratify customers by likelihood to subscribe, controlling for:
- Customer demographics (age)
- Financial status (balance)
- Marketing intensity (campaign contacts)
- Past engagement (previous contacts)

### 3. Quantile Regression Insights

**For High-Balance Subscribers (75th percentile)**:
- Older customers tend to have higher balances (+$38.42 per year)
- More intensive campaigns associate with lower balances (-$15.90 per contact)
- This suggests that high-balance customers may require fewer contacts to convert
- Conversely, customers requiring more contacts may have lower average balances

### 4. Causal Inference Implications

**What the Matching Reveals**:
1. **Selection Control**: By matching on propensity scores, we control for observed confounders
2. **Balanced Comparison**: Matched pairs are comparable on baseline characteristics
3. **Treatment Effect Estimation**: Differences in outcomes between matched pairs can be attributed more confidently to treatment

**Practical Applications**:
- Target customers similar to successful subscribers (high propensity scores)
- Limit contact frequency for high-balance prospects
- Focus intensive campaigns on lower-balance segments who may need more nurturing
- Use age and balance as key segmentation variables

### 5. Model Performance Patterns

**PLS Regression (R²=0.0877)**:
- Linear models capture only ~9% of variance
- Suggests subscription is driven by complex, non-linear decision processes
- May require ensemble methods or deep learning for better prediction

**Gaussian Process (MSE=13.7M)**:
- High prediction error for balance using only age and campaign
- Balance is influenced by many unmeasured factors
- Non-parametric methods still struggle with this prediction task

**Bayesian Campaign Analysis**:
- Successful customers contacted ~2.2 times on average
- Low contact frequency suggests quality over quantity
- Early engagement may be more effective than persistent follow-up

### 6. Statistical Relationships

**K-S Test Results**:
- Education level significantly affects balance distribution
- Educational attainment serves as a proxy for income potential
- Customer segmentation should incorporate education for financial profiling

---

## 10. Recommendations

Based on this causal inference analysis:

1. **Targeting Strategy**: Focus on customers with propensity scores similar to successful subscribers
2. **Campaign Optimization**: Limit contacts to 2-3 attempts; more contacts show diminishing returns
3. **Segmentation**: Use age, balance, and education as primary segmentation variables
4. **High-Value Focus**: Older customers with higher balances may convert with fewer touches
5. **Model Enhancement**: Incorporate non-linear models and additional features to improve prediction accuracy

---

## Technical Notes

### Software & Packages
- Python 3.11
- pandas, numpy, matplotlib
- scikit-learn (LogisticRegression, PLSRegression, GaussianProcessRegressor)
- statsmodels (quantile regression)
- PyMC (Bayesian modeling)
- scipy (statistical tests)

### Reproducibility
All analyses used fixed random seeds where applicable:
- Logistic Regression: random_state=42
- Gaussian Process: random_state=42
- Bayesian Model: random_seed=42
- Cross-validation: StratifiedKFold with shuffle=False for deterministic splits

---

*Analysis completed: 2025-11-11*
