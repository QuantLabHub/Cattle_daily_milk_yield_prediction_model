#finding the relation b/t variables
# =====================================================
# JANU'S FARM - CATTLE MILK YIELD ANALYSIS
# Correlation, Multiple Linear Regression & Visualization
# =====================================================

# 1. IMPORT LIBRARIES

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf


# 2. LOAD DATASET

df = pd.read_csv('janus_farm_3_month_cattle_dataset.csv')

print("\nDATASET PREVIEW:")
print(df.head())

print("\nDATASET INFORMATION:")
df.info()


# =====================================================
# 3. CORRELATION ANALYSIS
# =====================================================

numeric_data = df[
    [
        'age_years',
        'body_weight_kg',
        'daily_milk_yield_l',
        'daily_energy_intake_kcal'
    ]
].dropna()

# Pearson correlation

correlation = numeric_data.corr(method='pearson')

print("\nCORRELATION MATRIX:")
print(correlation)

# Correlation with milk yield

print("\nCORRELATION WITH MILK YIELD:")

print(
    correlation['daily_milk_yield_l']
    .sort_values(ascending=False)
)


# Correlation heatmap

plt.figure(figsize=(9, 6))

sns.heatmap(
    correlation,
    annot=True,
    cmap='coolwarm',
    fmt='.2f',
    linewidths=0.5
)

plt.title('Correlation with Daily Milk Yield')
plt.tight_layout()
plt.show()


# =====================================================
# 4. MULTIPLE LINEAR REGRESSION
# =====================================================

reg_data = df[
    [
        'breed',
        'age_years',
        'body_weight_kg',
        'daily_energy_intake_kcal',
        'daily_milk_yield_l'
    ]
].dropna()

model = smf.ols(
    'daily_milk_yield_l ~ age_years + body_weight_kg + '
    'daily_energy_intake_kcal + C(breed)',
    data=reg_data
).fit()


# Regression summary

print("\nMULTIPLE LINEAR REGRESSION RESULTS:")
print(model.summary())


# =====================================================
# 5. REGRESSION COEFFICIENTS
# =====================================================

print("\nREGRESSION COEFFICIENTS:")
print(model.params)

print("\nR-SQUARED:", model.rsquared)

print("ADJUSTED R-SQUARED:", model.rsquared_adj)

print("F-STATISTIC:", model.fvalue)

print("MODEL P-VALUE:", model.f_pvalue)


# =====================================================
# 6. ACTUAL VS PREDICTED MILK YIELD
# =====================================================

predicted = model.predict(reg_data)

plt.figure(figsize=(8, 6))

plt.scatter(
    reg_data['daily_milk_yield_l'],
    predicted,
    color='blue',
    alpha=0.7
)

plt.xlabel('Actual Milk Yield (L)')
plt.ylabel('Predicted Milk Yield (L)')

plt.title('Actual vs Predicted Milk Yield')

plt.tight_layout()
plt.show()


# =====================================================
# 7. INDIVIDUAL REGRESSION PLOTS
# =====================================================

variables = [
    'age_years',
    'body_weight_kg',
    'daily_energy_intake_kcal'
]

for var in variables:

    plt.figure(figsize=(7, 5))

    sns.regplot(
        data=reg_data,
        x=var,
        y='daily_milk_yield_l',
        scatter_kws={'alpha': 0.6},
        line_kws={'color': 'red'}
    )

    plt.title(f'{var} vs Daily Milk Yield')

    plt.tight_layout()

    plt.show()


# =====================================================
# END OF JANU'S FARM ANALYSIS
# =====================================================

#output
 
DATASET PREVIEW:
     farm_name record_date  ear_tag_id  breed_code breed  age_years  \
0  Janu's Farm  2026-08-15        1001           1    HF        3.1   
1  Janu's Farm  2026-08-15        1002           1    HF        6.5   
2  Janu's Farm  2026-08-15        1003           1    HF        7.2   
3  Janu's Farm  2026-08-15        1004           1    HF        7.1   
4  Janu's Farm  2026-08-15        1005           1    HF        5.6   

   body_weight_kg  daily_milk_yield_l  daily_energy_intake_kcal  \
0           563.3               32.26                     44299   
1           651.9               31.75                     44401   
2           642.8               27.52                     44654   
3           620.7               27.66                     42442   
4           664.7               33.03                     48185   

  lactating_status dry_period_status  
0        Lactating           Not dry  
1        Lactating           Not dry  
2        Lactating           Not dry  
3        Lactating           Not dry  
4        Lactating           Not dry  

DATASET INFORMATION:
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 5520 entries, 0 to 5519
Data columns (total 11 columns):
 #   Column                    Non-Null Count  Dtype  
---  ------                    --------------  -----  
 0   farm_name                 5520 non-null   object 
 1   record_date               5520 non-null   object 
 2   ear_tag_id                5520 non-null   int64  
 3   breed_code                5520 non-null   int64  
 4   breed                     5520 non-null   object 
 5   age_years                 5520 non-null   float64
 6   body_weight_kg            5520 non-null   float64
 7   daily_milk_yield_l        5520 non-null   float64
 8   daily_energy_intake_kcal  5520 non-null   int64  
 9   lactating_status          5520 non-null   object 
 10  dry_period_status         5520 non-null   object 
dtypes: float64(3), int64(3), object(5)
memory usage: 474.5+ KB

CORRELATION MATRIX:
                          age_years  body_weight_kg  daily_milk_yield_l  \
age_years                  1.000000        0.071886           -0.083270   
body_weight_kg             0.071886        1.000000            0.748997   
daily_milk_yield_l        -0.083270        0.748997            1.000000   
daily_energy_intake_kcal  -0.043266        0.875664            0.966305   

                          daily_energy_intake_kcal  
age_years                                -0.043266  
body_weight_kg                            0.875664  
daily_milk_yield_l                        0.966305  
daily_energy_intake_kcal                  1.000000  

CORRELATION WITH MILK YIELD:
daily_milk_yield_l          1.000000
daily_energy_intake_kcal    0.966305
body_weight_kg              0.748997
age_years                  -0.083270
Name: daily_milk_yield_l, dtype: float64

MULTIPLE LINEAR REGRESSION RESULTS:
                            OLS Regression Results                            
==============================================================================
Dep. Variable:     daily_milk_yield_l   R-squared:                       0.979
Model:                            OLS   Adj. R-squared:                  0.979
Method:                 Least Squares   F-statistic:                 3.163e+04
Date:                Mon, 16 Nov 2026   Prob (F-statistic):               0.00
Time:                        17:08:52   Log-Likelihood:                -8582.4
No. Observations:                5520   AIC:                         1.718e+04
Df Residuals:                    5511   BIC:                         1.724e+04
Df Model:                           8                                         
Covariance Type:            nonrobust                                         
============================================================================================
                               coef    std err          t      P>|t|      [0.025      0.975]
--------------------------------------------------------------------------------------------
Intercept                   -2.5567      0.378     -6.763      0.000      -3.298      -1.816
C(breed)[T.Gir]             -3.6240      0.119    -30.458      0.000      -3.857      -3.391
C(breed)[T.HF]               1.1325      0.067     16.906      0.000       1.001       1.264
C(breed)[T.Jersey]          -3.1206      0.107    -29.116      0.000      -3.331      -2.910
C(breed)[T.Red Sindhi]      -4.9751      0.157    -31.761      0.000      -5.282      -4.668
C(breed)[T.Sahiwal]         -3.7400      0.125    -29.994      0.000      -3.984      -3.496
age_years                   -0.0148      0.009     -1.645      0.100      -0.032       0.003
body_weight_kg              -0.0357      0.000    -85.606      0.000      -0.037      -0.035
daily_energy_intake_kcal     0.0012    8.8e-06    137.902      0.000       0.001       0.001
==============================================================================
Omnibus:                       26.691   Durbin-Watson:                   2.361
Prob(Omnibus):                  0.000   Jarque-Bera (JB):               26.983
Skew:                          -0.165   Prob(JB):                     1.38e-06
Kurtosis:                       3.093   Cond. No.                     9.57e+05
==============================================================================

Notes:
[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
[2] The condition number is large, 9.57e+05. This might indicate that there are
strong multicollinearity or other numerical problems.

REGRESSION COEFFICIENTS:
Intercept                  -2.556707
C(breed)[T.Gir]            -3.624022
C(breed)[T.HF]              1.132484
C(breed)[T.Jersey]         -3.120554
C(breed)[T.Red Sindhi]     -4.975115
C(breed)[T.Sahiwal]        -3.740026
age_years                  -0.014803
body_weight_kg             -0.035747
daily_energy_intake_kcal    0.001214
dtype: float64

R-SQUARED: 0.9786849480660454
ADJUSTED R-SQUARED: 0.978654006237798
F-STATISTIC: 31629.835840325573
MODEL P-VALUE: 0.0


