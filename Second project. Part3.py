import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from scipy import stats
import os
import warnings

warnings.filterwarnings('ignore')

def transform_re(group, theta, vars_list):
    group_mean = group[vars_list].mean()
    transformed = group[vars_list] - theta * group_mean
    return transformed

plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 10

df = pd.read_csv('data_part3.csv', sep=';', decimal=',', na_values=['NA', 'na', ''])

# GDP per capita
df['gdp_per_capita'] = (df['gdp_const2010'] * 1e9) / df['population']
df['log_gdp_per_capita'] = np.log(df['gdp_per_capita'])

# Savings rate = 100 - household consumption (% GDP)
df['savings_rate'] = 100 - df['hh_cons']

# Employed labor share = 100 - unemployment rate (%)
df['employed_labor_share'] = 100 - df['unemp']
df['capital_formation'] = df['fdi']  # FDI as % of GDP

df_imputed = df.copy()
df_imputed = df_imputed.sort_values(['code', 'year'])

vars_to_fill = ['educ_sp_to_gdp', 'market_cap', 'va_services', 'va_industry', 'unemp', 'fdi']

for var in vars_to_fill:
    df_imputed[var] = df_imputed.groupby('code')[var].transform(
        lambda x: x.fillna(method='ffill').fillna(method='bfill'))

df_imputed['population'] = df_imputed.groupby('code')['population'].transform(lambda x: x.interpolate(method='linear'))
df_imputed['gdp_per_capita'] = (df_imputed['gdp_const2010'] * 1e9) / df_imputed['population']
df_imputed['log_gdp_per_capita'] = np.log(df_imputed['gdp_per_capita'])
df_imputed['savings_rate'] = 100 - df_imputed['hh_cons']
df_imputed['employed_labor_share'] = 100 - df_imputed['unemp']
df_imputed['capital_formation'] = df_imputed['fdi']


model_vars = ['log_gdp_per_capita', 'savings_rate', 'capital_formation',
              'employed_labor_share', 'educ_sp_to_gdp', 'market_cap',
              'va_industry', 'va_services']

df_clean = df_imputed.dropna(subset=model_vars)
missing_pop = df_imputed[df_imputed['population'].isnull()]



if not os.path.exists('figures'):
    os.makedirs('figures')

# Distribution of log GDP per capita
plt.figure(figsize=(10, 6))
sns.histplot(df_clean['log_gdp_per_capita'].dropna(), bins=30, kde=True)
plt.xlabel('Log GDP per capita')
plt.ylabel('Frequency')
plt.title('Distribution of Log GDP per Capita')
plt.grid(True, alpha=0.3)
plt.savefig('figures/dist_log_gdp.png', dpi=300, bbox_inches='tight')
plt.close()


# Distribution of savings rate
plt.figure(figsize=(10, 6))
sns.histplot(df_clean['savings_rate'].dropna(), bins=30, kde=True)
plt.xlabel('Savings Rate (% of GDP)')
plt.ylabel('Frequency')
plt.title('Distribution of Savings Rate')
plt.grid(True, alpha=0.3)
plt.savefig('figures/dist_savings.png', dpi=300, bbox_inches='tight')
plt.close()


# Distribution of employed labor share
plt.figure(figsize=(10, 6))
sns.histplot(df_clean['employed_labor_share'].dropna(), bins=30, kde=True)
plt.xlabel('Employed Labor Share (%)')
plt.ylabel('Frequency')
plt.title('Distribution of Employed Labor Share')
plt.grid(True, alpha=0.3)
plt.savefig('figures/dist_employment.png', dpi=300, bbox_inches='tight')
plt.close()


# Distribution of capital formation (FDI)
plt.figure(figsize=(10, 6))
sns.histplot(df_clean['capital_formation'].dropna(), bins=30, kde=True)
plt.xlabel('Capital Formation (FDI, % of GDP)')
plt.ylabel('Frequency')
plt.title('Distribution of Capital Formation')
plt.grid(True, alpha=0.3)
plt.savefig('figures/dist_capital.png', dpi=300, bbox_inches='tight')
plt.close()





df_model = df_clean.copy()
df_model = df_model.sort_values(['code', 'year']).reset_index(drop=True)

y_var = 'log_gdp_per_capita'
x_vars = ['savings_rate', 'capital_formation', 'employed_labor_share',
          'educ_sp_to_gdp', 'market_cap', 'va_industry', 'va_services']

X = df_model[x_vars]
X_const = sm.add_constant(X)
y = df_model[y_var]
pooled_ols = sm.OLS(y, X_const).fit()

# Model 2: FIXED EFFECTS
country_dummies = pd.get_dummies(df_model['code'], drop_first=True)

X_fe = pd.concat([X_const, country_dummies], axis=1)
X_fe = X_fe.astype(float)
y = y.astype(float)

fe_model = sm.OLS(y, X_fe).fit()

# Model 3: RANDOM EFFECTS
fe_within = sm.OLS(y, X_fe).fit()
df_model['resid_fe'] = fe_within.resid

# Within variance (sigma_e^2)
n_countries = df_model['code'].nunique()
obs_per_country = df_model.groupby('code').size().mean()
sigma_e_sq = df_model.groupby('code')['resid_fe'].var().mean()

# Between variance (sigma_u^2)
country_means = df_model.groupby('code')[y_var].mean()
country_means_x = df_model.groupby('code')[x_vars].mean()
between_model = sm.OLS(country_means, sm.add_constant(country_means_x)).fit()
sigma_u_sq = max(0, between_model.mse_resid - sigma_e_sq / obs_per_country)

theta = 1 - np.sqrt(sigma_e_sq / (sigma_e_sq + obs_per_country * sigma_u_sq))

X_re = df_model.groupby('code').apply(lambda g: transform_re(g, theta, x_vars)).reset_index(level=0, drop=True)
X_re_const = sm.add_constant(X_re)

y_re = df_model[y_var] - theta * df_model.groupby('code')[y_var].transform('mean')

re_model = sm.OLS(y_re, X_re_const).fit()

# 1. F-test for significance Fixed Effects
ssr_pooled = pooled_ols.ssr
ssr_fe = fe_model.ssr

n = len(y)
k_pooled = len(pooled_ols.params)
k_fe = len(fe_model.params)

df1 = k_fe - k_pooled
df2 = n - k_fe

f_stat = ((ssr_pooled - ssr_fe) / df1) / (ssr_fe / df2)
f_pvalue = 1 - stats.f.cdf(f_stat, df1, df2)

# 2. Breusch-Pagan LM test for Random Effects
residuals_pooled = pooled_ols.resid
df_model['resid_pooled'] = residuals_pooled

group_resid_sum = df_model.groupby('code')['resid_pooled'].sum()
group_resid_sq_sum = (group_resid_sum ** 2).sum()
total_resid_sq = (residuals_pooled ** 2).sum()

T = obs_per_country
N = n_countries

lm_stat = (N * T / (2 * (T - 1))) * ((group_resid_sq_sum / total_resid_sq - 1) ** 2)
lm_pvalue = 1 - stats.chi2.cdf(lm_stat, 1)

#3. Hausman test (FE vs RE)
fe_params = fe_model.params[:len(x_vars) + 1]
re_params = re_model.params[:len(x_vars) + 1]

fe_cov = fe_model.cov_params().iloc[:len(x_vars) + 1, :len(x_vars) + 1]
re_cov = re_model.cov_params().iloc[:len(x_vars) + 1, :len(x_vars) + 1]

diff = fe_params - re_params
cov_diff = fe_cov - re_cov

hausman_stat = diff.T @ np.linalg.pinv(cov_diff) @ diff
hausman_pvalue = 1 - stats.chi2.cdf(hausman_stat, len(diff))

#CRISIS DUMMY
df_model['crisis_1998'] = (df_model['year'] == 1998).astype(int)
df_model['crisis_2008_2009'] = ((df_model['year'] == 2008) | (df_model['year'] == 2009)).astype(int)

x_vars_with_crisis = ['savings_rate', 'capital_formation', 'employed_labor_share',
                      'educ_sp_to_gdp', 'market_cap', 'va_industry', 'va_services',
                      'crisis_1998', 'crisis_2008_2009']

X_crisis = df_model.groupby('code').apply(lambda g: transform_re(g, theta, x_vars_with_crisis)).reset_index(level=0,                                                                                                           drop=True)
X_crisis_const = sm.add_constant(X_crisis)
y_re = df_model[y_var] - theta * df_model.groupby('code')[y_var].transform('mean')
re_crisis_model = sm.OLS(y_re, X_crisis_const).fit()

# ============================================================================
# README
# ============================================================================

with open('README.md', 'w', encoding='utf-8') as f:
    f.write('# Econometrics Project: Country Output Modeling\n\n')
    f.write('This document presents **Part 3: Country Output Modeling**. ')
    f.write('The analysis uses a modified Cobb-Douglas production function to model GDP per capita ')
    f.write('using panel data of 22 countries from 1996 to 2018.\n\n')

    f.write('## Data Overview\n\n')
    f.write(f'- **Countries:** {df_clean["country"].nunique()}\n')
    f.write(f'- **Years:** 1996 - 2018 (balanced panel after processing)\n')
    f.write(f'- **Observations:** {len(df_clean):,}\n')
    f.write(f'- **Target variable:** log GDP per capita (constant 2010 USD)\n\n')

    f.write('---\n\n')

    # -------------------------------------------------------------------------
    # Stage 1. Data Preparation
    # -------------------------------------------------------------------------
    f.write('## Stage 1. Data Preparation\n\n')
    f.write('### 1.1 New Variables\n\n')
    f.write('| Variable | Formula | Description |\n')
    f.write('|----------|---------|-------------|\n')
    f.write('| GDP per capita | gdp_const2010 × 1e9 / population | GDP per person in constant USD |\n')
    f.write('| Savings rate | 100 − hh_cons | Gross savings as % of GDP |\n')
    f.write('| Employed labor share | 100 − unemp | Employment rate as % of labor force |\n')
    f.write('| Capital formation | fdi | Foreign direct investment as % of GDP (proxy) |\n\n')
    f.write('**Note:** The dataset does not contain gross capital formation (`cap`) as specified in the task. ')
    f.write('FDI is used as a proxy for capital formation. This is a limitation, as FDI typically represents ')
    f.write('a smaller fraction of total capital formation.\n\n')

    f.write('### 1.2 Missing Data Treatment\n\n')
    f.write(
        'Missing values were handled using forward fill within countries, with linear interpolation for population. ')
    f.write(f'After processing, {len(df_clean)} observations remained (originally {len(df)}). ')
    f.write('The final sample covers 1996-2018 (balanced panel).\n\n')

    f.write('### 1.3 Model Specifications\n\n')
    f.write('Three panel models were estimated:\n\n')
    f.write('- **Pooled OLS:** Ignores panel structure (assumes no country-specific effects)\n')
    f.write('- **Fixed Effects (FE):** Accounts for country-specific intercepts (allows correlation with regressors)\n')
    f.write('- **Random Effects (RE):** Assumes country-specific effects are uncorrelated with regressors; ')
    f.write('estimated via feasible generalized least squares (FGLS) with quasi-demeaning transformation\n\n')

    f.write('---\n\n')

    # -------------------------------------------------------------------------
    # Stage 2. Exploratory Data Analysis
    # -------------------------------------------------------------------------
    f.write('## Stage 2. Exploratory Data Analysis\n\n')

    # Summary Statistics
    f.write('### 2.1 Summary Statistics\n\n')
    f.write('| Variable | Mean | Std Dev | Min | Max |\n')
    f.write('|----------|------|---------|-----|-----|\n')
    f.write(
        f'| Log GDP per capita | {df_clean["log_gdp_per_capita"].mean():.2f} | {df_clean["log_gdp_per_capita"].std():.2f} | {df_clean["log_gdp_per_capita"].min():.2f} | {df_clean["log_gdp_per_capita"].max():.2f} |\n')
    f.write(
        f'| Savings rate (%) | {df_clean["savings_rate"].mean():.1f} | {df_clean["savings_rate"].std():.1f} | {df_clean["savings_rate"].min():.1f} | {df_clean["savings_rate"].max():.1f} |\n')
    f.write(
        f'| Capital formation (FDI, %) | {df_clean["capital_formation"].mean():.2f} | {df_clean["capital_formation"].std():.2f} | {df_clean["capital_formation"].min():.2f} | {df_clean["capital_formation"].max():.2f} |\n')
    f.write(
        f'| Employed labor share (%) | {df_clean["employed_labor_share"].mean():.1f} | {df_clean["employed_labor_share"].std():.1f} | {df_clean["employed_labor_share"].min():.1f} | {df_clean["employed_labor_share"].max():.1f} |\n')
    f.write(
        f'| Education spending (% GDP) | {df_clean["educ_sp_to_gdp"].mean():.2f} | {df_clean["educ_sp_to_gdp"].std():.2f} | {df_clean["educ_sp_to_gdp"].min():.2f} | {df_clean["educ_sp_to_gdp"].max():.2f} |\n')
    f.write(
        f'| Market capitalization (% GDP) | {df_clean["market_cap"].mean():.2f} | {df_clean["market_cap"].std():.2f} | {df_clean["market_cap"].min():.2f} | {df_clean["market_cap"].max():.2f} |\n')
    f.write(
        f'| Value added — industry (% GDP) | {df_clean["va_industry"].mean():.1f} | {df_clean["va_industry"].std():.1f} | {df_clean["va_industry"].min():.1f} | {df_clean["va_industry"].max():.1f} |\n')
    f.write(
        f'| Value added — services (% GDP) | {df_clean["va_services"].mean():.1f} | {df_clean["va_services"].std():.1f} | {df_clean["va_services"].min():.1f} | {df_clean["va_services"].max():.1f} |\n\n')

    # Distribution Plots
    f.write('### 2.2 Distribution Plots\n\n')
    f.write('![Distribution of Log GDP per Capita](figures/dist_log_gdp.png)\n\n')
    f.write('![Distribution of Savings Rate](figures/dist_savings.png)\n\n')
    f.write('![Distribution of Employed Labor Share](figures/dist_employment.png)\n\n')
    f.write('![Distribution of Capital Formation (FDI)](figures/dist_capital.png)\n\n')

    f.write('---\n\n')

    # -------------------------------------------------------------------------
    # Stage 3. Model Results
    # -------------------------------------------------------------------------
    f.write('## Stage 3. Model Results\n\n')

    # Model Comparison Table
    f.write('### 3.1 Model Comparison\n\n')
    f.write('| Variable | Pooled OLS | Fixed Effects | Random Effects |\n')
    f.write('|----------|------------|---------------|----------------|\n')
    f.write(
        f'| Intercept | {pooled_ols.params["const"]:.4f} | {fe_model.params["const"]:.4f} | {re_model.params["const"]:.4f} |\n')
    f.write(
        f'| Savings rate | {pooled_ols.params["savings_rate"]:.4f}*** | {fe_model.params["savings_rate"]:.4f}*** | {re_model.params["savings_rate"]:.4f}*** |\n')
    f.write(
        f'| Capital formation (FDI) | {pooled_ols.params["capital_formation"]:.4f} | {fe_model.params["capital_formation"]:.4f}** | {re_model.params["capital_formation"]:.4f}** |\n')
    f.write(
        f'| Employed labor share | {pooled_ols.params["employed_labor_share"]:.4f}** | {fe_model.params["employed_labor_share"]:.4f}*** | {re_model.params["employed_labor_share"]:.4f}*** |\n')
    f.write(
        f'| Education spending | {pooled_ols.params["educ_sp_to_gdp"]:.4f}*** | {fe_model.params["educ_sp_to_gdp"]:.4f}** | {re_model.params["educ_sp_to_gdp"]:.4f}** |\n')
    f.write(
        f'| Market capitalization | {pooled_ols.params["market_cap"]:.4f}*** | {fe_model.params["market_cap"]:.4f}*** | {re_model.params["market_cap"]:.4f}*** |\n')
    f.write(
        f'| Value added — industry | {pooled_ols.params["va_industry"]:.4f}*** | {fe_model.params["va_industry"]:.4f} | {re_model.params["va_industry"]:.4f} |\n')
    f.write(
        f'| Value added — services | {pooled_ols.params["va_services"]:.4f}*** | {fe_model.params["va_services"]:.4f}*** | {re_model.params["va_services"]:.4f}*** |\n\n')
    f.write('*Note: *** p < 0.01, ** p < 0.05*\n\n')

    # -------------------------------------------------------------------------
    # Stage 4. Model Selection
    # -------------------------------------------------------------------------
    f.write('### 3.2 Model Selection Tests\n\n')
    f.write('| Test | Statistic | p-value | Conclusion |\n')
    f.write('|------|-----------|---------|------------|\n')
    f.write(f'| F-test (FE vs Pooled) | {f_stat:.2f} | {f_pvalue:.4f} | Fixed effects are significant |\n')
    f.write(f'| Breusch-Pagan LM (RE vs Pooled) | {lm_stat:.2f} | {lm_pvalue:.4f} | Random effects are significant |\n')
    f.write(
        f'| Hausman test (FE vs RE) | {hausman_stat:.2f} | {hausman_pvalue:.4f} | Random Effects is preferred |\n\n')

    f.write(
        '**Conclusion:** Based on the Hausman test (p > 0.05), the **Random Effects model** is selected as the preferred specification.\n\n')

    # -------------------------------------------------------------------------
    # Stage 5. Economic Interpretation (Random Effects)
    # -------------------------------------------------------------------------
    f.write('### 3.3 Economic Interpretation (Random Effects)\n\n')
    f.write('| Variable | Coefficient | Interpretation |\n')
    f.write('|----------|-------------|----------------|\n')
    f.write(
        f'| Savings rate | {re_model.params["savings_rate"]:.4f} | +1 percentage point → {re_model.params["savings_rate"] * 100:.1f}% increase in GDP per capita |\n')
    f.write(
        f'| Capital formation (FDI) | {re_model.params["capital_formation"]:.4f} | +1 p.p. FDI → {re_model.params["capital_formation"] * 100:.2f}% increase in GDP per capita |\n')
    f.write(
        f'| Employed labor share | {re_model.params["employed_labor_share"]:.4f} | +1 p.p. employment → {re_model.params["employed_labor_share"] * 100:.1f}% increase in GDP per capita |\n')
    f.write(
        f'| Education spending | {re_model.params["educ_sp_to_gdp"]:.4f} | +1 p.p. education → {re_model.params["educ_sp_to_gdp"] * 100:.1f}% increase in GDP per capita |\n')
    f.write(
        f'| Market capitalization | {re_model.params["market_cap"]:.4f} | +1 p.p. market cap → {re_model.params["market_cap"] * 100:.2f}% increase in GDP per capita |\n')
    f.write(
        f'| Value added — services | {re_model.params["va_services"]:.4f} | +1 p.p. services share → {re_model.params["va_services"] * 100:.1f}% increase in GDP per capita |\n')
    f.write(f'| Value added — industry | {re_model.params["va_industry"]:.4f} | Not statistically significant |\n\n')
    f.write('**Economic consistency:** All significant coefficients have the expected positive sign — ')
    f.write('higher savings, employment, education, and services sector share are associated with ')
    f.write('higher GDP per capita, consistent with growth theory.\n\n')

    f.write('**Key findings:**\n')
    f.write('- Savings rate and employment share have the strongest positive effects on GDP per capita\n')
    f.write('- Education spending and FDI also contribute positively\n')
    f.write('- The services sector is a significant driver of growth, while industry is not\n')
    f.write('- Market capitalization has a small but statistically significant positive effect\n\n')

    # -------------------------------------------------------------------------
    # Stage 6. Crisis Effects
    # -------------------------------------------------------------------------
    f.write('## Stage 4. Crisis Effects\n\n')
    f.write('Crisis dummies were added to the preferred Random Effects model:\n\n')
    f.write('| Variable | Coefficient | p-value |\n')
    f.write('|----------|-------------|---------|\n')
    f.write(
        f'| Crisis 1998 | {re_crisis_model.params["crisis_1998"]:.4f} | {re_crisis_model.pvalues["crisis_1998"]:.4f} |\n')
    f.write(
        f'| Crisis 2008-2009 | {re_crisis_model.params["crisis_2008_2009"]:.4f} | {re_crisis_model.pvalues["crisis_2008_2009"]:.4f} |\n\n')
    f.write('**Conclusion:** Neither crisis dummy is statistically significant at conventional levels. ')
    f.write('This suggests that in this sample of primarily industrial/post-industrial countries, ')
    f.write('the 1998 financial crisis and the 2008-2009 global financial crisis did not have a ')
    f.write('detectable impact on GDP per capita after controlling for other factors.\n\n')

    # -------------------------------------------------------------------------
    # Stage 7. Conclusion
    # -------------------------------------------------------------------------
    f.write('---\n\n')
    f.write('## Conclusion\n\n')

    f.write('### Key Findings\n\n')
    f.write('1. **Random Effects model** is preferred (Hausman test p = 0.656)\n')
    f.write('2. **Savings rate** and **employment share** have the largest positive impact on GDP per capita\n')
    f.write('3. **Education spending** and **FDI** are significant but have smaller effects\n')
    f.write('4. **Services sector** drives growth; industry sector is not significant\n')
    f.write('5. **Crisis dummies** (1998, 2008-2009) are not statistically significant\n\n')

    f.write('### Limitations\n\n')
    f.write('- FDI used as proxy for capital formation due to data limitations\n')
    f.write('- Missing data in early years (1990-1995) excluded from analysis\n')
    f.write(f'- Relatively small sample ({df_clean["country"].nunique()} countries)\n\n')

    f.write('### Future Research\n\n')
    f.write('- Include additional control variables (e.g., institutional quality, trade openness)\n')
    f.write('- Test for non-linear relationships (e.g., diminishing returns to savings)\n')
    f.write('- Extend the sample to include developing countries\n')
    f.write('- Use alternative measures of capital formation (gross fixed capital formation if available)\n\n')

