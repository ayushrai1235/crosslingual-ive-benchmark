# Statistical Analysis & Inferential Architecture

## 1. Primary Inferential Hypothesis Testing
Because the research question centers on whether moral allocation bias shifts across languages, the primary statistical tests evaluate the **Condition $\times$ Language** interaction:

1. **Omnibus Non-Parametric Friedman Test**: Evaluates whether IVE ranks differ significantly across English, Hindi, and Spanish.
2. **Paired Wilcoxon Signed-Rank Tests**: Evaluates pairwise contrasts ($\Delta_{hi-en}$, $\Delta_{es-en}$, $\Delta_{es-hi}$).
3. **Multiple Testing Adjustments**:
   - **Holm-Bonferroni step-down** procedure (controlling Family-Wise Error Rate, FWER $\le 0.05$).
   - **Benjamini-Hochberg procedure** (controlling False Discovery Rate, FDR $\le 0.05$).

## 2. Scenario-Clustered Bootstrapping ($B=10,000$)
To account for within-scenario correlations across languages and models without relying on parametric normality assumptions:
- Entire scenario clusters $s \in \{1, \dots, S\}$ are resampled with replacement $B = 10,000$ times.
- 95% Percentile and Bias-Corrected & Accelerated (BCa) confidence intervals are computed for all point estimates and contrast differences.

## 3. Secondary / Exploratory Analysis: Linear Mixed-Effects Model
To explore variance decomposition across judge models and scenarios:
$$\text{Allocation}_{m,s,l} = \beta_0 + \beta_1 \cdot \text{Condition} + \beta_2 \cdot \text{Language} + \beta_3 \cdot (\text{Condition} \times \text{Language}) + u_{m} + v_{s} + \epsilon_{m,s,l}$$
where $u_m \sim \mathcal{N}(0, \sigma_m^2)$ is the random intercept for model family, and $v_s \sim \mathcal{N}(0, \sigma_s^2)$ is the random intercept for scenario.

## 4. Scientific Non-Fabrication Commitment
- The benchmark never simulates, fabricates, or hardcodes empirical results.
- If empirical judgment logs are absent, all analytical scripts and dashboards report **"Results pending"**.
