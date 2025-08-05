# Understanding Service Accessibility in Georgia through Mobile Mobility Patterns

Our framework is shown in the figure below.  
![screenshot](https://github.com/SMIL-AI/GRACE-GA/blob/main/pics/Fig1.png)

1. **Data collection and organisation** fuse monthly mobility traces from Dewey Patterns with ACS socio-demographics and NAICS facility categories, producing a longitudinal panel of 72 attributes for each census block group.  
2. **Enhanced visitation-weighted gravity model** combines empirical CBG-to-POI trip shares, Gaussian distance decay and facility capacity to produce annual accessibility surfaces for the state and for four service domains — healthcare, food, education and recreation.
3. **Spatiotemporal accessibility mapping** uses Local Moran’s $I$ to track year-over-year change, delineating high–high and low–low clusters and benchmarking pre-pandemic, pandemic and recovery phases.
4. **Explainable machine-learning models** — Random Forest, LightGBM, XGBoost, CatBoost and TabPFN — predict accessibility from the 78 contextual variables; TreeSHAP diagnostics then expose global importance and directional effects, visualised with violin-dot and ranked-bar plots.


## 1.Data Collection and Organization

We use the **Advan Monthly Patterns** dataset, which provides visitor and demographic aggregations for points of interest (POIs) across the U.S. on a monthly basis.  
For this project, we filter the data to include only **Georgia** and retain key columns relevant to mobility analysis ([source](10.82551/beb1-2831)).

Neighbourhood-level socio-demographic features were sourced from the **American Community Survey (ACS)** for each year from 2019 to 2023 ([source](http://api.census.gov/data/2022/acs/acs5)).

---

## 2.Enhanced Visitation-Weighted Gravity Accessibility
![screenshot](https://github.com/SMIL-AI/GRACE-GA/blob/main/pics/Fig2.png)
As illustrated in the figure above, our framework extends the classical gravity model by injecting device-observed mobility into every stage of the calculation. Rather than relying on assumed trip counts or ad-hoc attraction coefficients, we begin with anonymised visit records that reveal how each census block group (CBG) actually connects to every point of interest (POI).  

As shown in Figure 2, the visit-share matrix records three attributes for every CBG–POI pair:  
- **L** — the origin-to-destination distance  
- **D** — the POI’s peak-month visit count (a proxy for popularity)  
- **S** — the service category (healthcare, food, education, or entertainment)  

Since these weights are empirical, the model naturally accounts for competition: popular facilities receive larger demand weights, while under-visited sites stand out as potential service gaps. Multiplying this demand surface by each facility’s supply capacity (**S**) and the Gaussian distance-decay kernel yields a competition-aware accessibility score:  

**Aᵥ = S × D × L**

---

## 3.Spatial accessibility patterns
![screenshot](https://github.com/SMIL-AI/GRACE-GA/blob/main/pics/Fig3.png)

Local Moran’s \(I\) exposes where gains in accessibility have been spatially uneven. Figures above show relatively clusters and outliers for overall accessibility and food accessibility of 2019 and 2023.

---

## 4.Model Training and Interpretation

We modeled CBG-level accessibility across three pandemic phases using four tree-ensemble regressors: Random Forest, LightGBM, XGBoost, and CatBoost.  
All models used the same 72 socio-demographic and transport features, with median-imputation for missing values and label encoding where needed.

The best configuration (highest mean cross-validated R²) for each model was retrained on the full dataset and saved for SHAP-based interpretation.  
A second 5-fold cross-validation produced out-of-fold predictions to compute **R²**, **MSE**, and **MAE**.

For interpretation, we applied **SHAP (TreeSHAP)**, which assigns each feature a Shapley value indicating its contribution to a prediction. This ensures local accuracy by decomposing each prediction into the sum of feature contributions plus a baseline.
