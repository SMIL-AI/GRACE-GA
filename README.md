Understanding Service Accessibility in Georgia through Mobile Mobility Patterns

Figure 1 traces a three-part workflow. 
1. Data collection and organisation fuse monthly mobility traces from Dewey Patterns with ACS socio-demographics and NAICS facility categories , producing a longitudinal panel of 72 attributes for each census block group from 2019 to 2023.
2. An enhanced visitation-weighted gravity model combines empirical CBG-to-POI trip shares, Gaussian distance decay and facility capacity to produce annual accessibility surfaces for the state and for four service domains—healthcare, food, education and recreation.
3. Explainable machine-learning models—Random Forest, LightGBM, XGBoost, CatBoost and TabPFN—predict accessibility from the 78 contextual variables; TreeSHAP diagnostics then expose global importance and directional effects, visualised with violin-dot and ranked-bar plots.
