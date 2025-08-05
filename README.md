# Understanding Service Accessibility in Georgia through Mobile Mobility Patterns

Figure 1 traces a three-part workflow.  
[Issue Link](https://github.com/SMIL-AI/GRACE-GA/issues/1#issue-3294442041)  

1. **Data collection and organisation** fuse monthly mobility traces from Dewey Patterns with ACS socio-demographics and NAICS facility categories, producing a longitudinal panel of 72 attributes for each census block group.  
2. **Enhanced visitation-weighted gravity model** combines empirical CBG-to-POI trip shares, Gaussian distance decay and facility capacity to produce annual accessibility surfaces for the state and for four service domains — healthcare, food, education and recreation.  
3. **Explainable machine-learning models** — Random Forest, LightGBM, XGBoost, CatBoost and TabPFN — predict accessibility from the 78 contextual variables; TreeSHAP diagnostics then expose global importance and directional effects, visualised with violin-dot and ranked-bar plots.

---

## Enhanced Visitation-Weighted Gravity Accessibility

As illustrated in Figure 2, our framework extends the classical gravity model by injecting device-observed mobility into every stage of the calculation. Rather than relying on assumed trip counts or ad-hoc attraction coefficients, we begin with anonymised visit records that reveal how each census block group (CBG) actually connects to every point of interest (POI).  

As shown in Figure 2, the visit-share matrix records three attributes for every CBG–POI pair:  
- **L** — the origin-to-destination distance  
- **D** — the POI’s peak-month visit count (a proxy for popularity)  
- **S** — the service category (healthcare, food, education, or entertainment)  

Since these weights are empirical, the model naturally accounts for competition: popular facilities receive larger demand weights, while under-visited sites stand out as potential service gaps. Multiplying this demand surface by each facility’s supply capacity (**S**) and the Gaussian distance-decay kernel yields a competition-aware accessibility score:  

**Aᵥ = S × D × L**
