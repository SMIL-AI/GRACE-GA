"""
Data Collection & Organisation for GA POI Mobility Data (2019)
Steps:
1. Load raw POI Monthly Patterns (2019)
2. Aggregate by PLACEKEY
3. Merge VISITOR_HOME_CBGS JSONs
4. Expand POI-CBG relationships
5. Load 2019 CBG shapefile & compute centroids
6. Calculate POI-CBG distances
7. Map NAICS categories to service domains
8. Merge with ACS socio-demographics (population)
9. Compute annual visits
10. Save final merged dataset
"""

import os
import json
import pickle
import pandas as pd
import geopandas as gpd
from collections import defaultdict

# -------------------------
# Config
# -------------------------
BASE_PATH = "./sample_data"
POI_FILE = os.path.join(BASE_PATH, "Monthly_Patterns_Foot_Traffic_GA-2019.pkl")
CATEGORY_FILE = os.path.join(BASE_PATH, "category_to_label.txt")
CBG_SHP = os.path.join(BASE_PATH, "tl_2019_13_bg.shp")
ACS_FILE = os.path.join(BASE_PATH, "acs_2019.csv")
OUTPUT_FILE = os.path.join(BASE_PATH, "merged_data_2019.pkl")

# -------------------------
# Helpers
# -------------------------
def load_pickle(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")
    with open(path, "rb") as f:
        return pickle.load(f)

def load_category_mapping(path):
    with open(path, encoding="utf-8") as f:
        return dict(line.strip().split(": ") for line in f if ": " in line)

def merge_json_objects(json_str):
    """Merge multiple JSON objects from VISITOR_HOME_CBGS into one dict."""
    merged_dict = defaultdict(int)
    if pd.isna(json_str):
        return {}
    if isinstance(json_str, dict):
        return json_str
    try:
        json_str = json_str.replace('""', '"').replace("'", '"')
        data = json.loads(json_str)
        if isinstance(data, dict):
            for k, v in data.items():
                merged_dict[k] += v
    except json.JSONDecodeError:
        pass
    return dict(merged_dict)

# -------------------------
# Pipeline
# -------------------------
if __name__ == "__main__":
    print("1. Loading raw 2019 POI data...")
    poi_df = load_pickle(POI_FILE)

    print("2. Aggregating by PLACEKEY...")
    aggregated_df = poi_df.groupby('PLACEKEY').agg({
        'RAW_VISIT_COUNTS': 'max',
        'VISITOR_HOME_CBGS': lambda x: ','.join(x.dropna().astype(str)),
        'LATITUDE': 'first',
        'LONGITUDE': 'first',
        'TOP_CATEGORY': 'first',
        'SUB_CATEGORY': 'first',
        'POI_CBG': 'first'
    }).reset_index()

    print("3. Merging VISITOR_HOME_CBGS JSONs...")
    aggregated_df['VISITOR_HOME_CBGS'] = aggregated_df['VISITOR_HOME_CBGS'].fillna('{}').apply(merge_json_objects)

    print("4. Expanding POI–CBG relationships...")
    expanded_rows = []
    for _, row in aggregated_df.iterrows():
        for cbg_id, visits in row['VISITOR_HOME_CBGS'].items():
            expanded_rows.append({
                'PLACEKEY': row['PLACEKEY'],
                'LONGITUDE': row['LONGITUDE'],
                'LATITUDE': row['LATITUDE'],
                'CBG_ID': str(cbg_id),
                'Vij': visits
            })
    poi_cbg_df = pd.DataFrame(expanded_rows)

    print("5. Loading CBG shapefile and computing centroids...")
    cbg_gdf = gpd.read_file(CBG_SHP).to_crs('EPSG:32616')
    cbg_gdf['CBG_ID'] = cbg_gdf['GEOID'].astype(str)
    cbg_gdf['geometry_center'] = cbg_gdf.geometry.centroid
    cbg_centers = cbg_gdf[['CBG_ID', 'geometry_center']]

    print("6. Calculating POI–CBG distances...")
    poi_gdf = gpd.GeoDataFrame(poi_cbg_df, geometry=gpd.points_from_xy(
        poi_cbg_df['LONGITUDE'], poi_cbg_df['LATITUDE']
    ), crs='EPSG:4326').to_crs('EPSG:32616')
    poi_with_centers = poi_gdf.merge(cbg_centers, on='CBG_ID', how='left')
    poi_with_centers['Distance'] = poi_with_centers.geometry.distance(poi_with_centers['geometry_center'])

    print("7. Mapping NAICS categories to service domains...")
    category_map = load_category_mapping(CATEGORY_FILE)
    aggregated_df['keyword_label'] = aggregated_df['TOP_CATEGORY'].map(category_map).fillna("other")

    print("8. Merging POI–CBG distance data with POI metadata...")
    merged_df = poi_with_centers.merge(
        aggregated_df[['PLACEKEY', 'RAW_VISIT_COUNTS', 'keyword_label']],
        on='PLACEKEY', how='left'
    )

    print("9. Merging with ACS socio-demographic data...")
    pop_df = pd.read_csv(ACS_FILE, usecols=["CBG_ID", "Total Population"])
    pop_df['CBG_ID'] = pop_df['CBG_ID'].astype(str)
    merged_df = merged_df.merge(pop_df, on="CBG_ID", how="left")

    print("10. Computing annual visits...")
    annual_visits_df = (
        merged_df.groupby("PLACEKEY", as_index=False)["Vij"]
        .sum()
        .rename(columns={"Vij": "AnnualVisits"})
    )
    merged_df = merged_df.merge(annual_visits_df, on="PLACEKEY", how="left")

    print(f"11. Saving final dataset → {OUTPUT_FILE}")
    merged_df.to_pickle(OUTPUT_FILE)

    print(f"✅ Done. {len(merged_df)} rows saved.")
