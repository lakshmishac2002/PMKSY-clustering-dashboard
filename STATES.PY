from flask import Flask, render_template, jsonify
import pandas as pd
import geopandas as gpd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import json
import os

app = Flask(__name__)

# Load and process data (state-wise)
def load_and_process_data_statewise():
    df = pd.read_csv("pmksy-mipms-physical-and-financial-report.csv")

    df = df[
        (df['total_fin_achievements'].notna()) & (df['total_fin_achievements'] > 0) &
        (df['total_fin_target'].notna()) & (df['total_fin_target'] > 0) &
        (df['total_phy_achievements'].notna()) & (df['total_phy_achievements'] > 0) &
        (df['total_phy_target'].notna()) & (df['total_phy_target'] > 0)
    ].copy()

    #Feature scaling
    df['fund_utilization_ratio'] = df['total_fin_achievements'] / df['total_fin_target']
    df['physical_efficiency'] = df['total_phy_achievements'] / df['total_phy_target']
    df['cost_per_hectare'] = df['total_fin_achievements'] / df['total_phy_achievements']

    # Normalize state names
    df['state_name'] = df['state_name'].str.lower().str.strip()

    # Fix mismatches between CSV and GeoJSON
    state_name_mapping = {
        'andaman and nicobar islands': 'andaman & nicobar',
        'andaman & nicobar islands': 'andaman & nicobar',
        'nct of delhi': 'delhi',
        'delhi': 'delhi',
        'pondicherry': 'puducherry',
        'puducherry': 'puducherry',
        'dadra and nagar haveli and daman and diu': 'dadra & nagar haveli',
        'dadra & nagar haveli': 'dadra & nagar haveli',
        'daman and diu': 'daman & diu',
        'daman & diu': 'daman & diu',
        'jammu and kashmir': 'jammu & kashmir',
        'ladakh': 'ladakh',
        'meghalay': 'meghalaya',
        'uttaranchal': 'uttarakhand',
        'orissa': 'odisha',
    }

    df['state_name'] = df['state_name'].replace(state_name_mapping)
    df['state_name'] = df['state_name'].str.lower().str.strip()

    # Group by state
    df_state = df.groupby('state_name')[['fund_utilization_ratio', 'physical_efficiency', 'cost_per_hectare']].mean().reset_index()

    # Standardize and cluster
    features = df_state[['fund_utilization_ratio', 'physical_efficiency', 'cost_per_hectare']]
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
    df_state['cluster'] = kmeans.fit_predict(features_scaled)

    # Add missing states
    missing_states = [
        'chandigarh', 'delhi', 'meghalaya', 'andaman & nicobar', 'lakshadweep',
        'puducherry', 'daman & diu', 'dadra & nagar haveli', 'ladakh', 'jammu & kashmir'
    ]

    existing_states = df_state['state_name'].tolist()
    for state in missing_states:
        if state not in existing_states:
            df_state = pd.concat([
                df_state,
                pd.DataFrame([{
                    'state_name': state,
                    'fund_utilization_ratio': 0,
                    'physical_efficiency': 0,
                    'cost_per_hectare': 0,
                    'cluster': 0
                }])
            ], ignore_index=True)

    return df_state

# Merge with GeoJSON (state-level)
def merge_geojson_with_state_data(df_state):
    geojson_path = os.path.join(os.path.dirname(__file__), "INDIA_STATES.geojson")
    gdf = gpd.read_file(geojson_path)

    print("GeoJSON Columns:", gdf.columns)

    # Find the matching state column
    state_col = None
    for col in gdf.columns:
        if col.lower() in ["state", "state_name", "name_1", "stname"]:
            state_col = col
            break

    if state_col is None:
        raise KeyError("Could not find a matching state column in GeoJSON.")

    # Normalize state names in GeoJSON
    gdf[state_col] = gdf[state_col].str.lower().str.strip()

    # Show unmatched states (if any)
    unmatched = gdf[~gdf[state_col].isin(df_state['state_name'])]
    print("\n⚠️ Unmatched States in GeoJSON:")
    print(unmatched[state_col].unique())

    # Merge GeoJSON with clustered data
    merged = gdf.merge(df_state, how='left', left_on=state_col, right_on='state_name')

    # Select and rename columns
    merged = merged[['geometry', state_col, 'fund_utilization_ratio', 'physical_efficiency', 'cost_per_hectare', 'cluster']]
    merged = merged.rename(columns={state_col: 'state_name'})

    return merged.to_json()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/cluster-data')
def cluster_data():
    df_state = load_and_process_data_statewise()
    merged_geojson = merge_geojson_with_state_data(df_state)
    return jsonify(json.loads(merged_geojson))

if __name__ == '__main__':
    app.run(debug=True)
