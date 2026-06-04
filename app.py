import streamlit as st
import pandas as pd
import numpy as np
import joblib

# 1. Load the bundled model artifacts
artifacts = joblib.load('svr_production_model.joblib')
model = artifacts['model']
brand_map = artifacts['target_mean_map']
fallback_mean = artifacts['overall_train_mean']

st.set_page_config(page_title="Medicare Cost Predictor", layout="centered")
st.title("Medicare Part D Drug Cost Predictor")
st.write("Select a real drug brand and input provider metrics to forecast total expenditures.")

# 2. Dynamic Dropdown Menu for Real Brands
sorted_brands = sorted(list(brand_map.keys()))
selected_brand = st.selectbox(" Select Prescription Drug Brand Name:", sorted_brands)

st.divider()

# 3. Form fields for provider volume input
st.subheader(" Provider Volume Metrics")
tot_clms = st.number_input("Total Claims", min_value=0, value=100, step=1)
tot_30day_fills = st.number_input("Total 30-Day Fills", min_value=0, value=120, step=1)
tot_day_suply = st.number_input("Total Day Supply", min_value=0, value=3000, step=1)
tot_benes = st.number_input("Total Beneficiaries (Unique Patients)", min_value=0, value=50, step=1)

if st.button(" Calculate Predicted Cost", type="primary"):
    # 4. Extract the historical target-encoded score for the chosen brand
    encoded_brand_value = brand_map.get(selected_brand, fallback_mean)
    
    # 5. Format inputs to mirror the training dataset layout exactly
    input_data = pd.DataFrame([{
        'Tot_Clms': tot_clms,
        'Tot_30day_Fills': tot_30day_fills,
        'Tot_Day_Suply': tot_day_suply,
        'Tot_Benes': tot_benes,
        'Brnd_Name_Encoded': encoded_brand_value
    }])
    
    # 6. Run prediction and invert the log transformation
    predicted_log_cost = model.predict(input_data)[0]
    predicted_real_cost = np.expm1(predicted_log_cost)
    
    # Display performance result cleanly
    st.success(f"**Predicted Total Cost for {selected_brand}:** ${predicted_real_cost:,.2f}")