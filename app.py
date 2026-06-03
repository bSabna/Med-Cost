import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load the model and encoding artifacts
artifacts = joblib.load('svr_production_model.joblib')
pipeline = artifacts['pipeline']
target_mean_map = artifacts['target_mean_map']
overall_train_mean = artifacts['overall_train_mean']

st.title("Drug Cost Prediction App")
st.write("Enter the metrics below to estimate the total drug cost.")

# 1. User Input Fields
brnd_name = st.text_input("Brand Name", value="MyDrug")
tot_clms = st.number_input("Total Claims", min_value=0, value=10)
tot_30day_fills = st.number_input("Total 30-Day Fills", min_value=0.0, value=10.0)
tot_day_suply = st.number_input("Total Day Supply", min_value=0, value=300)
tot_benes = st.number_input("Total Beneficiaries", min_value=0.0, value=5.0)

if st.button("Predict Cost"):
    # 2. Replicate your custom Target Encoding
    # Look up the brand name in our saved training map; fallback to overall mean if brand is new
    encoded_brand = target_mean_map.get(brnd_name, overall_train_mean)
    
    # 3. Structure input data exactly like X_train/X_test columns
    input_data = pd.DataFrame([{
        'Tot_Clms': tot_clms,
        'Tot_30day_Fills': tot_30day_fills,
        'Tot_Day_Suply': tot_day_suply,
        'Tot_Benes': tot_benes,
        'Brnd_Name_Encoded': encoded_brand
    }])
    
    # 4. Predict (Returns log-transformed value)
    log_prediction = pipeline.predict(input_data)[0]
    
    # 5. Reverse the log1p transformation: exp(x) - 1
    final_cost = np.expm1(log_prediction)
    
    st.success(f"Estimated Total Drug Cost: ${final_cost:,.2f}")
