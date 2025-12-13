import joblib
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.express as px
import plotly.graph_objects as go
import joblib
from joblib import dump,load
good_model = joblib.load("best_good_investment_pipeline11.joblib")
price_model = joblib.load("xgboost_pipeline11.joblib")

st.write("Good model type:", type(good_model))
st.write("Price model type:", type(price_model))

# -------------------------------
# PAGE TITLE
# -------------------------------
st.title("🏠 Real Estate Investment Analyzer")
st.markdown("Enter property details and find out:")
st.markdown("- **Is this a Good Investment?** (Classification)")
st.markdown("- **Estimated Price after 5 Years** (Regression)")
st.markdown("---")

# -------------------------------
# USER INPUT FORM
# -------------------------------
st.header("📋 Property Details Form")

col1, col2, col3 = st.columns(3)

with col1:
    state = st.selectbox("State", ["Karnataka", "Delhi", "Maharashtra", "Tamil Nadu", "Haryana"])
    city = st.text_input("City")
    locality = st.text_input("Locality")

with col2:
    bhk = st.number_input("BHK", 1, 10)
    size = st.number_input("Size (SqFt)", 200, 10000)
    price = st.number_input("Price (Lakhs)", 1.0, 10000.0)

with col3:
    age = st.number_input("Age of Property (Years)", 0, 50)
    schools = st.number_input("Nearby Schools Count", 0, 20)
    hospitals = st.number_input("Nearby Hospitals Count", 0, 20)

transport = st.selectbox("Public Transport Accessibility", ["Low", "Medium", "High"])
facing = st.selectbox("Facing Direction", ["North", "South", "East", "West"])
furnished = st.selectbox("Furnished Status", ["Furnished", "Semi-Furnished", "Unfurnished"])

# Convert form to a single-row DataFrame
input_data = pd.DataFrame({
    "State": [state],
    "City": [city],
    "Locality": [locality],
    "BHK": [bhk],
    "Size_in_SqFt": [size],
    "Price_in_Lakhs": [price],
    "Age_of_Property": [age],
    "Nearby_Schools": [schools],
    "Nearby_Hospitals": [hospitals],
    "Public_Transport_Accessibility": [transport],
    "Facing": [facing],
    "Furnished_Status": [furnished]
})

st.markdown("---")


# -------------------------------
# PREDICT BUTTON
# -------------------------------
if st.button("🔍 Analyze Property"):
    is_good = good_model.predict(input_data)[0]
    good_prob = good_model.predict_proba(input_data)[0][1]


    # Regression output
    future_price = price_model.predict(input_data)[0]

    st.subheader("📈 Prediction Results")
    
    # Good Investment?
    if is_good == 1:
        st.success(f"✅ **This is a Good Investment** (Confidence: {good_prob:.2f})")
    else:
        st.error(f"❌ **Not a Good Investment** (Confidence: {good_prob:.2f})")

    # Future Price
    st.info(f"💰 **Estimated Price in 5 Years: ₹ {future_price:.2f} Lakhs**")

    st.markdown("---")

    # -------------------------------
    # FEATURE IMPORTANCE (For RF / XGB)
    # -------------------------------
    st.subheader("🔍 Feature Importance")

    try:
        importances = price_model.named_steps["model"].feature_importances_
        feature_names = price_model.named_steps["preprocess"].get_feature_names_out()

        fi_df = pd.DataFrame({
            "Feature": feature_names,
            "Importance": importances
        }).sort_values("Importance", ascending=False).head(20)

        fig_imp = px.bar(fi_df, x="Importance", y="Feature", orientation="h")
        st.plotly_chart(fig_imp)

    except:
        st.warning("Feature importance not available for this model.")

    st.markdown("---")