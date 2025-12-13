import joblib
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.express as px
import plotly.graph_objects as go
import joblib
from joblib import dump,load
import os
import requests


GOOD_MODEL_URL = "https://github.com/<USERNAME>/<REPO>/releases/download/v1.0/best_good_investment_pipelines.joblib"
XGB_MODEL_URL  = "https://github.com/<USERNAME>/<REPO>/releases/download/v1.0/xgboost_pipelines.joblib"

PRICE_MODEL_PATH = "best_good_investment_pipelines.joblib"
XGB_MODEL_PATH   = "xgboost_pipelines.joblib"


@st.cache_resource
def load_price_model():
    if not os.path.exists(PRICE_MODEL_PATH):
        with st.spinner("Downloading Price Prediction Model..."):
            r = requests.get(GOOD_MODEL_URL)
            r.raise_for_status()
            with open(PRICE_MODEL_PATH, "wb") as f:
                f.write(r.content)
    return joblib.load(PRICE_MODEL_PATH)


@st.cache_resource
def load_xgb_model():
    if not os.path.exists(XGB_MODEL_PATH):
        with st.spinner("Downloading XGBoost Model..."):
            r = requests.get(XGB_MODEL_URL)
            r.raise_for_status()
            with open(XGB_MODEL_PATH, "wb") as f:
                f.write(r.content)
    return joblib.load(XGB_MODEL_PATH)



good_model = load_price_model()
xgb_model  = load_xgb_model()