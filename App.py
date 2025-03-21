#!/usr/bin/env python
# coding: utf-8

# 
import streamlit as st
import joblib
import os
import zipfile
import gdown
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import TruncatedSVD

# ---------------------------
# Load Data
# ---------------------------
#@st.cache_data
def load_data():
    return pd.read_csv('data/categorized_specializations.csv')


# ---------------------------
# Download & Extract Models
# ---------------------------
#@st.cache_resource
def download_and_extract_models():
    model_zip_path = "MODELS.zip"
    extract_to_path = "MODELS"
    gdrive_file_id = "1TgULdbzFn9_MMfbFO4S_nMyEgn7HVrzI"

    # Check if models already exist
    if os.path.exists(extract_to_path) and os.path.isdir(extract_to_path):
        return extract_to_path

    # Download the model zip file
    st.info("📥 Downloading model files from Google Drive...")
    url = f"https://drive.google.com/uc?export=download&id={gdrive_file_id}"
    gdown.download(url, model_zip_path, quiet=False)

    # Extract models
    st.info("📂 Extracting models...")
    try:
        with zipfile.ZipFile(model_zip_path, "r") as zip_ref:
            zip_ref.extractall(extract_to_path)
        os.remove(model_zip_path)  # Cleanup
        st.success("✅ Model extraction complete!")
    except zipfile.BadZipFile:
        st.error("❌ Invalid ZIP file. Please check the Google Drive file.")
        return None

    return extract_to_path


# ---------------------------
# Load Models
# ---------------------------
#@st.cache_resource
def load_models():
    model_dir = download_and_extract_models()
    if model_dir is None:
        st.error("❌ Models directory not found.")
        return None

    major_models_path = os.path.join(model_dir, "MODELS/major_models")
    university_models_path = os.path.join(model_dir, "MODELS/university_models")

    required_files = [
        ("rf_specialization.pkl", major_models_path),
        ("rf_university.pkl", university_models_path),
        ("xgb_specialization.json", major_models_path),
        ("label_encoders_specialization.pkl", major_models_path),
        ("label_encoders_university.pkl", university_models_path),
        ("scaler_specialization.pkl", major_models_path),
        ("scaler_university.pkl", university_models_path),
        ("svd_specialization.pkl", major_models_path),
        ("knn_specialization.pkl", major_models_path),
        ("svd_univ.pkl", university_models_path),
        ("knn_univ.pkl", university_models_path),
        ("le_y_spec.pkl", major_models_path),
        ("le_y_univ.pkl", university_models_path),
        ("one_hot_columns_university.pkl", university_models_path),
    ]

    # Verify all required model files exist
    missing_files = [f"{path}/{file}" for file, path in required_files if not os.path.exists(os.path.join(path, file))]
    if missing_files:
        st.error(f"❌ Missing model files: {missing_files}")
        return None

    try:
        models = {
            "rf_specialization": joblib.load(os.path.join(major_models_path, "rf_specialization.pkl")),
            "rf_university": joblib.load(os.path.join(university_models_path, "rf_university.pkl")),
            "xgb_specialization": xgb.XGBClassifier(),
            "label_encoders_specialization": joblib.load(os.path.join(major_models_path, "label_encoders_specialization.pkl")),
            "label_encoders_university": joblib.load(os.path.join(university_models_path, "label_encoders_university.pkl")),
            "scaler_specialization": joblib.load(os.path.join(major_models_path, "scaler_specialization.pkl")),
            "scaler_university": joblib.load(os.path.join(university_models_path, "scaler_university.pkl")),
            "svd_specialization": joblib.load(os.path.join(major_models_path, "svd_specialization.pkl")),
            "knn_specialization": joblib.load(os.path.join(major_models_path, "knn_specialization.pkl")),
            "svd_university": joblib.load(os.path.join(university_models_path, "svd_univ.pkl")),
            "knn_university": joblib.load(os.path.join(university_models_path, "knn_univ.pkl")),
            "le_y_specialization": joblib.load(os.path.join(major_models_path, "le_y_spec.pkl")),
            "le_y_university": joblib.load(os.path.join(university_models_path, "le_y_univ.pkl")),
            "one_hot_columns_university": joblib.load(os.path.join(university_models_path, "one_hot_columns_university.pkl")),
        }

        # Load XGBoost model separately
        models["xgb_specialization"].load_model(os.path.join(major_models_path, "xgb_specialization.json"))

        st.write("Models loaded successfully!")  # Check if this prints
    except Exception as e:
        st.write(f"Error loading model: {str(e)}")  # Catch and display the error
        return None


# Load models
models = load_models()
if models is None:
    st.error("❌ Failed to load models. Exiting...")
    st.stop()

