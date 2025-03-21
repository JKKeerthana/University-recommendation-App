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
@st.cache_data
def load_data():
    return pd.read_csv('data/categorized_specializations.csv')


# ---------------------------
# Download & Extract Models
# ---------------------------
@st.cache_resource
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

        st.success("✅ Models loaded successfully!")
        return models
    except Exception as e:
        st.error(f"❌ Error loading models: {e}")
        return None


# Load models
models = load_models()
if models is None:
    st.error("❌ Failed to load models. Exiting...")
    st.stop()





# Load data
df = load_data()




st.markdown("""
    <style>
        body {
            background-color: #1A1A2E; /* Dark navy blue */
            color: white; /* Light gray text */
        }
        .stApp {
            background-color: #1A1A2E;
        }
        .stMarkdown {
            color: white;
        }
        .stDataFrame, .stTable {
            background-color: #16213E; /* Slightly lighter blue */
            color: white;
            border-radius: 10px;
            padding: 10px;
        }
        .stButton > button {
            background-color: #0F3460; /* Deep blue */
            color: white;
            border-radius: 8px;
            padding: 10px 15px;
            font-size: 16px;
        }
        .stButton > button:hover {
            background-color: #533483; /* Purple hover effect */
        }
        .stSelectbox, .stTextInput, .stNumberInput, .stRadio, .stSlider {
            background-color: #16213E !important;
            color: white !important;
            border-radius: 8px;
        }
    </style>
""", unsafe_allow_html=True)


# ---------------------------
# Preprocessing function
# ---------------------------
def preprocess_input(data, categorical_features, numerical_features, label_encoders, scaler, feature_order, one_hot_columns=None):
    df_input = pd.DataFrame([data])
    # Encode categorical features
    for col in categorical_features:
        if col in label_encoders and df_input[col].iloc[0] in label_encoders[col].classes_:
            df_input[col] = label_encoders[col].transform([df_input[col].iloc[0]])[0]
        else:
            df_input[col] = -1  # Handle missing/unknown values
    # Scale numerical features
    df_input[numerical_features] = scaler.transform(df_input[numerical_features])
    if one_hot_columns:
        df_input = df_input.reindex(columns=feature_order + one_hot_columns, fill_value=0)
        state_val = data.get("univ_state", None)
        state_column = f"univ_state_{state_val}"
        if state_val and state_val != "Select All" and state_column in one_hot_columns:
            df_input[state_column] = 1
    return df_input

# ---------------------------
# Hybrid recommendation for Specialization (unchanged)
# ---------------------------
def hybrid_recommendation(input_df, rf_model, xgb_model, collab_model, svd_model, label_encoder, is_university=False, state=None):
    # Content-based predictions
    rf_probs = rf_model.predict_proba(input_df)[0]
    xgb_probs = xgb_model.predict_proba(input_df)[0]

    # Ensemble: Averaging the probabilities
    content_probs = (rf_probs + xgb_probs) / 2
    
    try:
        transformed_features = svd_model.transform(input_df)
        similar_users = collab_model.kneighbors(transformed_features, return_distance=False)
        collab_probs = np.zeros_like(content_probs)
        # (Using content_model.predict_proba on the same input as a fallback)
        for idx in similar_users.flatten():
            collab_probs += content_model.predict_proba([input_df.iloc[0]])[0]
        collab_probs /= len(similar_users.flatten()) if len(similar_users.flatten()) > 0 else 1
    except Exception as e:
        collab_probs = np.zeros_like(content_probs)
    
    alpha = np.var(content_probs) / (np.var(content_probs) + np.var(collab_probs) + 1e-5)
    final_probs = (content_probs * alpha) + (collab_probs * (1 - alpha))
    
    top_n = 5 if not is_university else 3
    top_indices = np.argsort(final_probs)[-top_n:][::-1]
    
    final_recommendations = label_encoder.inverse_transform(top_indices)
    return final_recommendations

# ---------------------------
# Hybrid University Recommendation (with Collaborative Filtering)
# ---------------------------
def hybrid_university_recommendation(input_df, content_model, collab_model, svd_model, label_encoder,
                          is_university=False, state=None, one_hot_columns=None):
    if is_university:
        # If no particular state is selected ("Select All"), average over all possible state one-hot encodings.
        if state is None and one_hot_columns is not None:
            probs_list = []
            for state_col in one_hot_columns:
                temp_df = input_df.copy()
                # Ensure all state columns are zero then set the current one-hot state to 1.
                temp_df[one_hot_columns] = 0
                temp_df[state_col] = 1
                try:
                    probs = content_model.predict_proba(temp_df)[0]
                except Exception as e:
                    probs = content_model.predict(temp_df)
                probs_list.append(probs)
            content_probs = np.mean(probs_list, axis=0)
        else:
            # If a specific state is selected, use the input_df (which already has the proper one-hot column set)
            try:
                content_probs = content_model.predict_proba(input_df)[0]
            except Exception as e:
                content_probs = content_model.predict(input_df)
        
        # Collaborative Filtering: Use KNN and SVD models to generate collaborative probabilities
        try:
            transformed_features = svd_model.transform(input_df)
            similar_users = collab_model.kneighbors(transformed_features, return_distance=False)
            collab_probs = np.zeros_like(content_probs)
            # Aggregate the collaborative predictions based on similar users
            for idx in similar_users.flatten():
                collab_probs += content_model.predict_proba([input_df.iloc[0]])[0]
            collab_probs /= len(similar_users.flatten()) if len(similar_users.flatten()) > 0 else 1
        except Exception as e:
            collab_probs = np.zeros_like(content_probs)
        
        # Combine content-based and collaborative-based predictions
        alpha = np.var(content_probs) / (np.var(content_probs) + np.var(collab_probs) + 1e-5)
        final_probs = (content_probs * alpha) + (collab_probs * (1 - alpha))
        
        # Select top N universities
        top_n = 3  # Recommend top 3 universities
        
        top_indices = np.argsort(final_probs)[-top_n:][::-1]
        final_recommendations = label_encoder.inverse_transform(top_indices)
        
        # If a specific state was requested, filter the recommendations accordingly.
        if state is not None:
            filtered_recommendations = [rec for rec in final_recommendations 
                                        if rec in df[df['univ_state'] == state]['univName'].values]
            if len(filtered_recommendations) < top_n:
                st.warning(f"⚠️ Only {len(filtered_recommendations)} universities found in {state}. Expanding recommendations.")
                filtered_recommendations = final_recommendations
            return filtered_recommendations
        
        return final_recommendations

# Helper function to compute aggregated statistics for a given specialization
def display_specialization_stats(specialization, df):
    subset = df[df['specialization_category'] == specialization]
    if subset.empty:
        return None
    stats = {
        "🔢 Number of Students": subset.shape[0],
        "📊 Avg Normalized CGPA": round(subset['normalized_cgpa'].mean(), 2),
        "📝 Avg TOEFL Score": round(subset['toeflScore'].mean(), 2),
        "🗣 Avg GRE Verbal": round(subset['greV'].mean(), 2),
        "📈 Avg GRE Quant": round(subset['greQ'].mean(), 2),
        "📉 Avg GRE Analytical": round(subset['greA'].mean(), 2),
#        "🔬 Avg Research Exp (yrs)": round(subset['researchExp'].mean(), 2),
#        "🏭 Avg Industry Exp (yrs)": round(subset['industryExp'].mean(), 2),
#        "💼 Avg Internship Exp (yrs)": round(subset['internExp'].mean(), 2)
    }
    return pd.DataFrame(stats.items(), columns=["Metric", "Value"])

# ---------------------------
# Streamlit UI
# ---------------------------
st.title("🎓 University & Major Recommender System")
# Navigation Sidebar
page = st.sidebar.radio("Select a Page", ["Home", "Major Recommendation", "University Recommendation"])

if page == "Home":
    st.header("Welcome to the Recommender System")
    st.write("""
    This system helps you find the most suitable **courses** and **universities** for your academic and professional background.
    
    - Go to **Course Recommendation** to find potential academic majors.
    - Go to **University Recommendation** to find the best universities based on your profile.
    """)

# Streamlit UI for Course Recommendation
elif page == "Major Recommendation":
    st.header("📌 Major Recommendation")
    user_input = {
        "major": st.selectbox("Major", sorted(df['major'].unique())),
        "researchExp": st.number_input("Research Experience (years)", min_value=0.0, step=0.1),
        "industryExp": st.number_input("Industry Experience (years)", min_value=0.0, step=0.1),
        "toeflScore": st.number_input("TOEFL Score", min_value=0, max_value=120),
        "internExp": st.number_input("Internship Experience (years)", min_value=0.0, step=0.1),
        "greV": st.number_input("GRE Verbal Score", min_value=130, max_value=170),
        "greQ": st.number_input("GRE Quantitative Score", min_value=130, max_value=170),
        "greA": st.number_input("GRE Analytical Score", min_value=0.0, max_value=6.0, step=0.1),
        "normalized_cgpa": st.number_input("Normalized CGPA", min_value=0.0, max_value=10.0, step=0.1),
    }
    feature_order_spec = ['major', 'researchExp', 'industryExp', 'toeflScore', 'internExp', 'greV', 'greQ', 'greA', 'normalized_cgpa']
    input_df_spec = preprocess_input(
        user_input, 
        categorical_features=['major'], 
        numerical_features=['researchExp', 'industryExp', 'toeflScore', 'internExp', 'greV', 'greQ', 'greA', 'normalized_cgpa'],
        label_encoders=models['label_encoders_specialization'], 
        scaler=models['scaler_specialization'], 
        feature_order=feature_order_spec
    )
    
    if st.button("🔍 Recommend Major"):
        recs = hybrid_recommendation(
            input_df_spec, 
            models['rf_specialization'][0], 
            models['xgb_specialization'],
            models['knn_specialization'], 
            models['svd_specialization'],
            models['le_y_specialization']
        )
        st.success("🎯 Recommended Majors:")
        for r in recs:
            st.markdown(f"#### {r.upper()} 🚀")
            stats_df = display_specialization_stats(r, df)
            if stats_df is not None:
                st.table(stats_df)
            else:
                st.info("ℹ️ No additional details available.")



elif page == "University Recommendation":
    st.header("🏫 University Recommendation")
    user_input = {
        "ugCollege": st.selectbox("Undergraduate College", sorted(df['ugCollege'].dropna().unique())),
        "univ_state": st.selectbox("Preferred University State", ["Select All"] + sorted(df['univ_state'].unique())),
        "toeflScore": st.number_input("TOEFL Score", min_value=0, max_value=120),
        "greV": st.number_input("GRE Verbal Score", min_value=130, max_value=170),
        "greQ": st.number_input("GRE Quantitative Score", min_value=130, max_value=170),
        "greA": st.number_input("GRE Analytical Score", min_value=0.0, max_value=6.0, step=0.1),
        "normalized_cgpa": st.number_input("Normalized CGPA", min_value=0.0, max_value=10.0, step=0.1),
        "specialization_category": st.selectbox("Intended Specialization", sorted(df['specialization_category'].dropna().unique())),
    }
    feature_order_univ = ['ugCollege', 'specialization_category', 'toeflScore', 'greV', 'greQ', 'greA', 'normalized_cgpa']
    input_df_univ = preprocess_input(
        user_input, 
        categorical_features=['ugCollege', 'specialization_category'], 
        numerical_features=['toeflScore', 'greV', 'greQ', 'greA', 'normalized_cgpa'], 
        label_encoders=models['label_encoders_university'], 
        scaler=models['scaler_university'], 
        feature_order=feature_order_univ, 
        one_hot_columns=models['one_hot_columns_university']
    )
    
    if st.button("🔍 Recommend Universities"):
        # Extract the Random Forest model from the loaded dictionary.
        # (In training, rf_university.pkl was saved as a dict with key "Random Forest".)
        rf_university_model = models["rf_university"]["Random Forest"][0]
        
        # If user selects "Select All", pass state as None.
        selected_state = user_input['univ_state'] if user_input['univ_state'] != "Select All" else None
        
        recommendations = hybrid_university_recommendation(
            input_df_univ, 
            rf_university_model, 
            models['knn_university'], 
            models['svd_university'], 
            models['le_y_university'], 
            is_university=True, 
            state=selected_state,
            one_hot_columns=models['one_hot_columns_university']
        )
        
        # Prepare a dataframe for better display
        university_data_list = []
        
        for rec in recommendations:
            university_data = df[df['univName'] == rec]
    
            if not university_data.empty:
                # Extract the rank and acceptance rate for the current university
                university_rank = university_data['univName_rank'].values[0]
                acceptance_rate = university_data['acceptance_rate'].values[0]
                
                # Append the data to the list
                university_data_list.append({
                    'University': rec.upper(),
                    '📊 World University Ranking': university_rank,
                    '📉 Acceptance Rate': f"{acceptance_rate}%",
                })
            else:
                # If data is missing, show a warning
                st.warning(f"Details for {rec} not found.")
        
        # Convert the list of university data into a pandas DataFrame
        if university_data_list:
            university_df = pd.DataFrame(university_data_list)
            # Reset the index and start it from 1 instead of 0
            university_df.index = university_df.index + 1  
            st.table(university_df)
        else:
            st.warning("No recommendations found.")
