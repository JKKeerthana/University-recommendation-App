#!/usr/bin/env python
# coding: utf-8

# In[1]:


import streamlit as st
import joblib
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import TruncatedSVD
import requests, zipfile, io, os
from memory_cf import MemoryBasedCF  # Make sure this file exists

# ------------------------------
# Load CSV data
# ------------------------------
@st.cache_data
def load_data():
    return pd.read_csv("data/categorized_specializations.csv")

df = load_data()

# ------------------------------
# Load specialization models
# ------------------------------
@st.cache_resource
def load_specialization_models():
    try:
        models = {
            "rf_specialization": joblib.load("major_models_compressed/rf_specialization.pkl")[0],
            "xgb_specialization": xgb.XGBClassifier(),
            "label_encoders_specialization": joblib.load("major_models_compressed/label_encoders_specialization.pkl"),
            "scaler_specialization": joblib.load("major_models_compressed/scaler_specialization.pkl"),
            "cf_model_spec": joblib.load("major_models_compressed/cf_model_spec.pkl"),
            "le_y_specialization": joblib.load("major_models_compressed/le_y_spec.pkl"),
        }
        models["xgb_specialization"].load_model("major_models_compressed/xgb_specialization.json")
        return models
    except Exception as e:
        st.error(f"Error loading specialization models: {e}")
        return None

# ------------------------------
# Load university models
# ------------------------------
@st.cache_resource
def load_university_models():
    try:
        models = {
            "rf_university": joblib.load("university_models_compressed/rf_university.pkl"),
            "label_encoders_university": joblib.load("university_models_compressed/label_encoders_university.pkl"),
            "scaler_university": joblib.load("university_models_compressed/scaler_university.pkl"),
            "cf_model_univ": joblib.load("university_models_compressed/cf_model_memory.pkl"),
            "le_y_university": joblib.load("university_models_compressed/le_y_univ.pkl"),
            "one_hot_columns_university": joblib.load("university_models_compressed/one_hot_columns_university.pkl"),
        }
        return models
    except Exception as e:
        st.error(f"Error loading university models: {e}")
        return None

page = st.sidebar.radio("Select a Page", ["Home", "Major Recommendation", "University Recommendation"])

#models = None
#if page == "Major Recommendation":
#    models = load_specialization_models()
#elif page == "University Recommendation":
#    models = load_university_models()



st.markdown("""
    <style>
        .stApp {
            background-color: #f0f4f8;
            color: #212529;
            font-family: 'Segoe UI', sans-serif;
        }

        h1, h2, h3, h4, h5, h6 {
            color: #1a1a2e;
        }

        .stMarkdown p {
            color: #343a40;
        }

        .css-6qob1r {
            background-color: #ffffff !important;
            color: #212529;
            border-right: 1px solid #dee2e6;
        }

        .stSelectbox, .stTextInput, .stNumberInput, .stSlider, .stRadio {
            background-color: #e9f0f5 !important;
            color: #212529 !important;
            border-radius: 10px;
            border: 1px solid #cbd5e1;
        }

        .stButton > button {
            background-color: #3b82f6;
            color: white;
            font-weight: 500;
            border-radius: 10px;
            border: none;
            padding: 10px 20px;
        }

        .stButton > button:hover {
            background-color: #2563eb;
        }

        .stDataFrame, .stTable {
            background-color: white;
            color: #212529;
            border-radius: 10px;
            box-shadow: 0 0 10px rgba(0,0,0,0.05);
        }

        .block-container {
            padding: 2rem 3rem;
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
def hybrid_recommendation(input_df, rf_model, xgb_model, cf_model, label_encoder):
    input_features = input_df.drop(columns=['userName'], errors='ignore')

    rf_probs = rf_model.predict_proba(input_features)[0]
    xgb_probs = xgb_model.predict_proba(input_features)[0]
    content_probs = (rf_probs + xgb_probs) / 2

    collab_probs = np.zeros_like(content_probs)

    if 'userName' in input_df.columns:
        input_user = input_df.iloc[0]['userName']
        cf_raw_probs = cf_model.predict(input_user)

        # Align CF output to label encoder classes
        cf_items = cf_model.item_ids  # item_ids should be a list of labels used in training
        cf_prob_dict = dict(zip(cf_items, cf_raw_probs))

        aligned_cf_probs = [cf_prob_dict.get(cls, 0.0) for cls in label_encoder.classes_]
        collab_probs = np.array(aligned_cf_probs)

    # Variance-based fusion
    alpha = np.var(content_probs) / (np.var(content_probs) + np.var(collab_probs) + 1e-5)
    final_probs = (content_probs * alpha) + (collab_probs * (1 - alpha))

    top_n = 5
    top_indices = np.argsort(final_probs)[-top_n:][::-1]
    recommendations = label_encoder.inverse_transform(top_indices)
    return recommendations


# ---------------------------
# Hybrid University Recommendation (with Collaborative Filtering)
# ---------------------------
def hybrid_university_recommendation(input_df, content_model, collab_model, label_encoder,
                                     is_university=False, state=None, one_hot_columns=None):
    if is_university:
        if state is None and one_hot_columns is not None:
            probs_list = []
            for state_col in one_hot_columns:
                temp_df = input_df.copy()
                temp_df[one_hot_columns] = 0
                temp_df[state_col] = 1
                try:
                    probs = content_model.predict_proba(temp_df)[0]
                except Exception:
                    probs = content_model.predict(temp_df)
                probs_list.append(probs)
            content_probs = np.mean(probs_list, axis=0)
        else:
            try:
                content_probs = content_model.predict_proba(input_df)[0]
            except Exception:
                content_probs = content_model.predict(input_df)
    else:
        content_probs = content_model.predict_proba(input_df)[0]

    # Collaborative Filtering
    collab_probs = np.zeros_like(content_probs)
    if 'userName' in input_df.columns:
        input_user = input_df.iloc[0]['userName']
        cf_raw_probs = collab_model.predict(input_user)
        cf_items = collab_model.item_ids
        cf_prob_dict = dict(zip(cf_items, cf_raw_probs))
        aligned_cf_probs = [cf_prob_dict.get(cls, 0.0) for cls in label_encoder.classes_]
        collab_probs = np.array(aligned_cf_probs)

    alpha = np.var(content_probs) / (np.var(content_probs) + np.var(collab_probs) + 1e-5)
    final_probs = (content_probs * alpha) + (collab_probs * (1 - alpha))

    top_n = 3
    top_indices = np.argsort(final_probs)[-top_n:][::-1]
    final_recommendations = label_encoder.inverse_transform(top_indices)

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
#page = st.sidebar.radio("Select a Page", ["Home", "Major Recommendation", "University Recommendation"])

if page == "Home":
    st.header("Welcome to the Recommender System")
    st.write("""
    This system helps you find the most suitable *courses* and *universities* for your academic and professional background.
    
    - Go to *Course Recommendation* to find potential academic majors.
    - Go to *University Recommendation* to find the best universities based on your profile.
    """)

# Streamlit UI for Course Recommendation
elif page == "Major Recommendation":
    models = load_specialization_models()
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
            models['cf_model_spec'],  # <-- Use CF model
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
    models = load_university_models()
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
            models['cf_model_univ'],  # <-- Use CF model
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
