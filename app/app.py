import streamlit as st
import pandas as pd
import joblib
import numpy as np

# ------------------------------------
# Load Data & Models
# ------------------------------------

master_df = pd.read_csv("../data/processed/master_cleaned.csv")

clf = joblib.load("../models/visit_mode_model.pkl")
le = joblib.load("../models/label_encoder.pkl")

# ------------------------------------
# Page Title
# ------------------------------------

st.title("🌍 Tourism Experience Recommendation System")

st.write("Get personalized attraction recommendations and visit mode prediction.")

# ------------------------------------
# User Input Section
# ------------------------------------

st.sidebar.header("Enter User Details")

continent = st.sidebar.selectbox(
    "Select Continent",
    master_df["Continent"].dropna().unique()
)

country = st.sidebar.selectbox(
    "Select Country",
    master_df["Country"].dropna().unique()
)

region = st.sidebar.selectbox(
    "Select Region",
    master_df["Region"].dropna().unique()
)

attraction_type = st.sidebar.selectbox(
    "Preferred Attraction Type",
    master_df["AttractionType"].dropna().unique()
)

# ------------------------------------
# Predict Visit Mode
# ------------------------------------

if st.sidebar.button("Predict Visit Mode & Recommend"):

    # Create input dataframe
    input_data = pd.DataFrame({
        "Continent": [continent],
        "Country": [country],
        "Region": [region],
        "AttractionType": [attraction_type]
    })

    # One-hot encode same way as training
    X_input = pd.get_dummies(input_data)
    X_input = X_input.reindex(columns=clf.feature_names_in_, fill_value=0)

    prediction = clf.predict(X_input)
    visit_mode = le.inverse_transform(prediction)

    st.subheader("🎯 Predicted Visit Mode:")
    st.success(visit_mode[0])

    # ------------------------------------
    # Simple Content-Based Recommendation
    # ------------------------------------

    recommendations = master_df[
        (master_df["AttractionType"] == attraction_type) &
        (master_df["Country"] == country)
    ][["Attraction", "Region"]].drop_duplicates().head(5)

    st.subheader("⭐ Recommended Attractions:")
    st.dataframe(recommendations)

    # ------------------------------------
    # Popular Attractions Visualization
    # ------------------------------------

    st.subheader("📊 Popular Attractions")

    popular = (
        master_df.groupby("Attraction")["Rating"]
        .mean()
        .sort_values(ascending=False)
        .head(10)
    )

    st.bar_chart(popular)