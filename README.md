# Tourism Experience Analytics

## Project Overview

This project presents an end-to-end **Tourism Analytics System** that leverages Machine Learning techniques to:

- Predict user ratings for attractions (Regression)
- Classify user visit modes (Classification)
- Recommend personalized tourist attractions (Recommendation System)

The system integrates supervised learning and recommender systems into a unified framework and is deployed using **Streamlit** as an interactive web application.

---

## Problem Statement

Tourism platforms aim to enhance user experience by providing personalized recommendations and understanding user behavior. This project analyzes user preferences, travel patterns, and attraction attributes to achieve:

- Rating prediction
- Visit mode classification
- Attraction recommendation

---

## Dataset Description

The dataset consists of multiple relational tables:

- **Transaction Dataset** – User visits, ratings, visit mode
- **User Dataset** – Demographic and location information
- **Attraction Dataset** – Attraction details and type
- **Geographic Hierarchy Tables** – City, Country, Region, Continent mappings

After preprocessing and feature engineering:

- Total Records: ~52,890
- Total Features: 56
- Rating Scale: 1–5
- Visit Modes: Business, Couples, Family, Friends, Solo

---

## Exploratory Data Analysis (EDA)

Key insights:

- Significant class imbalance in VisitMode
- Ratings skewed towards higher values (4–5)
- Beaches and religious sites among most popular attraction types
- Geographic distribution influences user behavior

---

# Model Development

## 1️. Regression – Rating Prediction

**Models Used:**
- Linear Regression
- Random Forest Regressor

| Model | R² Score | RMSE |
|--------|----------|------|
| Linear Regression | 0.74 | 0.48 |
| Random Forest | 0.69 | 0.53 |

Linear Regression performed best, indicating a strong linear relationship between features and ratings.

---

## 2️. Classification – Visit Mode Prediction

**Models Used:**
- Random Forest Classifier
- XGBoost Classifier

**Accuracy:** ~50%

Due to class imbalance, performance varied across categories, with better recall for dominant classes.

---

## 3️. Recommendation System

###  Collaborative Filtering (User-Based)

- Built using User-Item matrix
- Cosine similarity
- Top-K neighborhood selection

**RMSE: 1.62**

###  Content-Based Filtering

- Built using attraction attributes
- Cosine similarity between attractions

**RMSE: 3.35**

Collaborative filtering significantly outperformed content-based filtering.

---

#  Deployment

The system is deployed using **Streamlit**, allowing users to:

- Input demographic and travel preferences
- Predict likely visit mode
- Receive personalized attraction recommendations
- View tourism analytics visualizations

---

#  Technologies Used

- Python
- Pandas & NumPy
- Scikit-learn
- XGBoost
- Matplotlib & Seaborn
- Streamlit
- Cosine Similarity (Recommender System)

