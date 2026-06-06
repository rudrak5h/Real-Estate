import streamlit as st
import pandas as pd
import numpy as np
import joblib

import matplotlib.pyplot as plt
import seaborn as sns


# ==================================================
# PAGE CONFIG
# ==================================================

st.set_page_config(
    page_title="Real Estate Investment Advisor",
    layout="wide"
)


# ==================================================
# LOAD FILES
# ==================================================

df = pd.read_csv("cleaned.csv")

classification_model = joblib.load(
    "best_classification_model.pkl"
)

regression_model = joblib.load(
    "best_regression_model.pkl"
)

label_encoders = joblib.load(
    "label_encoders.pkl"
)

scaler = joblib.load(
    "scaler.pkl"
)

feature_columns = joblib.load(
    "feature_columns.pkl"
)


# ==================================================
# MEDIAN VALUES
# ==================================================

median_price = df["Price_in_Lakhs"].median()

median_price_sqft = df["Price_per_SqFt"].median()


# ==================================================
# TITLE
# ==================================================

st.title("🏠 Real Estate Investment Advisor")

st.markdown(
    """
    Predict whether a property is a good investment
    and estimate future prices.
    """
)


# ==================================================
# SIDEBAR
# ==================================================

st.sidebar.title("Navigation")

page = st.sidebar.radio(
    "Go To",
    [
        "Property Prediction",
        "Visual Insights"
    ]
)


# ==================================================
# PROPERTY PREDICTION PAGE
# ==================================================

if page == "Property Prediction":

    st.header("Enter Property Details")


    col1, col2 = st.columns(2)


    # ==================================================
    # LEFT COLUMN
    # ==================================================

    with col1:

        city = st.selectbox(
            "Select City",
            sorted(df["City"].unique())
        )

        property_type = st.selectbox(
            "Property Type",
            sorted(df["Property_Type"].unique())
        )

        bhk = st.slider(
            "BHK",
            1,
            5,
            3
        )

        size_sqft = st.slider(
            "Size in SqFt",
            500,
            5000,
            2000
        )

        nearby_schools = st.slider(
            "Nearby Schools",
            1,
            10,
            5
        )


    # ==================================================
    # RIGHT COLUMN
    # ==================================================

    with col2:

        price_lakhs = st.slider(
            "Current Price (Lakhs)",
            10,
            500,
            150
        )

        furnished_status = st.selectbox(
            "Furnished Status",
            sorted(df["Furnished_Status"].unique())
        )

        parking_space = st.selectbox(
            "Parking Space",
            sorted(df["Parking_Space"].unique())
        )

        nearby_hospitals = st.slider(
            "Nearby Hospitals",
            1,
            10,
            5
        )

        transport = st.selectbox(
            "Transport Accessibility",
            sorted(
                df["Public_Transport_Accessibility"]
                .unique()
            )
        )


    # ==================================================
    # PREDICT BUTTON
    # ==================================================

    if st.button("Predict Investment"):

        price_per_sqft = (
            (price_lakhs * 100000)
            / size_sqft
        )


        # ==================================================
        # CREATE INVESTMENT SCORE
        # ==================================================

        investment_score = 0


        if price_lakhs <= median_price:
            investment_score += 1

        if price_per_sqft <= median_price_sqft:
            investment_score += 1

        if bhk >= 3:
            investment_score += 1

        if parking_space == "Yes":
            investment_score += 1

        if furnished_status in [
            "Furnished",
            "Semi-furnished"
        ]:
            investment_score += 1

        if nearby_schools >= 4:
            investment_score += 1

        if nearby_hospitals >= 4:
            investment_score += 1

        if transport == "High":
            investment_score += 1


        # ==================================================
        # INPUT DATAFRAME
        # ==================================================

        input_data = pd.DataFrame({

            "State": ["Delhi"],

            "City": [city],

            "Locality": ["Locality_1"],

            "Property_Type": [property_type],

            "BHK": [bhk],

            "Size_in_SqFt": [size_sqft],

            "Price_in_Lakhs": [price_lakhs],

            "Year_Built": [2015],

            "Furnished_Status": [furnished_status],

            "Floor_No": [5],

            "Total_Floors": [10],

            "Age_of_Property": [10],

            "Nearby_Schools": [nearby_schools],

            "Nearby_Hospitals": [nearby_hospitals],

            "Public_Transport_Accessibility": [transport],

            "Parking_Space": [parking_space],

            "Security": ["Yes"],

            "Amenities": ["Gym"],

            "Facing": ["East"],

            "Owner_Type": ["Owner"],

            "Availability_Status": [
                "Ready_to_Move"
            ],

            "Price_per_SqFt": [
                price_per_sqft
            ],

            "Investment_Score": [
                investment_score
            ],

            "Transport_Score": [
                3 if transport == "High"
                else 2 if transport == "Medium"
                else 1
            ],

            "Amenities_Count": [
                nearby_schools + nearby_hospitals
            ],

            "Future_Price_5Y": [0],

            "Good_Investment": [0]
        })


        # ==================================================
        # ENCODE CATEGORICAL COLUMNS
        # ==================================================

        categorical_cols = input_data.select_dtypes(
            include="object"
        ).columns


        for col in categorical_cols:

            le = label_encoders[col]

            input_data[col] = le.transform(
                input_data[col]
            )


        # ==================================================
        # CLASSIFICATION INPUT
        # ==================================================

        X_class = input_data.drop(
            [
                "Good_Investment",
                "Future_Price_5Y"
            ],
            axis=1
        )

        X_class = X_class[feature_columns]

        X_class_scaled = scaler.transform(
            X_class
        )


        # ==================================================
        # REGRESSION INPUT
        # ==================================================

        X_reg = input_data.drop(
            [
                "Future_Price_5Y",
                "Good_Investment"
            ],
            axis=1
        )

        X_reg = X_reg[feature_columns]

        X_reg_scaled = scaler.transform(
            X_reg
        )


        # ==================================================
        # PREDICTIONS
        # ==================================================

        class_prediction = (
            classification_model.predict(
                X_class_scaled
            )[0]
        )

        class_probability = (
            classification_model.predict_proba(
                X_class_scaled
            )[0]
        )

        regression_prediction = (
            regression_model.predict(
                X_reg_scaled
            )[0]
        )


        regression_prediction = max(
            regression_prediction,
            price_lakhs
        )


        # ==================================================
        # RESULTS
        # ==================================================

        st.subheader("Prediction Results")

        col1, col2, col3 = st.columns(3)


        # ==================================================
        # INVESTMENT RESULT
        # ==================================================

        with col1:

            if class_prediction == 1:

                st.success(
                    "✅ Good Investment"
                )

            else:

                st.error(
                    "❌ Not a Good Investment"
                )


        # ==================================================
        # CONFIDENCE SCORE
        # ==================================================

        with col2:

            confidence = (
                np.max(class_probability)
                * 100
            )

            st.metric(
                "Confidence Score",
                f"{confidence:.2f}%"
            )


        # ==================================================
        # FUTURE PRICE
        # ==================================================

        with col3:

            st.metric(
                "Estimated Price After 5 Years",
                f"{regression_prediction:.2f} Lakhs"
            )


# ==================================================
# VISUAL INSIGHTS PAGE
# ==================================================

if page == "Visual Insights":

    st.header("Visual Insights")


    # ==================================================
    # PRICE DISTRIBUTION
    # ==================================================

    st.subheader("Price Distribution")

    fig, ax = plt.subplots(figsize=(8,4))

    sns.histplot(
        df["Price_in_Lakhs"],
        kde=True,
        ax=ax
    )

    st.pyplot(fig)


    # ==================================================
    # CITY-WISE PRICE
    # ==================================================

    st.subheader(
        "Top Cities by Average Price"
    )

    city_price = (

        df.groupby("City")
        ["Price_in_Lakhs"]

        .mean()

        .sort_values(
            ascending=False
        )

        .head(10)
    )

    fig, ax = plt.subplots(figsize=(10,5))

    city_price.plot(
        kind="bar",
        ax=ax
    )

    ax.set_ylabel("Average Price")

    st.pyplot(fig)


    # ==================================================
    # HEATMAP
    # ==================================================

    st.subheader(
        "Feature Correlation Heatmap"
    )

    numeric_df = df.select_dtypes(
        include=np.number
    )

    fig, ax = plt.subplots(figsize=(10,6))

    sns.heatmap(
        numeric_df.corr(),
        cmap="coolwarm",
        ax=ax
    )

    st.pyplot(fig)


    # ==================================================
    # FEATURE IMPORTANCE
    # ==================================================

    st.subheader(
        "Feature Importance"
    )


    if hasattr(
        classification_model,
        "feature_importances_"
    ):

        importance = (
            classification_model
            .feature_importances_
        )

    else:

        importance = np.abs(
            classification_model.coef_[0]
        )


    importance_df = pd.DataFrame({

        "Feature": feature_columns,

        "Importance": importance
    })


    # Remove unwanted features

    importance_df = importance_df[
        ~importance_df["Feature"].isin(
            [
                "Investment_Score",
                "ID"
            ]
        )
    ]


    importance_df = (
        importance_df
        .sort_values(
            by="Importance",
            ascending=False
        )

        .head(10)
    )


    fig, ax = plt.subplots(figsize=(8,5))

    sns.barplot(
        x="Importance",
        y="Feature",
        data=importance_df,
        ax=ax
    )

    ax.set_title(
        "Top Feature Importance"
    )

    st.pyplot(fig)
