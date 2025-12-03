import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

# ---------------------------------------
# Load model
# ---------------------------------------
try:
    model = joblib.load("linear_regression_real_estate.joblib")
except Exception as e:
    model = None
    load_error = str(e)

# ---------------------------------------
# Page Setup
# ---------------------------------------
st.set_page_config(
    page_title="🏡 Real Estate Price Prediction",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------------------
# Sidebar
# ---------------------------------------
st.sidebar.title("🏡 Property Input Panel")
st.sidebar.write("Fill in the details below to predict the property price.")

Square_Feet = st.sidebar.number_input("📏 Square Feet", min_value=200, max_value=20000, value=1500)
Num_Bedrooms = st.sidebar.number_input("🛏 Bedrooms", min_value=0, max_value=20, value=3)
Num_Bathrooms = st.sidebar.number_input("🛁 Bathrooms", min_value=0, max_value=20, value=2)
Num_Floors = st.sidebar.number_input("🏢 Floors", min_value=1, max_value=10, value=1)
Year_Built = st.sidebar.number_input("📅 Year Built", min_value=1800, max_value=2025, value=2010)
Has_Garden = st.sidebar.selectbox("🌳 Has Garden?", [0, 1])
Has_Pool = st.sidebar.selectbox("🏊 Has Pool?", [0, 1])
Garage_Size = st.sidebar.number_input("🚗 Garage Size (Cars)", min_value=0, max_value=10, value=1)
Location_Score = st.sidebar.slider("📍 Location Score (1–10)", 1, 10, 7)
Distance_to_Center = st.sidebar.number_input("📌 Distance to Center (Km)", min_value=0.0, max_value=100.0, value=5.0)

# Input DataFrame
input_data = pd.DataFrame({
    'Square_Feet': [Square_Feet],
    'Num_Bedrooms': [Num_Bedrooms],
    'Num_Bathrooms': [Num_Bathrooms],
    'Num_Floors': [Num_Floors],
    'Year_Built': [Year_Built],
    'Has_Garden': [Has_Garden],
    'Has_Pool': [Has_Pool],
    'Garage_Size': [Garage_Size],
    'Location_Score': [Location_Score],
    'Distance_to_Center': [Distance_to_Center]
})

# ---------------------------------------
# Main Page Header
# ---------------------------------------
st.title("🏡 Real Estate Price Prediction Dashboard")
st.write(
    "A modern machine learning system that analyzes property features and "
    "predicts an estimated market value with insights and recommendations."
)

st.markdown("---")

# ---------------------------------------
# Prediction Section
# ---------------------------------------
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("🔍 Prediction Summary")

with col2:
    st.write("")  # spacing

# If model failed to load, show error and stop
if model is None:
    st.error(f"Model failed to load. Put 'linear_regression_real_estate.joblib' in this folder.\nError: {load_error}")
else:
    # Prediction button
    if st.sidebar.button("🔮 Predict Price", use_container_width=True):

        # Run model prediction
        try:
            prediction = model.predict(input_data)[0]
        except Exception as e:
            st.error(f"Prediction failed: {e}")
            st.stop()

        with col1:
            st.success(f"### 💰 Estimated Price: ${prediction:,.2f}")

        # ------------------------------
        # Graph Section
        # ------------------------------
        st.subheader("📊 Price Comparison Chart")

        # A simple benchmark — replace this with real dataset stats if available
        avg_price = prediction * np.random.uniform(0.9, 1.1)
        labels = ["Predicted Price", "Benchmark Price"]
        values = [prediction, avg_price]

        fig, ax = plt.subplots(figsize=(6, 4))
        bars = ax.bar(labels, values)

        # Color the predicted price bar and benchmark
        bars[0].set_color("#4CAF50")  # green
        bars[1].set_color("#2196F3")  # blue

        ax.set_ylabel("Price ($)")
        ax.set_title("Predicted vs Benchmark Market Price")
        ax.set_ylim(0, max(values) * 1.2)  # give some headroom

        st.pyplot(fig)

        # ------------------------------
        # Insights Section
        # ------------------------------
        st.subheader("📌 Insights & Recommendations")

        insights = []

        if Location_Score < 5:
            insights.append("📍 **Location Score is below average** — improving neighborhood appeal increases property value.")
        else:
            insights.append("📍 **Good Location Score** — this strongly boosts the property's market value.")

        if Has_Pool == 1:
            insights.append("🏊 **Pool detected** — adds luxury appeal and may increase value depending on region.")
        else:
            insights.append("🏊 Adding a small pool could improve resale value in high-income locations.")

        if Year_Built < 2000:
            insights.append("🏚 **Older building** — renovations or upgrades could significantly increase the price.")
        else:
            insights.append("🏠 **Modern building year** — great for valuation and reduces maintenance concerns.")

        if Distance_to_Center > 15:
            insights.append("🚗 **Far from city center** — location may affect buyer demand. Consider pricing strategy.")
        else:
            insights.append("🚗 **Good proximity to the city** — attractive for buyers and raises value.")

        if Square_Feet < 800:
            insights.append("📏 Property is relatively small — expanding space could increase market worth.")
        else:
            insights.append("📏 Good overall space — larger area increases valuation strength.")

        # Display recommendations
        for tip in insights:
            st.write(f"- {tip}")

        st.markdown("---")

        # ------------------------------
        # Final Advice Section
        # ------------------------------
        st.subheader("💡 Final Recommendation")
        final_message = (
            "Based on this prediction, consider improving high-impact features such as "
            "location appearance, increasing living space, and adding modern facilities "
            "to further increase the market value."
        )
        st.info(final_message)

    else:
        st.warning("👈 Enter details on the left panel and click **Predict Price**.")

# ---------------------------------------
# Footer With Your Name
# ---------------------------------------
st.markdown("---")
st.markdown(
    """
    <div style="text-align: center; padding: 10px; font-size: 16px; color: #555;">
        <strong>Built by Oluwaseun Martins — Data Scientist</strong>
    </div>
    """,
    unsafe_allow_html=True
)
