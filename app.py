import streamlit as st
import pandas as pd
import joblib
from pathlib import Path


# =====================================================
# PAGE CONFIGURATION
# =====================================================

st.set_page_config(
    page_title="Car Price Prediction",
    page_icon="🚘",
    layout="wide"
)


# =====================================================
# MODEL PATH
# =====================================================

MODEL_PATH = Path("model") / "car_price_model.pkl"


# =====================================================
# LOAD MODEL
# =====================================================

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


# =====================================================
# CUSTOM CSS
# =====================================================

st.markdown("""
<style>

.stApp {
    background-color: #f5f7fb;
}

/* Main title */
.main-title {
    text-align: center;
    color: #17365d;
    font-size: 42px;
    font-weight: 700;
    margin-bottom: 5px;
}

/* Subtitle */
.subtitle {
    text-align: center;
    color: #64748b;
    font-size: 18px;
    margin-bottom: 30px;
}

/* Section titles */
.section-title {
    color: #17365d;
    font-size: 25px;
    font-weight: 600;
}

/* Information card */
.info-card {
    background-color: white;
    padding: 25px;
    border-radius: 15px;
    border: 1px solid #e2e8f0;
    box-shadow: 0px 4px 15px rgba(0,0,0,0.05);
}

/* Prediction card */
.prediction-card {
    background-color: #dcfce7;
    padding: 30px;
    border-radius: 15px;
    border: 1px solid #86efac;
    text-align: center;
    margin-top: 20px;
}

.prediction-label {
    color: #166534;
    font-size: 20px;
    font-weight: 500;
}

.prediction-price {
    color: #14532d;
    font-size: 40px;
    font-weight: 700;
    margin-top: 10px;
}

/* Button */
div.stFormSubmitButton > button {
    background-color: #2563eb;
    color: white;
    border: none;
    border-radius: 10px;
    height: 50px;
    font-size: 17px;
    font-weight: 600;
}

div.stFormSubmitButton > button:hover {
    background-color: #1d4ed8;
    color: white;
}

/* Footer */
.footer {
    text-align: center;
    color: #64748b;
    font-size: 14px;
    margin-top: 40px;
}

</style>
""", unsafe_allow_html=True)


# =====================================================
# HEADER
# =====================================================

st.markdown(
    '<div class="main-title">🚘 Car Price Prediction</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">Machine Learning Based Used Car Price Estimator</div>',
    unsafe_allow_html=True
)

st.divider()


# =====================================================
# CHECK MODEL
# =====================================================

if not MODEL_PATH.exists():

    st.error(
        "Trained model not found. "
        "Please run `python train.py` first."
    )

    st.stop()


try:
    model = load_model()

except Exception as error:

    st.error(f"Unable to load the model: {error}")

    st.stop()


# =====================================================
# MAIN LAYOUT
# =====================================================

left, right = st.columns(
    [1.2, 1],
    gap="large"
)


# =====================================================
# INPUT SECTION
# =====================================================

with left:

    st.markdown(
        '<div class="section-title">Enter Car Details</div>',
        unsafe_allow_html=True
    )

    st.write("")

    with st.form("car_prediction_form"):

        year = st.number_input(
            "Manufacturing Year",
            min_value=2000,
            max_value=2035,
            value=2020,
            step=1
        )

        present_price = st.number_input(
            "Present Price (₹ lakhs)",
            min_value=0.1,
            max_value=1000.0,
            value=9.85,
            step=0.1
        )

        fuel_type = st.selectbox(
            "Fuel Type",
            ["Petrol", "Diesel", "CNG"]
        )

        kms_driven = st.number_input(
            "Kilometres Driven",
            min_value=0,
            max_value=2000000,
            value=8000,
            step=1000
        )

        st.write("")

        submitted = st.form_submit_button(
            "🚘 Predict Selling Price",
            use_container_width=True
        )


# =====================================================
# MODEL INFORMATION
# =====================================================

with right:

    st.markdown(
        '<div class="section-title">About This Model</div>',
        unsafe_allow_html=True
    )

    st.write("")

    st.markdown(
        """
<div class="info-card">

<p style="color:#334155;">
This application predicts the estimated selling price
of a used car using Machine Learning.
</p>

<hr>

<p style="color:#334155;">
<b>Algorithm:</b> Linear Regression
</p>

<p style="color:#334155;">
<b>Features:</b> Year, Present Price, Fuel Type, Kilometres Driven
</p>

<p style="color:#334155;">
<b>Target:</b> Selling Price
</p>

<p style="color:#334155;">
<b>Evaluation:</b> Train/Test Split
</p>

</div>
""",
        unsafe_allow_html=True
    )


# =====================================================
# PREDICTION
# =====================================================

if submitted:

    input_data = pd.DataFrame([{
        "Year": year,
        "Present_Price": present_price,
        "Fuel_Type": fuel_type,
        "Kms_Driven": kms_driven
    }])

    try:

        prediction = float(
            model.predict(input_data)[0]
        )

        st.divider()

        if prediction >= 0:

            st.markdown(
                f"""<div class="prediction-card">
<div class="prediction-label">Estimated Selling Price</div>
<div class="prediction-price">₹{prediction:.2f} Lakhs</div>
</div>""",
                unsafe_allow_html=True
            )

            st.caption(
                "⚠️ This is a machine-learning estimate and may "
                "differ from the actual market price."
            )

        else:

            st.warning(
                "The model returned an unrealistic negative price. "
                "Please check the input values."
            )

    except Exception as error:

        st.error(
            f"Prediction failed: {error}"
        )


# =====================================================
# FOOTER
# =====================================================

st.markdown(
    """
<div class="footer">
Car Price Prediction • Python • Pandas • Scikit-learn • Streamlit
</div>
""",
    unsafe_allow_html=True
)