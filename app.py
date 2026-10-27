from pathlib import Path
import json

import joblib
import pandas as pd
import streamlit as st


ROOT = Path(__file__).resolve().parent

MODEL_PATH = (
    ROOT
    / "artifacts"
    / "used_car_price_model.joblib"
)

METADATA_PATH = (
    ROOT
    / "artifacts"
    / "metadata.json"
)

MODEL_INFO_PATH = (
    ROOT
    / "artifacts"
    / "model_info.json"
)


# --------------------------------------------------
# Page settings
# --------------------------------------------------

st.set_page_config(
    page_title="Used Car Price Predictor",
    page_icon="🚗",
    layout="centered"
)


st.title(
    "🚗 Used Car Price Predictor"
)

st.write(
    """
    Enter information about a used car below.
    The machine learning model will estimate
    the vehicle's price.
    """
)


# --------------------------------------------------
# Check model exists
# --------------------------------------------------

if not MODEL_PATH.exists():

    st.error(
        """
        The trained model does not exist yet.

        Run:

        py src/train_model.py

        before starting the website.
        """
    )

    st.stop()


# --------------------------------------------------
# Load model
# --------------------------------------------------

model = joblib.load(
    MODEL_PATH
)


with open(
    METADATA_PATH,
    "r",
    encoding="utf-8"
) as file:

    metadata = json.load(
        file
    )


with open(
    MODEL_INFO_PATH,
    "r",
    encoding="utf-8"
) as file:

    model_info = json.load(
        file
    )


# --------------------------------------------------
# User inputs
# --------------------------------------------------

brand = st.selectbox(
    "Brand",
    metadata["brands"]
)


car_model = st.selectbox(
    "Model",
    metadata["models"]
)


model_year = st.number_input(
    "Model year",
    min_value=1980,
    max_value=2030,
    value=2020
)


milage = st.number_input(
    "Mileage",
    min_value=0,
    max_value=500000,
    value=50000,
    step=1000
)


fuel_type = st.selectbox(
    "Fuel type",
    metadata[
        "fuel_types"
    ]
)


transmission = st.selectbox(
    "Transmission",
    metadata[
        "transmissions"
    ]
)


horsepower = st.number_input(
    "Horsepower",
    min_value=20.0,
    max_value=2000.0,
    value=200.0,
    step=10.0
)


engine_liters = st.number_input(
    "Engine size (litres)",
    min_value=0.5,
    max_value=10.0,
    value=2.0,
    step=0.1
)


cylinders = st.number_input(
    "Number of cylinders",
    min_value=2,
    max_value=16,
    value=4
)


accident = st.selectbox(
    "Accident history",
    metadata[
        "accident_values"
    ]
)


clean_title = st.selectbox(
    "Clean title",
    metadata[
        "clean_title_values"
    ]
)


# --------------------------------------------------
# Prediction
# --------------------------------------------------

if st.button(
    "Estimate price",
    type="primary"
):

    input_data = pd.DataFrame([
        {

            "model_year":
                float(
                    model_year
                ),

            "milage":
                float(
                    milage
                ),

            "horsepower":
                float(
                    horsepower
                ),

            "engine_liters":
                float(
                    engine_liters
                ),

            "cylinders":
                float(
                    cylinders
                ),

            "brand":
                brand,

            "model":
                car_model,

            "fuel_type":
                fuel_type,

            "transmission":
                transmission,

            "accident":
                accident,

            "clean_title":
                clean_title
        }
    ])


    prediction = model.predict(
        input_data
    )[0]


    prediction = max(
        prediction,
        0
    )


    st.success(
        "Prediction complete!"
    )


    st.metric(
        "Estimated price",
        f"${prediction:,.0f}"
    )


    st.info(
        """
        This is a machine-learning estimate,
        not a guaranteed market value.

        Real vehicle prices can also depend on
        condition, location, optional equipment,
        maintenance history and current market
        conditions.
        """
    )


# --------------------------------------------------
# Model information
# --------------------------------------------------

st.divider()


with st.expander(
    "About the machine learning model"
):

    st.write(
        "Selected model:"
    )

    st.write(
        model_info[
            "best_model"
        ]
    )


    st.write(
        "Test RMSE:"
    )

    st.write(
        round(
            model_info[
                "RMSE"
            ],
            2
        )
    )


    st.write(
        "Test MAE:"
    )

    st.write(
        round(
            model_info[
                "MAE"
            ],
            2
        )
    )


    st.write(
        "Test R²:"
    )

    st.write(
        round(
            model_info[
                "R2"
            ],
            4
        )
    )