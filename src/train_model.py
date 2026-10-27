from pathlib import Path
import json
import time

import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import Ridge
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    OneHotEncoder,
    OrdinalEncoder,
    StandardScaler
)


# ==================================================
# PATHS
# ==================================================

ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = ROOT / "data" / "train.csv"

ARTIFACTS = ROOT / "artifacts"

RESULTS = ROOT / "results"


ARTIFACTS.mkdir(exist_ok=True)

RESULTS.mkdir(exist_ok=True)


RANDOM_STATE = 42


# ==================================================
# FEATURES
# ==================================================

NUMERIC_FEATURES = [
    "model_year",
    "milage",
    "horsepower",
    "engine_liters",
    "cylinders"
]


CATEGORICAL_FEATURES = [
    "brand",
    "model",
    "fuel_type",
    "transmission",
    "accident",
    "clean_title"
]


FEATURES = (
    NUMERIC_FEATURES
    +
    CATEGORICAL_FEATURES
)


TARGET = "price"


# ==================================================
# FEATURE ENGINEERING
# ==================================================

def prepare_features(df):

    df = df.copy()


    # ----------------------------------------------
    # Convert numerical columns
    # ----------------------------------------------

    df["model_year"] = pd.to_numeric(
        df["model_year"],
        errors="coerce"
    )


    df["milage"] = pd.to_numeric(
        df["milage"],
        errors="coerce"
    )


    # ----------------------------------------------
    # Extract information from engine text
    #
    # Example:
    #
    # 300.0HP 3.0L V6 Cylinder Engine
    #
    # becomes:
    #
    # horsepower = 300
    # engine_liters = 3.0
    # cylinders = 6
    # ----------------------------------------------

    engine = df["engine"].astype("string")


    df["horsepower"] = pd.to_numeric(

        engine.str.extract(
            r"(\d+(?:\.\d+)?)\s*HP",
            expand=False
        ),

        errors="coerce"
    )


    df["engine_liters"] = pd.to_numeric(

        engine.str.extract(
            r"(\d+(?:\.\d+)?)\s*L",
            expand=False
        ),

        errors="coerce"
    )


    df["cylinders"] = pd.to_numeric(

        engine.str.extract(
            r"(\d+)\s*(?:Cylinder|Cylinders|Cyl)",
            expand=False
        ),

        errors="coerce"
    )


    return df[FEATURES]


# ==================================================
# RMSE FUNCTION
# ==================================================

def calculate_rmse(
    y_true,
    y_pred
):

    mse = mean_squared_error(
        y_true,
        y_pred
    )

    return float(
        np.sqrt(mse)
    )


# ==================================================
# RIDGE REGRESSION MODEL
# ==================================================

def create_ridge_model():


    # ----------------------------------------------
    # Numerical preprocessing
    # ----------------------------------------------

    numeric_pipeline = Pipeline([

        (
            "imputer",
            SimpleImputer(
                strategy="median"
            )
        ),

        (
            "scaler",
            StandardScaler()
        )
    ])


    # ----------------------------------------------
    # Categorical preprocessing
    # ----------------------------------------------

    categorical_pipeline = Pipeline([

        (
            "imputer",
            SimpleImputer(
                strategy="most_frequent"
            )
        ),

        (
            "encoder",
            OneHotEncoder(
                handle_unknown="ignore",
                min_frequency=20
            )
        )
    ])


    preprocessing = ColumnTransformer([

        (
            "numeric",
            numeric_pipeline,
            NUMERIC_FEATURES
        ),

        (
            "categorical",
            categorical_pipeline,
            CATEGORICAL_FEATURES
        )
    ])


    # ----------------------------------------------
    # Full model pipeline
    # ----------------------------------------------

    model = Pipeline([

        (
            "preprocessing",
            preprocessing
        ),

        (
            "model",
            Ridge(
                alpha=10.0,
                solver="lsqr"
            )
        )
    ])


    return model


# ==================================================
# TREE PREPROCESSOR
# ==================================================

def create_tree_preprocessor():


    numeric_pipeline = Pipeline([

        (
            "imputer",
            SimpleImputer(
                strategy="median"
            )
        )
    ])


    categorical_pipeline = Pipeline([

        (
            "imputer",
            SimpleImputer(
                strategy="most_frequent"
            )
        ),

        (
            "encoder",
            OrdinalEncoder(
                handle_unknown="use_encoded_value",
                unknown_value=-1
            )
        )
    ])


    preprocessing = ColumnTransformer([

        (
            "numeric",
            numeric_pipeline,
            NUMERIC_FEATURES
        ),

        (
            "categorical",
            categorical_pipeline,
            CATEGORICAL_FEATURES
        )
    ])


    return preprocessing


# ==================================================
# RANDOM FOREST MODEL
# ==================================================

def create_random_forest():


    model = Pipeline([

        (
            "preprocessing",
            create_tree_preprocessor()
        ),

        (
            "model",
            RandomForestRegressor(

                n_estimators=50,

                max_depth=20,

                min_samples_leaf=2,

                n_jobs=-1,

                random_state=
                    RANDOM_STATE
            )
        )
    ])


    return model


# ==================================================
# GRADIENT BOOSTING MODEL
# ==================================================

def create_gradient_boosting():


    model = Pipeline([

        (
            "preprocessing",
            create_tree_preprocessor()
        ),

        (
            "model",
            HistGradientBoostingRegressor(

                learning_rate=0.06,

                max_iter=150,

                max_leaf_nodes=31,

                random_state=
                    RANDOM_STATE
            )
        )
    ])


    return model


# ==================================================
# MAIN PROGRAM
# ==================================================

def main():


    # ==============================================
    # LOAD DATASET
    # ==============================================

    print()
    print("==============================")
    print("LOADING DATASET")
    print("==============================")
    print()


    if not DATA_PATH.exists():

        print(
            "ERROR: train.csv was not found."
        )

        print()
        print(
            "Expected location:"
        )

        print(
            DATA_PATH
        )

        print()
        print(
            "Download train.csv from Kaggle "
            "and put it inside the data folder."
        )

        return


    df = pd.read_csv(
        DATA_PATH
    )


    print(
        "Dataset rows:",
        len(df)
    )


    print(
        "Dataset columns:",
        len(df.columns)
    )


    print()
    print("Columns:")

    print(
        df.columns.tolist()
    )


    # ==============================================
    # CHECK REQUIRED COLUMNS
    # ==============================================

    required_columns = [

        "brand",
        "model",
        "model_year",
        "milage",
        "fuel_type",
        "engine",
        "transmission",
        "accident",
        "clean_title",
        "price"
    ]


    missing_columns = [

        column

        for column
        in required_columns

        if column
        not in df.columns
    ]


    if missing_columns:

        print()
        print(
            "ERROR:"
        )

        print(
            "The following required columns "
            "are missing:"
        )

        print(
            missing_columns
        )

        return


    # ==============================================
    # CLEAN TARGET
    # ==============================================

    df[TARGET] = pd.to_numeric(

        df[TARGET],

        errors="coerce"
    )


    df = df.dropna(
        subset=[TARGET]
    )


    df = df[
        df[TARGET] > 0
    ]


    print()
    print(
        "Rows after cleaning target:",
        len(df)
    )


    # ==============================================
    # SAVE DATASET INFORMATION
    # ==============================================

    data_summary = pd.DataFrame({

        "information": [

            "Number of rows",

            "Number of columns",

            "Average price",

            "Median price",

            "Minimum price",

            "Maximum price"
        ],

        "value": [

            len(df),

            len(df.columns),

            df[TARGET].mean(),

            df[TARGET].median(),

            df[TARGET].min(),

            df[TARGET].max()
        ]
    })


    data_summary.to_csv(

        RESULTS /
        "data_summary.csv",

        index=False
    )


    # ----------------------------------------------
    # Missing values
    # ----------------------------------------------

    missing_values = pd.DataFrame({

        "column":
            df.columns,

        "missing_values":
            df.isna().sum().values,

        "missing_percent":
            (
                df.isna().sum().values
                /
                len(df)
                *
                100
            )
    })


    missing_values = (
        missing_values
        .sort_values(
            "missing_values",
            ascending=False
        )
    )


    missing_values.to_csv(

        RESULTS /
        "missing_values.csv",

        index=False
    )


    # ==============================================
    # CREATE FEATURES
    # ==============================================

    print()
    print(
        "Creating features..."
    )


    X = prepare_features(
        df
    )


    y = df[TARGET]


    # ==============================================
    # TRAIN TEST SPLIT
    # ==============================================

    X_train, X_test, y_train, y_test = (
        train_test_split(

            X,

            y,

            test_size=0.20,

            random_state=
                RANDOM_STATE
        )
    )


    print()
    print(
        "Training rows:",
        len(X_train)
    )


    print(
        "Test rows:",
        len(X_test)
    )


    # ==============================================
    # BASELINE MODEL
    # ==============================================

    print()
    print("==============================")
    print("BASELINE")
    print("==============================")


    baseline = DummyRegressor(
        strategy="median"
    )


    baseline.fit(

        np.zeros(
            (
                len(y_train),
                1
            )
        ),

        y_train
    )


    baseline_predictions = (
        baseline.predict(

            np.zeros(
                (
                    len(y_test),
                    1
                )
            )
        )
    )


    baseline_rmse = calculate_rmse(

        y_test,

        baseline_predictions
    )


    baseline_mae = (
        mean_absolute_error(

            y_test,

            baseline_predictions
        )
    )


    baseline_r2 = (
        r2_score(

            y_test,

            baseline_predictions
        )
    )


    print(
        "Baseline RMSE:",
        round(
            baseline_rmse,
            2
        )
    )


    print(
        "Baseline MAE:",
        round(
            baseline_mae,
            2
        )
    )


    print(
        "Baseline R2:",
        round(
            baseline_r2,
            4
        )
    )


    results = []


    results.append({

        "model":
            "Median baseline",

        "RMSE":
            baseline_rmse,

        "MAE":
            baseline_mae,

        "R2":
            baseline_r2,

        "training_time_seconds":
            0
    })


    # ==============================================
    # CREATE ML MODELS
    # ==============================================

    models = {

        "Ridge Regression":
            create_ridge_model(),

        "Random Forest":
            create_random_forest(),

        "Gradient Boosting":
            create_gradient_boosting()
    }


    trained_models = {}

    model_predictions = {}


    # ==============================================
    # TRAIN MODELS
    # ==============================================

    for name, model in models.items():


        print()
        print("==============================")
        print(
            "TRAINING:",
            name
        )
        print("==============================")


        start_time = (
            time.perf_counter()
        )


        model.fit(

            X_train,

            y_train
        )


        training_time = (

            time.perf_counter()
            -
            start_time
        )


        predictions = (
            model.predict(
                X_test
            )
        )


        # Prices cannot be negative

        predictions = np.maximum(

            predictions,

            0
        )


        # ------------------------------------------
        # Metrics
        # ------------------------------------------

        rmse = calculate_rmse(

            y_test,

            predictions
        )


        mae = (
            mean_absolute_error(

                y_test,

                predictions
            )
        )


        r2 = (
            r2_score(

                y_test,

                predictions
            )
        )


        results.append({

            "model":
                name,

            "RMSE":
                rmse,

            "MAE":
                mae,

            "R2":
                r2,

            "training_time_seconds":
                training_time
        })


        trained_models[
            name
        ] = model


        model_predictions[
            name
        ] = predictions


        print(
            "RMSE:",
            round(
                rmse,
                2
            )
        )


        print(
            "MAE:",
            round(
                mae,
                2
            )
        )


        print(
            "R2:",
            round(
                r2,
                4
            )
        )


        print(
            "Training time:",
            round(
                training_time,
                2
            ),
            "seconds"
        )


    # ==============================================
    # CREATE RESULTS TABLE
    # ==============================================

    results_df = pd.DataFrame(
        results
    )


    results_df = (
        results_df
        .sort_values(
            "RMSE"
        )
        .reset_index(
            drop=True
        )
    )


    results_df.to_csv(

        RESULTS /
        "model_metrics.csv",

        index=False
    )


    print()
    print("==============================")
    print("MODEL COMPARISON")
    print("==============================")
    print()


    print(
        results_df.to_string(
            index=False
        )
    )


    # ==============================================
    # SELECT BEST ML MODEL
    # ==============================================

    ml_results = results_df[

        results_df["model"]
        !=
        "Median baseline"
    ]


    best_row = (
        ml_results.iloc[0]
    )


    best_model_name = (
        best_row["model"]
    )


    best_model = (

        trained_models[
            best_model_name
        ]
    )


    best_predictions = (

        model_predictions[
            best_model_name
        ]
    )


    print()
    print("==============================")
    print("BEST MODEL")
    print("==============================")


    print(
        best_model_name
    )


    print(
        "RMSE:",
        round(
            best_row["RMSE"],
            2
        )
    )


    print(
        "MAE:",
        round(
            best_row["MAE"],
            2
        )
    )


    print(
        "R2:",
        round(
            best_row["R2"],
            4
        )
    )


    # ==============================================
    # SAVE MODEL
    # ==============================================

    model_file = (

        ARTIFACTS
        /
        "used_car_price_model.joblib"
    )


    joblib.dump(

        best_model,

        model_file
    )


    print()
    print(
        "Saved model:"
    )

    print(
        model_file
    )


    # ==============================================
    # SAVE MODEL INFORMATION
    # ==============================================

    model_info = {

        "best_model":
            best_model_name,

        "RMSE":
            float(
                best_row["RMSE"]
            ),

        "MAE":
            float(
                best_row["MAE"]
            ),

        "R2":
            float(
                best_row["R2"]
            ),

        "training_rows":
            int(
                len(X_train)
            ),

        "test_rows":
            int(
                len(X_test)
            )
    }


    with open(

        ARTIFACTS
        /
        "model_info.json",

        "w",

        encoding="utf-8"

    ) as file:


        json.dump(

            model_info,

            file,

            indent=4
        )


    # ==============================================
    # SAVE WEBSITE METADATA
    # ==============================================

    metadata = {


        "brands":

            sorted(

                df["brand"]
                .dropna()
                .astype(str)
                .unique()
                .tolist()
            ),


        "models":

            sorted(

                df["model"]
                .dropna()
                .astype(str)
                .unique()
                .tolist()
            ),


        "fuel_types":

            sorted(

                df["fuel_type"]
                .dropna()
                .astype(str)
                .unique()
                .tolist()
            ),


        "transmissions":

            sorted(

                df["transmission"]
                .dropna()
                .astype(str)
                .unique()
                .tolist()
            ),


        "accident_values":

            sorted(

                df["accident"]
                .dropna()
                .astype(str)
                .unique()
                .tolist()
            ),


        "clean_title_values":

            sorted(

                df["clean_title"]
                .dropna()
                .astype(str)
                .unique()
                .tolist()
            )
    }


    with open(

        ARTIFACTS
        /
        "metadata.json",

        "w",

        encoding="utf-8"

    ) as file:


        json.dump(

            metadata,

            file,

            indent=4
        )


    # ==============================================
    # SAVE ACTUAL VS PREDICTED DATA
    # ==============================================

    comparison = pd.DataFrame({

        "actual_price":
            y_test.values,

        "predicted_price":
            best_predictions,

        "error":
            (
                best_predictions
                -
                y_test.values
            ),

        "absolute_error":
            np.abs(
                best_predictions
                -
                y_test.values
            )
    })


    comparison = comparison.sort_values(

        "absolute_error",

        ascending=False
    )


    comparison.to_csv(

        RESULTS
        /
        "actual_vs_predicted.csv",

        index=False
    )


    # ==============================================
    # ERROR ANALYSIS
    # ==============================================

    print()
    print("==============================")
    print("LARGEST PREDICTION ERRORS")
    print("==============================")
    print()


    print(
        comparison
        .head(10)
        .to_string(
            index=False
        )
    )


    # ==============================================
    # FEATURE IMPORTANCE
    # ==============================================

    print()
    print("==============================")
    print("FEATURE IMPORTANCE")
    print("==============================")
    print()


    importance_size = min(

        2000,

        len(X_test)
    )


    X_importance = (

        X_test.iloc[
            :importance_size
        ]
    )


    y_importance = (

        y_test.iloc[
            :importance_size
        ]
    )


    importance = (
        permutation_importance(

            best_model,

            X_importance,

            y_importance,

            n_repeats=3,

            random_state=
                RANDOM_STATE,

            scoring=
                "neg_root_mean_squared_error",

            n_jobs=-1
        )
    )


    importance_df = pd.DataFrame({

        "feature":
            FEATURES,

        "importance":
            importance.importances_mean,

        "standard_deviation":
            importance.importances_std
    })


    importance_df = (
        importance_df
        .sort_values(

            "importance",

            ascending=False
        )
    )


    importance_df.to_csv(

        RESULTS
        /
        "feature_importance.csv",

        index=False
    )


    print(
        importance_df.to_string(
            index=False
        )
    )


    # ==============================================
    # FINISHED
    # ==============================================

    print()
    print("==============================")
    print("TRAINING FINISHED")
    print("==============================")
    print()


    print(
        "Files created:"
    )


    print(
        "artifacts/"
        "used_car_price_model.joblib"
    )


    print(
        "artifacts/"
        "model_info.json"
    )


    print(
        "artifacts/"
        "metadata.json"
    )


    print(
        "results/"
        "model_metrics.csv"
    )


    print(
        "results/"
        "data_summary.csv"
    )


    print(
        "results/"
        "missing_values.csv"
    )


    print(
        "results/"
        "actual_vs_predicted.csv"
    )


    print(
        "results/"
        "feature_importance.csv"
    )



if __name__ == "__main__":

    main()