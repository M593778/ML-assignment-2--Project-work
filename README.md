# DAT158 Used Car Price Predictor

This is my final project in DAT158.

The project is a machine learning web app that predicts the price of a used car based on information entered by the user.

## Dataset

The project uses the Kaggle dataset:

**Regression of Used Car Prices – Playground Series, Season 4, Episode 9**

The dataset is included in:

```text
data/train.csv
```

## Models

The project compares:

- Median baseline
- Ridge Regression
- Random Forest
- Gradient Boosting

The models are evaluated using RMSE, MAE and R².

The model with the lowest RMSE is used in the website.

## Run the project

Install the required packages:

```powershell
pip install -r requirements.txt
```

Train the models:

```powershell
python src\train_model.py
```

Start the website:

```powershell
streamlit run app.py
```

The website will then open locally in the browser.

## Author

Martin Olsnes Helland