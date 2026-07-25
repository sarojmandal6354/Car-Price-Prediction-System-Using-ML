#  Car Price Prediction System

A machine learning web app that predicts the resale price of a used car based on its brand, model, manufacturing year, fuel type, and kilometers driven. Built with **scikit-learn** for modeling and **Streamlit** for the interactive UI.

---

##  Overview

Buying or selling a used car often comes down to one hard question: *"What's it actually worth?"* This project answers that using historical listing data from CarDekho, training a linear regression model that learns pricing patterns across thousands of real car listings.

The end result is a simple web app where a user selects a car's details and instantly gets an estimated market price.

---

## Features

-  **Automated data cleaning** — standardizes column names, merges rare fuel categories, removes duplicates, and derives a `company` column from the car name
-  **ML pipeline** — one-hot encodes categorical features and trains a `LinearRegression` model inside a single `sklearn` `Pipeline`
-  **Optimized train/test split** — tests 1000 random splits and keeps the one yielding the best R² score
-  **Model evaluation visualization** — plots R² score across all 1000 trained models and highlights the best-performing one
-  **Interactive Streamlit app** — cascading dropdowns (Company → Model → Year → Fuel Type) plus a kilometers-driven input for real-time price prediction

---

##  Project Structure

```
├── CAR_DETAILS_FROM_CAR_DEKHO.csv   # Raw dataset (4,340 car listings)
├── Car_Dekho_Cleaned.csv            # Cleaned dataset used for training/inference
├── data_cleaning.py                 # Cleans raw data + trains & saves the model
├── pyplot.py                        # Trains 1000 models and visualizes R² scores
├── app.py                          # Streamlit web app for price prediction
├── LinearRegression.pkl             # Serialized trained model pipeline
└── README.md
```

---

##  How It Works

### 1. Data Cleaning (`data_cleaning.py`)
- Renames raw columns to friendlier names:
  - `km_driven` → `kms_driven`
  - `fuel` → `fuel_type`
  - `selling_price` → `Price`
- Extracts the car **company** (brand) from the `name` column
- Merges the rare `LPG` fuel type into `CNG`
- Drops duplicate rows and resets the index
- Saves the result to `Car_Dekho_Cleaned.csv`

### 2. Model Training
- Features (`X`): `name`, `company`, `year`, `kms_driven`, `fuel_type`
- Target (`y`): `Price`
- Categorical columns (`name`, `company`, `fuel_type`) are transformed using `OneHotEncoder` via `make_column_transformer`
- A `LinearRegression` model is trained inside a `make_pipeline` so preprocessing and prediction happen in one step
- The script loops through **1000 random train/test splits**, tracks the R² score for each, and retrains a final model using the `random_state` that produced the best score
- The best pipeline is serialized to `LinearRegression.pkl` using `pickle`

### 3. Model Evaluation (`pyplot.py`)
- Re-trains the 1000 models independently and plots R² score vs. model number using `matplotlib`
- Annotates the best-performing model directly on the graph

### 4. Web App (`app.py`)
- Loads the trained pipeline and cleaned dataset
- Presents cascading dropdowns so the car model list updates based on the selected company
- Validates that all fields are filled before prediction
- Feeds the selected inputs into the pipeline and displays the estimated price in ₹

---

##  Getting Started

### Prerequisites
```bash
pip install streamlit pandas numpy scikit-learn matplotlib
```

### 1. Clean the data & train the model
```bash
python data_cleaning.py
```
This generates `Car_Dekho_Cleaned.csv` and `LinearRegression.pkl`.

### 2. (Optional) Visualize model performance
```bash
python pyplot.py
```

### 3. Run the web app
```bash
streamlit run app.py
```
Then open the local URL shown in your terminal (usually `http://localhost:8501`).

>  **Note:** `app.py` currently loads the model from a hardcoded path (`LinearRegression.pkl`). Update this to a relative path, e.g.:
> ```python
> model = pickle.load(open("LinearRegression.pkl", "rb"))
> ```
> so the app runs correctly on any machine.

---

##  Dataset

The dataset (`CAR_DETAILS_FROM_CAR_DEKHO.csv`) contains **4,340 used car listings** scraped from CarDekho, with the following raw columns:

| Column | Description |
|---|---|
| `name` | Full car name (brand + model + variant) |
| `year` | Manufacturing year |
| `selling_price` | Listed selling price (₹) — renamed to `Price` |
| `km_driven` | Total kilometers driven — renamed to `kms_driven` |
| `fuel` | Fuel type (Petrol, Diesel, CNG, LPG, Electric) — renamed to `fuel_type` |

---

##  Tech Stack

- **Python** — core language
- **pandas / numpy** — data manipulation
- **scikit-learn** — preprocessing, modeling, evaluation
- **matplotlib** — model performance visualization
- **Streamlit** — web app interface

---


Author ~ Saroj Mandal