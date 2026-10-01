# 🏠 Housing Price Prediction using Random Forest

A machine learning project that predicts **median house values** using **Random Forest Regression**. The project includes data preprocessing, stratified sampling, missing-value handling, feature scaling, categorical encoding, model training, saving, and prediction on new input data.

## 🚀 Features

* Loads housing data from CSV
* Performs stratified train/test splitting based on income categories
* Handles missing numerical values using median imputation
* Scales numerical features using `StandardScaler`
* Converts categorical features using `OneHotEncoder`
* Uses Scikit-learn `Pipeline` and `ColumnTransformer`
* Trains a `RandomForestRegressor`
* Saves trained model and preprocessing pipeline using Joblib
* Loads saved model for future predictions
* Generates predictions from `input.csv`
* Saves predictions to `output.csv`

## 🛠️ Technologies Used

* Python
* Pandas
* NumPy
* Scikit-learn
* Joblib

## 📂 Project Structure

```text
Housing-Price-Prediction/
│
├── main.py
├── housing.csv
├── input.csv
├── output.csv
├── model.pkl
├── pipeline.pkl
├── requirements.txt
└── README.md
```

## ⚙️ How It Works

### 1. Load Dataset

The program reads the housing dataset:

```python
housing = pd.read_csv('housing.csv')
```

### 2. Stratified Sampling

The `median_income` column is divided into income categories and `StratifiedShuffleSplit` is used to create a representative training dataset.

### 3. Data Preprocessing

Numerical columns are processed using:

* Median imputation
* Standard scaling

The categorical `ocean_proximity` column is processed using:

* One-hot encoding

### 4. Model Training

A Random Forest regression model is trained:

```python
model = RandomForestRegressor(random_state=42)
model.fit(housing_prepared, housing_labels)
```

### 5. Save Model

The trained model and preprocessing pipeline are saved:

```python
joblib.dump(model, 'model.pkl')
joblib.dump(pipeline, 'pipeline.pkl')
```

### 6. Prediction

When the saved model already exists, the program reads new data from `input.csv` and generates predictions.

```python
predictions = model.predict(transformed_input)
```

The predictions are saved in:

```text
output.csv
```

## ▶️ How to Run

Install the required libraries:

```bash
pip install -r requirements.txt
```

Then run:

```bash
python main.py
```

### First Run

If `model.pkl` does not exist, the program:

1. Loads `housing.csv`
2. Preprocesses the data
3. Trains the Random Forest model
4. Saves `model.pkl`
5. Saves `pipeline.pkl`

### Next Runs

If `model.pkl` already exists, the program:

1. Loads the saved model
2. Loads the saved preprocessing pipeline
3. Reads `input.csv`
4. Predicts house values
5. Creates `output.csv`

## 📊 Input and Output

### Input

The input file should contain the same feature columns used during training, such as:

```text
longitude
latitude
housing_median_age
total_rooms
total_bedrooms
population
households
median_income
ocean_proximity
```

### Output

The program adds:

```text
median_house_value
```

to the input data and saves the result as:

```text
output.csv
```

## 📦 Requirements

```text
pandas
numpy
scikit-learn
joblib
```

## 🎯 Learning Objectives

This project demonstrates practical machine learning concepts including:

* Data preprocessing
* Stratified sampling
* Feature engineering
* Missing-value handling
* Numerical feature scaling
* Categorical feature encoding
* Scikit-learn pipelines
* Random Forest Regression
* Model persistence
* ML inference

## 📌 Future Improvements

* Add model evaluation metrics such as RMSE and MAE
* Compare Random Forest with Linear Regression and Gradient Boosting
* Add hyperparameter tuning
* Create a web interface using Flask or Streamlit
* Add data visualization and exploratory data analysis

## 👨‍💻 Author

**Atharv Dashpute**

---

⭐ If you found this project useful, consider giving the repository a star!
