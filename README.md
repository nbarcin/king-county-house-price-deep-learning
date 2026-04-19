# King County House Price Prediction with Deep Learning

This project focuses on predicting house prices in King County using a deep learning regression model built with TensorFlow/Keras.

It covers the complete machine learning workflow, including data preprocessing, feature engineering, neural network modeling, training, and evaluation. The goal is to learn the relationship between housing features and sale price, and to build an effective regression model for real estate prediction.
Kaggle Notebook
This project is also presented as a Kaggle notebook based on the King County house price prediction task.
https://www.kaggle.com/code/shiv28/house-price-prediction-in-king-county-usa

---

## Project Overview

House price prediction is a classic regression problem in machine learning and an important use case in real estate analytics.  
In this project, a fully connected neural network is used to estimate house prices based on property features such as:

- bedrooms
- bathrooms
- square footage
- floors
- waterfront
- view
- condition
- grade
- year built
- location-related features
- neighborhood-based information

The model is trained on the King County housing dataset and evaluated using regression metrics such as **R² score** 

---

## Dataset

The dataset contains housing records from King County and includes both numerical and location-based features.

### Main Columns
- `bedrooms`
- `bathrooms`
- `sqft_living`
- `sqft_lot`
- `floors`
- `waterfront`
- `view`
- `condition`
- `grade`
- `sqft_above`
- `sqft_basement`
- `yr_built`
- `yr_renovated`
- `zipcode`
- `lat`
- `long`
- `sqft_living15`
- `sqft_lot15`
- `date`
- `price` (target)

---

## Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib / Seaborn
- Scikit-learn
- TensorFlow / Keras

---

## Workflow

1. Data loading
2. Data cleaning
3. Handling missing values
4. Date feature transformation
5. Train-test split
6. Deep learning model building
7. Model training
8. Prediction and evaluation

---

## Model Architecture

```python
model = Sequential()
model.add(Dense(80, activation='relu'))
model.add(Dense(120, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(30, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1))   # Regression output layer

## Results

The model achieved strong regression performance:

R² Score: 0.9234

This result shows that the neural network is able to capture important patterns in the housing data and explain a large portion of the variance in house prices.
