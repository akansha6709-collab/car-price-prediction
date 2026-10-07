# 🚘 Car Price Prediction

A Machine Learning based web application that predicts the estimated selling price of a used car.

## 📌 Project Overview

This project uses Machine Learning to estimate the selling price of a used car based on:

- Manufacturing Year
- Present Price
- Fuel Type
- Kilometres Driven

The trained model is integrated with a Streamlit web application where users can enter car details and receive an estimated selling price.

## 🛠️ Technologies Used

- Python
- Pandas
- Scikit-learn
- Streamlit
- Joblib

## 🤖 Machine Learning Model

**Algorithm:** Linear Regression

The model was trained using a train-test split and evaluated using:

- Mean Absolute Error (MAE)
- R² Score

### Model Performance

- **MAE:** 1.52 lakhs
- **R² Score:** 0.78

## 📊 Features

| Feature | Description |
|---|---|
| Year | Manufacturing year of the car |
| Present Price | Current price of the car in lakhs |
| Fuel Type | Petrol, Diesel or CNG |
| Kms Driven | Total kilometres driven |

## 🌐 Streamlit Application

The project includes an interactive Streamlit interface that allows users to enter car details and get an estimated selling price.

## ▶️ How to Run

### 1. Clone the repository

```bash
git clone https://github.com/akansha6709-collab/car-price-prediction.git