# Melbourne_RealEstate_PricePrediction

# 🏡 Melbourne Housing Price Prediction – Model Comparison

This project analyzes and compares multiple machine learning algorithms to predict housing prices using the Melbourne Real Estate dataset. The objective is to evaluate which model performs best in terms of accuracy and interpretability.

---

## 📁 Dataset

- **Source:** Melbourne Housing Market dataset (CSV)
- **Target Variable:** `Price`
- **Features:** Property details like `Rooms`, `Distance`, `Landsize`, `BuildingArea`, `Suburb`, `Type`, `CouncilArea`, etc.

---

## 🧹 Data Preprocessing

- Dropped non-predictive columns: `Address`, `Date`, `Postcode`
- Handled missing values with median imputation
- Removed outliers (top 1% of `Price`, `Landsize`, `BuildingArea`)
- Label encoded categorical variables
- Feature engineering (e.g., `Rooms * Distance`, `AreaRatio`, `Distance²`)

---

## 🤖 Models Compared

1. **LightGBM Regressor**
2. **XGBoost Regressor**
3. **Random Forest Regressor**
4. **Linear Regression**

> 🔍 **Support Vector Machine was tested but excluded due to poor performance (R² < 0)**

---

## 📊 Evaluation Metrics

- **MAE** – Mean Absolute Error
- **RMSE** – Root Mean Squared Error
- **R² Score** – Proportion of variance explained

| Model              | MAE       | RMSE      | R² Score |
|-------------------|-----------|-----------|----------|
| LightGBM           | ✅ Best   | ✅ Best   | ✅ 0.86   |
| XGBoost            | Good      | Good      | 0.84     |
| Random Forest      | Decent    | Decent    | 0.82     |
| Linear Regression  | Fair      | Fair      | 0.70     |

---

## 📈 Visualizations

- **Actual vs Predicted Price** (scatter plots per model)
- **Residual Distribution** (histograms per model)
- **Model Accuracy Bar Chart** (R² comparison)

---

## 📦 Requirements

```bash
pip install pandas numpy matplotlib seaborn scikit-learn lightgbm xgboost
