import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from xgboost import XGBRegressor
import lightgbm as lgb

# 📂 Load dataset
file_path = r"C:\Users\PCAdmin\Documents\Projects\MELRE_PricePrediction\Melbourne_RealEstate.csv"
df = pd.read_csv(file_path)

# 🧹 Remove Outliers (top 1% of price, landsize, building area)
for col in ['Price', 'Landsize', 'BuildingArea']:
    upper_limit = df[col].quantile(0.99)
    df = df[df[col] <= upper_limit]

# 🧼 Basic cleaning
df = df.drop(columns=['Address', 'Date', 'Postcode'])
df['Landsize'] = df['Landsize'].fillna(df['Landsize'].median())
df['BuildingArea'] = df['BuildingArea'].fillna(df['BuildingArea'].median())
df['YearBuilt'] = df['YearBuilt'].fillna(df['YearBuilt'].median())

# 🔁 Encode categorical variables
categorical_cols = ['Suburb', 'Type', 'Method', 'SellerG', 'CouncilArea', 'Regionname']
le = LabelEncoder()
for col in categorical_cols:
    df[col] = le.fit_transform(df[col].astype(str))  # convert to string to avoid NaN errors

# 🔧 Fill any remaining NaNs
for col in df.select_dtypes(include=[np.number]).columns:
    df[col] = df[col].fillna(df[col].median())

# 🎯 Split features and target
X = df.drop(columns=['Price'])
y = df['Price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 🤖 Define models
models = {
    "Random Forest": RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42),
    "XGBoost": XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=10, random_state=42, verbosity=0),
    "LightGBM": lgb.LGBMRegressor(n_estimators=100, learning_rate=0.1, max_depth=10, random_state=42),
    "Linear Regression": LinearRegression(),
    "Support Vector Machine": SVR(kernel='rbf')
}

# 📈 Train and evaluate each model
results = []
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = mean_squared_error(y_test, y_pred) ** 0.5
    r2 = r2_score(y_test, y_pred)
    results.append({
        "Model": name,
        "MAE": mae,
        "RMSE": rmse,
        "R² Score": r2
    })

# 🧾 Create results DataFrame
results_df = pd.DataFrame(results)
print(results_df)

# Remove SVM from model list if not done already
models.pop("Support Vector Machine", None)

# Re-run training and prediction if needed
rf_model = models["Random Forest"]
lgb_model = models["LightGBM"]
xgb_model = models["XGBoost"]
lr_model = models["Linear Regression"]

# Get predictions
y_pred_rf = rf_model.predict(X_test)
y_pred_lgb = lgb_model.predict(X_test)
y_pred_xgb = xgb_model.predict(X_test)
y_pred_lr = lr_model.predict(X_test)

# Create predictions DataFrame
predictions_df = pd.DataFrame({
    "Actual Price": y_test.reset_index(drop=True),
    "LightGBM": y_pred_lgb,
    "Random Forest": y_pred_rf,
    "XGBoost": y_pred_xgb,
    "Linear Regression": y_pred_lr
})

# 📊 SCATTER PLOTS: Actual vs Predicted
plt.figure(figsize=(12, 8))
for model in ["LightGBM", "XGBoost", "Random Forest", "Linear Regression"]:
    plt.scatter(predictions_df["Actual Price"], predictions_df[model], alpha=0.5, label=model)

plt.plot([predictions_df["Actual Price"].min(), predictions_df["Actual Price"].max()],
         [predictions_df["Actual Price"].min(), predictions_df["Actual Price"].max()],
         color='black', linestyle='--', label='Perfect Prediction')

plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Actual vs Predicted House Prices")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# 📊 RESIDUAL PLOTS
plt.figure(figsize=(12, 8))
for model in ["LightGBM", "XGBoost", "Random Forest", "Linear Regression"]:
    residuals = predictions_df["Actual Price"] - predictions_df[model]
    sns.histplot(residuals, kde=True, label=model, bins=40, alpha=0.4)

plt.title("Residuals (Prediction Errors) by Model")
plt.xlabel("Prediction Error (Actual - Predicted)")
plt.legend()
plt.tight_layout()
plt.show()

# Preview
print(predictions_df.head(10))
predictions_df.to_excel("Model_Predictions_Comparison.xlsx", index=False)

# 📊 Bar chart for R² Score comparison
plt.figure(figsize=(10, 6))
sns.barplot(x='Model', y='R² Score', data=results_df, hue='Model', palette='Set2', legend=False)
plt.title("Model Accuracy Comparison (R² Score)")
plt.ylabel("R² Score")
plt.ylim(0, 1)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig("r2_score_comparison.png")  # Save the chart
# or use this if running in Jupyter:
# %matplotlib inline


# Define models to visualize (excluding SVM)
models_to_plot = {
    "LightGBM": lgb_model.predict(X_test),
    "XGBoost": xgb_model.predict(X_test),
    "Random Forest": rf_model.predict(X_test),
    "Linear Regression": lr_model.predict(X_test)
}

# Loop over each model to create individual plots
for model_name, y_pred in models_to_plot.items():
    # Prepare prediction vs actual
    actual = y_test.reset_index(drop=True)
    
    # 📈 1. Scatter Plot: Actual vs Predicted
    plt.figure(figsize=(8, 6))
    plt.scatter(actual, y_pred, alpha=0.5, color='royalblue')
    plt.plot([actual.min(), actual.max()], [actual.min(), actual.max()], color='red', linestyle='--', label="Perfect Prediction")
    plt.title(f"{model_name} – Actual vs Predicted Prices")
    plt.xlabel("Actual Price")
    plt.ylabel("Predicted Price")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    
    # 📉 2. Residual Plot: Actual - Predicted
    residuals = actual - y_pred
    plt.figure(figsize=(8, 6))
    sns.histplot(residuals, kde=True, bins=40, color='orange')
    plt.axvline(0, color='black', linestyle='--')
    plt.title(f"{model_name} – Residuals (Prediction Error)")
    plt.xlabel("Error (Actual - Predicted)")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.show()