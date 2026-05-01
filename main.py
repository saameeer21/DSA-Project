import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import os

def load_and_explore_data(filepath='sustainability_data.csv'):
    print("--- Loading Data ---")
    if not os.path.exists(filepath):
        print(f"Error: {filepath} not found. Please run generate_data.py first.")
        return None
    
    df = pd.read_csv(filepath)
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    print(f"Data shape: {df.shape}")
    print("\nFirst 5 rows:")
    print(df.head())
    print("\nSummary Statistics:")
    print(df.describe())
    
    return df

def perform_eda(df):
    print("\n--- Performing Exploratory Data Analysis ---")
    sns.set_theme(style="whitegrid")
    
    # 1. Correlation Heatmap
    plt.figure(figsize=(10, 8))
    # Exclude Timestamp for correlation
    corr = df.drop('Timestamp', axis=1).corr()
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title('Correlation Heatmap of Sustainability Features')
    plt.tight_layout()
    plt.savefig('correlation_heatmap.png')
    print("Saved correlation heatmap to 'correlation_heatmap.png'")
    
    # 2. Time Series of Energy metrics for the first week (7 days * 24 hours = 168 hours)
    plt.figure(figsize=(14, 6))
    first_week = df.head(168)
    plt.plot(first_week['Timestamp'], first_week['Energy_Consumption_kWh'], label='Energy Consumption')
    plt.plot(first_week['Timestamp'], first_week['Renewable_Output_kWh'], label='Renewable Output')
    plt.plot(first_week['Timestamp'], first_week['Carbon_Emissions_kg'], label='Carbon Emissions', linestyle='--')
    
    plt.title('Energy Metrics Over Time (First Week)')
    plt.xlabel('Date')
    plt.ylabel('Value')
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('energy_trends_first_week.png')
    print("Saved time series plot to 'energy_trends_first_week.png'")

def train_and_evaluate_models(df):
    print("\n--- Training Machine Learning Models ---")
    
    # We will predict Energy_Consumption_kWh based on weather and time features
    features = ['Hour', 'Temperature_C', 'Humidity_percent', 'Solar_Radiation_W_m2', 'Wind_Speed_m_s']
    target = 'Energy_Consumption_kWh'
    
    X = df[features]
    y = df[target]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print(f"Training set size: {X_train.shape[0]}")
    print(f"Testing set size: {X_test.shape[0]}")
    
    # Linear Regression
    lr_model = LinearRegression()
    lr_model.fit(X_train, y_train)
    lr_preds = lr_model.predict(X_test)
    
    lr_rmse = np.sqrt(mean_squared_error(y_test, lr_preds))
    lr_r2 = r2_score(y_test, lr_preds)
    
    print("\n[Linear Regression Results]")
    print(f"RMSE: {lr_rmse:.2f} kWh")
    print(f"R-squared: {lr_r2:.4f}")
    
    # Random Forest Regressor
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    rf_preds = rf_model.predict(X_test)
    
    rf_rmse = np.sqrt(mean_squared_error(y_test, rf_preds))
    rf_r2 = r2_score(y_test, rf_preds)
    
    print("\n[Random Forest Regressor Results]")
    print(f"RMSE: {rf_rmse:.2f} kWh")
    print(f"R-squared: {rf_r2:.4f}")
    
    # Feature Importance Plot for Random Forest
    plt.figure(figsize=(10, 6))
    feature_importances = pd.Series(rf_model.feature_importances_, index=features)
    feature_importances.sort_values().plot(kind='barh', color='skyblue')
    plt.title('Feature Importances (Random Forest) for Energy Consumption')
    plt.xlabel('Importance')
    plt.tight_layout()
    plt.savefig('feature_importance.png')
    print("Saved feature importance plot to 'feature_importance.png'")

if __name__ == "__main__":
    df = load_and_explore_data()
    if df is not None:
        perform_eda(df)
        train_and_evaluate_models(df)
        print("\n--- Project Execution Completed Successfully ---")
