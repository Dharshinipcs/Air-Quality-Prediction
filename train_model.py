import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Load the dataset
df = pd.read_csv("air_quality.csv")

# Drop unnecessary columns
df_cleaned = df.drop(columns=["Unnamed: 15", ",,,,,", "Date", "Time"], errors="ignore")

# Convert comma-based decimal values to proper float format
for col in ["CO(GT)", "C6H6(GT)", "T", "RH", "AH"]:
    df_cleaned[col] = df_cleaned[col].astype(str).str.replace(",", ".").astype(float)

# Replace negative CO(GT) values with median of valid (positive) values
median_co = df_cleaned[df_cleaned["CO(GT)"] > 0]["CO(GT)"].median()
df_cleaned["CO(GT)"] = df_cleaned["CO(GT)"].apply(lambda x: median_co if x < 0 else x)

# Drop rows with missing values
df_cleaned = df_cleaned.dropna()

# Define target and features
target = "CO(GT)"
X = df_cleaned.drop(columns=[target])
y = df_cleaned[target]

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train multiple regression models
models = {
    "Linear Regression": LinearRegression(),
    "Decision Tree": DecisionTreeRegressor(),
    "Random Forest": RandomForestRegressor(),
    "Gradient Boosting": GradientBoostingRegressor()
}

results = []
best_model = None
best_mse = float("inf")

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    results.append((name, mse, r2))
    
    # Save the best model dynamically
    if mse < best_mse:
        best_mse = mse
        best_model = model

# Convert results to DataFrame and display
results_df = pd.DataFrame(results, columns=["Model", "MSE", "R2 Score"])
print(results_df.sort_values(by="MSE"))  # Show best model

# Save the best-performing model
if best_model:
    joblib.dump(best_model, "air_quality_model.pkl")
    print(f"✅ Best model saved as 'air_quality_model.pkl'")

print("Features used for training:", X.columns.tolist())
