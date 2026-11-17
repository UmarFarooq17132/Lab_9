import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import GridSearchCV, KFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

# 1. Import dataset and remove duplicates
concrete = pd.read_csv("concrete_strength.csv").drop_duplicates()
target = "Concrete compressive strength"
X = concrete.drop(columns=target)
y = concrete[target]

# (a) Hold out 20% test set and compute linear baseline 5-fold CV RMSE
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)
print(f"Training rows: {len(X_train)}, Test rows: {len(X_test)}")

folds = KFold(n_splits=5, shuffle=True, random_state=42)
scoring = "neg_root_mean_squared_error"

linear_baseline = LinearRegression()
baseline_rmse = -cross_val_score(
    linear_baseline, X_train, y_train, cv=folds, scoring=scoring
)
print(f"Linear Baseline Mean CV RMSE: {baseline_rmse.mean():.2f} MPa")

# (b) Tune Polynomial + StandardScaler + Ridge pipeline
pipe = Pipeline([
    ("poly", PolynomialFeatures(include_bias=False)),
    ("scale", StandardScaler()),
    ("ridge", Ridge()),
])

param_grid = {
    "poly__degree": [1, 2, 3],
    "ridge__alpha": [0.01, 0.1, 1, 10, 100],
}

search = GridSearchCV(
    pipe, param_grid, cv=folds, scoring=scoring, refit=True
)
search.fit(X_train, y_train)

results = pd.DataFrame(search.cv_results_)
top3 = results.sort_values("rank_test_score").head(3)
print("\nTop 3 GridSearchCV Settings:")
for _, row in top3.iterrows():
  print(
      f"Rank {int(row['rank_test_score'])}: {row['params']} -> Mean CV RMSE:"
      f" {-row['mean_test_score']:.2f} MPa"
  )

# (c) Evaluate refitted winner on held-out test set
y_pred = search.predict(X_test)
test_rmse = np.sqrt(mean_squared_error(y_test, y_pred))
test_mae = mean_absolute_error(y_test, y_pred)

print(f"\nBest Parameters: {search.best_params_}")
print(f"Test RMSE: {test_rmse:.2f} MPa")
print(f"Test MAE: {test_mae:.2f} MPa")

# Plot Actual vs Predicted
plt.figure(figsize=(7, 6))
plt.scatter(y_test, y_pred, alpha=0.65, color="teal", label="Test Predictions")
min_val = min(y_test.min(), y_pred.min())
max_val = max(y_test.max(), y_pred.max())
plt.plot(
    [min_val, max_val],
    [min_val, max_val],
    color="crimson",
    linestyle="--",
    linewidth=2,
    label="Ideal Fit (y = x)",
)
plt.xlabel("Actual Compressive Strength (MPa)")
plt.ylabel("Predicted Compressive Strength (MPa)")
plt.title("Concrete Strength: Actual vs. Predicted (Tuned Ridge)")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.5)
plt.tight_layout()
plt.savefig("task3_concrete.png", dpi=300)
plt.close()