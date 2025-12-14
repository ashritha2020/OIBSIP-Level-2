"""
model_training.py
Short script to train a linear regression model and save plots and model.
Run: python model_training.py  (requires pandas, scikit-learn, matplotlib)
"""
import os, pickle
import pandas as pd, numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

BASE = os.path.dirname(__file__)
DATA = os.path.join(BASE, "..", "Data")
PLOTS = os.path.join(BASE, "..", "Plots")
MODEL = os.path.join(BASE, "..", "Model")
os.makedirs(PLOTS, exist_ok=True)
os.makedirs(MODEL, exist_ok=True)

# Load data
for f in ("housing-cleaned.csv", "housing.csv"):
    p = os.path.join(DATA, f)
    if os.path.exists(p):
        df = pd.read_csv(p)
        print("Loaded", f)
        break
else:
    raise FileNotFoundError("No dataset found in Dataset/")

# Select target (prefer 'actual')
for t in ("actual","price","Price","SalePrice","median_house_value"):
    if t in df.columns:
        target = t
        break
else:
    numeric = df.select_dtypes(include=[float,int]).columns.tolist()
    target = numeric[-1]

# Drop leak columns
for leak in ("predicted","prediction","y_pred"):
    if leak in df.columns:
        df = df.drop(columns=[leak])

X = df.select_dtypes(include=[float,int]).drop(columns=[target], errors='ignore')
y = df[target]

imp = SimpleImputer(strategy="median")
X_imp = imp.fit_transform(X)
scaler = StandardScaler()
Xs = scaler.fit_transform(X_imp)

X_train, X_test, y_train, y_test = train_test_split(Xs, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"Test MSE: {mse:.3f}, R2: {r2:.3f}")

# Save model
with open(os.path.join(MODEL, "linear_regression_model.pkl"), "wb") as f:
    pickle.dump({"model": model, "imputer": imp, "scaler": scaler, "features": X.columns.tolist(), "target": target}, f)

# Plots (same as notebook will generate)
plt.figure(figsize=(6,6))
plt.scatter(y_test, y_pred, alpha=0.7)
lims = [min(min(y_test), min(y_pred)), max(max(y_test), max(y_pred))]
plt.plot(lims, lims, '--')
plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.title("Predicted vs Actual")
plt.savefig(os.path.join(PLOTS, "scatter_actual_vs_pred.png"))
plt.close()

resid = y_test - y_pred
plt.figure(figsize=(6,4))
plt.scatter(y_pred, resid, alpha=0.7)
plt.axhline(0, linestyle="--")
plt.xlabel("Predicted")
plt.ylabel("Residuals")
plt.title("Residuals vs Predicted")
plt.savefig(os.path.join(PLOTS, "residuals.png"))
plt.close()

plt.figure(figsize=(6,4))
plt.hist(y, bins=20)
plt.xlabel(target)
plt.title("Distribution of " + target)
plt.savefig(os.path.join(PLOTS, "histogram.png"))
plt.close()

# Top coeffs
import numpy as np
coefs = model.coef_
feat_names = X.columns.tolist()
abs_idx = np.argsort(np.abs(coefs))[::-1][:6]
names = [feat_names[i] for i in abs_idx]
vals = [coefs[i] for i in abs_idx]
plt.figure(figsize=(7,4))
plt.bar(names, vals)
plt.title("Top coefficients (abs)")
plt.ylabel("Coefficient")
plt.savefig(os.path.join(PLOTS, "coefficients.png"))
plt.close()

# Pie chart example if bedrooms col exists
if "bedrooms" in df.columns:
    counts = df["bedrooms"].value_counts().sort_index()
    plt.figure(figsize=(5,5))
    plt.pie(counts, labels=counts.index, autopct="%1.0f%%")
    plt.title("Bedrooms distribution")
    plt.savefig(os.path.join(PLOTS, "pie_bedrooms.png"))
    plt.close()
