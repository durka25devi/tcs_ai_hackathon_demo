"""
Dedicated script to train, evaluate, and save the Canteen Demand Estimation Model.
- Chronological train/test split (no date leakage)
- Feature engineering: product encodings, price, day-of-week, promotion, stock
- Compares multiple models: Baseline Historical Average, Ridge Regression, and Random Forest Regressor
- Reports evaluation metrics (MAE, RMSE, R2, Feature Importances)
- Saves the trained model and metadata for production serving
"""

import os
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "backend", "models")
os.makedirs(MODEL_DIR, exist_ok=True)

def train_demand_model():
    daily_path = os.path.join(DATA_DIR, "daily_sales.csv")
    if not os.path.exists(daily_path):
        raise FileNotFoundError(f"Missing {daily_path}. Run scripts/generate_data.py first.")

    df = pd.read_csv(daily_path)
    df["date"] = pd.to_datetime(df["date"])
    # 1. Strictly chronological sort to avoid any date leakage
    df = df.sort_values(by="date").reset_index(drop=True)

    print(f"Total daily sales observations: {len(df)}")
    print(f"Date range: {df['date'].min().strftime('%Y-%m-%d')} to {df['date'].max().strftime('%Y-%m-%d')}")

    # 2. Encodings
    unique_pids = sorted(df["product_id"].unique())
    pid_map = {pid: i for i, pid in enumerate(unique_pids)}

    dows = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    dow_map = {d: i for i, d in enumerate(dows)}

    df["pid_enc"] = df["product_id"].map(pid_map)
    df["dow_enc"] = df["day_of_week"].map(dow_map)

    features = ["pid_enc", "selling_price", "dow_enc", "promotion_active", "stock_available"]
    target = "quantity_sold"

    X = df[features]
    y = df[target]

    # 3. Chronological 80/20 train/test split
    split_idx = int(len(df) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    split_date = df.iloc[split_idx]["date"].strftime("%Y-%m-%d")

    print(f"\n--- Chronological Split ---")
    print(f"Train samples: {len(X_train)} (Up to {split_date})")
    print(f"Test samples:  {len(X_test)} (From {split_date} onwards)")

    # 4. Baseline Model: Historical average per (product, day_of_week)
    train_df = df.iloc[:split_idx]
    baseline_lookup = train_df.groupby(["pid_enc", "dow_enc"])["quantity_sold"].mean().to_dict()
    overall_mean = train_df["quantity_sold"].mean()

    baseline_preds = [
        baseline_lookup.get((r["pid_enc"], r["dow_enc"]), overall_mean)
        for _, r in X_test.iterrows()
    ]
    base_mae = mean_absolute_error(y_test, baseline_preds)
    base_rmse = np.sqrt(mean_squared_error(y_test, baseline_preds))
    base_r2 = r2_score(y_test, baseline_preds)

    print(f"\n--- 1. Baseline Model (Historical Mean) ---")
    print(f"MAE:  {base_mae:.3f} units")
    print(f"RMSE: {base_rmse:.3f} units")
    print(f"R²:   {base_r2:.3f}")

    # 5. Model 2: Linear Ridge Regression
    ridge = Ridge(alpha=1.0)
    ridge.fit(X_train, y_train)
    ridge_preds = ridge.predict(X_test)
    ridge_mae = mean_absolute_error(y_test, ridge_preds)
    ridge_rmse = np.sqrt(mean_squared_error(y_test, ridge_preds))
    ridge_r2 = r2_score(y_test, ridge_preds)

    print(f"\n--- 2. Linear Ridge Regression ---")
    print(f"MAE:  {ridge_mae:.3f} units")
    print(f"RMSE: {ridge_rmse:.3f} units")
    print(f"R²:   {ridge_r2:.3f}")

    # 6. Model 3: Random Forest Regressor
    rf = RandomForestRegressor(
        n_estimators=150,
        max_depth=6,
        min_samples_split=4,
        min_samples_leaf=2,
        random_state=42
    )
    rf.fit(X_train, y_train)
    rf_preds = rf.predict(X_test)
    rf_mae = mean_absolute_error(y_test, rf_preds)
    rf_rmse = np.sqrt(mean_squared_error(y_test, rf_preds))
    rf_r2 = r2_score(y_test, rf_preds)

    print(f"\n--- 3. Random Forest Regressor (Selected Model) ---")
    print(f"MAE:  {rf_mae:.3f} units")
    print(f"RMSE: {rf_rmse:.3f} units")
    print(f"R²:   {rf_r2:.3f}")

    print("\nFeature Importances:")
    for feat, imp in zip(features, rf.feature_importances_):
        print(f"  • {feat:18s}: {imp:.4f} ({imp*100:.1f}%)")

    # 7. Save model artifact and metadata
    model_path = os.path.join(MODEL_DIR, "demand_rf_model.joblib")
    meta_path = os.path.join(MODEL_DIR, "model_metadata.joblib")

    metadata = {
        "features": features,
        "product_encodings": pid_map,
        "dow_encodings": dow_map,
        "split_date": split_date,
        "metrics": {
            "baseline_mae": round(float(base_mae), 3),
            "baseline_rmse": round(float(base_rmse), 3),
            "baseline_r2": round(float(base_r2), 3),
            "model_mae": round(float(rf_mae), 3),
            "model_rmse": round(float(rf_rmse), 3),
            "model_r2": round(float(rf_r2), 3),
            "feature_importances": dict(zip(features, [round(float(x), 4) for x in rf.feature_importances_]))
        }
    }

    joblib.dump(rf, model_path)
    joblib.dump(metadata, meta_path)
    print(f"\nSaved trained model to {model_path}")
    print(f"Saved metadata to {meta_path}")

    # 8. Test Price Sensitivity Check
    print("\n--- Price Sensitivity Sanity Check ---")
    print("Testing Veg Sandwich (P101) across prices on Wednesday:")
    sample_pid_enc = pid_map["P101"]
    sample_dow_enc = dow_map["Wednesday"]
    sample_stock = 70
    for p in [30.0, 35.0, 40.0, 45.0, 50.0]:
        sample_row = pd.DataFrame([[sample_pid_enc, p, sample_dow_enc, 0, sample_stock]], columns=features)
        pred_qty = rf.predict(sample_row)[0]
        est_rev = p * pred_qty
        est_contrib = (p - 18.0) * pred_qty # var cost = 18
        print(f"  Price: ₹{p:.0f} -> Pred Qty: {pred_qty:.1f} units | Rev: ₹{est_rev:.0f} | Contrib: ₹{est_contrib:.0f}")

    return metadata

if __name__ == "__main__":
    train_demand_model()
