import os
import joblib
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from backend.data_loader import data_loader

MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "demand_rf_model.joblib")
META_PATH = os.path.join(os.path.dirname(__file__), "models", "model_metadata.joblib")

class DemandEstimator:
    def __init__(self):
        self.model = None
        self.baseline_mae = None
        self.model_mae = None
        self.model_rmse = None
        self.feature_columns = None
        self.product_encodings = {}
        self.dow_encodings = {}
        self.train_summary = {}

        if os.path.exists(MODEL_PATH) and os.path.exists(META_PATH):
            self._load_saved()
        else:
            self._train()

    def _load_saved(self):
        self.model = joblib.load(MODEL_PATH)
        meta = joblib.load(META_PATH)
        self.feature_columns = meta["features"]
        self.product_encodings = meta["product_encodings"]
        self.dow_encodings = meta["dow_encodings"]
        m = meta["metrics"]
        self.baseline_mae = m["baseline_mae"]
        self.model_mae = m["model_mae"]
        self.model_rmse = m["model_rmse"]
        self.train_summary = {
            "model_type": "RandomForestRegressor (Persisted)",
            "train_samples": 576,
            "test_samples": 144,
            "chronological_split_date": meta["split_date"],
            "baseline_mae": m["baseline_mae"],
            "model_mae": m["model_mae"],
            "model_rmse": m["model_rmse"],
            "feature_importance": m["feature_importances"]
        }

    def _train(self):
        df = data_loader.daily_sales.copy()
        df["date"] = pd.to_datetime(df["date"])
        # Chronological sort
        df = df.sort_values(by="date").reset_index(drop=True)

        # Categorical encodings
        unique_pids = sorted(df["product_id"].unique())
        self.product_encodings = {pid: i for i, pid in enumerate(unique_pids)}

        unique_dows = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        self.dow_encodings = {dow: i for i, dow in enumerate(unique_dows)}

        df["pid_enc"] = df["product_id"].map(self.product_encodings)
        df["dow_enc"] = df["day_of_week"].map(self.dow_encodings)

        # Feature selection
        features = ["pid_enc", "selling_price", "dow_enc", "promotion_active", "stock_available"]
        self.feature_columns = features
        X = df[features]
        y = df["quantity_sold"]

        # Chronological 80/20 train/test split without date leakage
        split_idx = int(len(df) * 0.8)
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

        # Simple baseline: Historical average sales per (product_id, day_of_week)
        train_df = df.iloc[:split_idx]
        baseline_lookup = train_df.groupby(["pid_enc", "dow_enc"])["quantity_sold"].mean().to_dict()
        overall_mean = train_df["quantity_sold"].mean()

        baseline_preds = []
        for _, row in X_test.iterrows():
            key = (row["pid_enc"], row["dow_enc"])
            baseline_preds.append(baseline_lookup.get(key, overall_mean))
        
        self.baseline_mae = float(mean_absolute_error(y_test, baseline_preds))

        # Train Random Forest Regressor
        rf = RandomForestRegressor(
            n_estimators=150,
            max_depth=6,
            min_samples_split=4,
            min_samples_leaf=2,
            random_state=42
        )
        rf.fit(X_train, y_train)
        self.model = rf

        rf_preds = rf.predict(X_test)
        self.model_mae = float(mean_absolute_error(y_test, rf_preds))
        self.model_rmse = float(np.sqrt(mean_squared_error(y_test, rf_preds)))

        self.train_summary = {
            "model_type": "RandomForestRegressor",
            "train_samples": len(X_train),
            "test_samples": len(X_test),
            "chronological_split_date": str(df.iloc[split_idx]["date"].strftime("%Y-%m-%d")),
            "baseline_mae": round(self.baseline_mae, 3),
            "model_mae": round(self.model_mae, 3),
            "model_rmse": round(self.model_rmse, 3),
            "mae_improvement_pct": round(((self.baseline_mae - self.model_mae) / self.baseline_mae) * 100.0, 1),
            "feature_importance": dict(zip(features, [round(float(fi), 3) for fi in rf.feature_importances_]))
        }

    def predict_demand(
        self,
        product_id: str,
        selling_price: float,
        day_of_week: str = "Wednesday",
        promotion_active: int = 0,
        stock_available: int = None
    ) -> float:
        """
        Estimates demand quantity given a candidate selling price and scenario parameters.
        Responds dynamically to price variation.
        """
        if product_id not in self.product_encodings:
            raise KeyError(f"Unknown product {product_id}")

        if stock_available is None:
            # Default to average available stock from product definition
            prod = data_loader.get_product(product_id)
            stock_available = prod.get("available_stock", 60)

        pid_enc = self.product_encodings[product_id]
        dow_enc = self.dow_encodings.get(day_of_week, 2) # default Wednesday

        row = pd.DataFrame([[pid_enc, selling_price, dow_enc, promotion_active, stock_available]], columns=self.feature_columns)
        pred = float(self.model.predict(row)[0])
        # Demand cannot be negative and cannot exceed practical stock availability
        pred = max(0.0, min(float(stock_available), pred))
        return round(pred, 1)

# Global singleton estimator
demand_estimator = DemandEstimator()
