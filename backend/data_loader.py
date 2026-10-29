import os
import pandas as pd
from typing import Dict, Any, List

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")

class DataLoader:
    def __init__(self, data_dir: str = DATA_DIR):
        self.data_dir = data_dir
        self._products_df = None
        self._transactions_df = None
        self._daily_sales_df = None
        self.reload()

    def reload(self):
        prod_path = os.path.join(self.data_dir, "products.csv")
        tx_path = os.path.join(self.data_dir, "transactions.csv")
        daily_path = os.path.join(self.data_dir, "daily_sales.csv")

        if not os.path.exists(prod_path) or not os.path.exists(tx_path) or not os.path.exists(daily_path):
            raise FileNotFoundError(f"Missing CSV files in {self.data_dir}. Run scripts/generate_data.py first.")

        self._products_df = pd.read_csv(prod_path)
        self._transactions_df = pd.read_csv(tx_path)
        self._daily_sales_df = pd.read_csv(daily_path)

        # Derived fields
        self._products_df["unit_variable_cost"] = (
            self._products_df["ingredient_cost"] + self._products_df["packaging_cost"]
        )
        self._products_df["current_unit_contribution"] = (
            self._products_df["current_price"] - self._products_df["unit_variable_cost"]
        )
        self._products_df["contribution_margin_pct"] = (
            (self._products_df["current_unit_contribution"] / self._products_df["current_price"]) * 100.0
        ).round(1)

    @property
    def products(self) -> pd.DataFrame:
        return self._products_df

    @property
    def transactions(self) -> pd.DataFrame:
        return self._transactions_df

    @property
    def daily_sales(self) -> pd.DataFrame:
        return self._daily_sales_df

    def get_product(self, product_id: str) -> Dict[str, Any]:
        match = self._products_df[self._products_df["product_id"] == product_id]
        if match.empty:
            raise KeyError(f"Product {product_id} not found.")
        return match.iloc[0].to_dict()

    def get_overview_metrics(self) -> Dict[str, Any]:
        tx = self._transactions_df
        prods = self._products_df.set_index("product_id")

        total_tx = int(tx["transaction_id"].nunique())
        total_units = int(tx["quantity"].sum())
        
        # Line revenue = (quantity * unit_price) - discount
        line_rev = (tx["quantity"] * tx["unit_price"]) - tx["discount"]
        total_rev = float(line_rev.sum())

        # Line variable cost
        tx_merged = tx.merge(
            self._products_df[["product_id", "unit_variable_cost"]],
            on="product_id",
            how="left"
        )
        total_var_cost = float((tx_merged["quantity"] * tx_merged["unit_variable_cost"]).sum())
        total_contrib = total_rev - total_var_cost
        overall_margin = (total_contrib / total_rev * 100.0) if total_rev > 0 else 0.0

        # Aggregation per product
        prod_perf = tx_merged.groupby(["product_id", "product_name"]).agg(
            units_sold=("quantity", "sum"),
            revenue=("unit_price", lambda s: float((tx.loc[s.index, "quantity"] * s - tx.loc[s.index, "discount"]).sum()))
        ).reset_index()

        prod_perf = prod_perf.merge(
            self._products_df[["product_id", "current_price", "unit_variable_cost", "category"]],
            on="product_id",
            how="left"
        )
        prod_perf["contribution"] = prod_perf["revenue"] - (prod_perf["units_sold"] * prod_perf["unit_variable_cost"])
        prod_perf["margin_pct"] = (prod_perf["contribution"] / prod_perf["revenue"] * 100.0).round(1)

        best_selling = prod_perf.sort_values(by="units_sold", ascending=False).to_dict(orient="records")
        slow_moving = prod_perf.sort_values(by="units_sold", ascending=True).to_dict(orient="records")

        date_start = str(tx["transaction_date"].min())
        date_end = str(tx["transaction_date"].max())

        return {
            "total_transactions": total_tx,
            "total_units_sold": total_units,
            "total_revenue": round(total_rev, 2),
            "total_estimated_contribution": round(total_contrib, 2),
            "total_variable_cost": round(total_var_cost, 2),
            "overall_contribution_margin_pct": round(overall_margin, 1),
            "date_range_start": date_start,
            "date_range_end": date_end,
            "best_selling_products": best_selling,
            "slow_moving_products": slow_moving
        }

# Global singleton loader
data_loader = DataLoader()
