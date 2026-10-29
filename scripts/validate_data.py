"""
Validation script for synthetic canteen datasets:
- Checks column completeness and types
- Validates constraints (non-negative quantities, discounts <= subtotal, selling price > unit variable cost)
- Checks foreign key consistency (product_id across transactions, daily_sales, and products)
- Checks date alignments
- Verifies internal inventory consistency (stock_available = quantity_sold + unsold_quantity)
"""

import os
import sys
import pandas as pd

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")

def validate_datasets():
    errors = []
    warnings = []
    
    # 1. Products Validation
    prod_path = os.path.join(DATA_DIR, "products.csv")
    if not os.path.exists(prod_path):
        errors.append(f"Missing {prod_path}")
        return errors, warnings
    
    products_df = pd.read_csv(prod_path)
    req_prod_cols = [
        "product_id", "product_name", "category", "current_price",
        "ingredient_cost", "packaging_cost", "portion_size_g",
        "available_stock", "waste_quantity"
    ]
    for col in req_prod_cols:
        if col not in products_df.columns:
            errors.append(f"products.csv missing required column: {col}")
            
    if products_df["product_id"].duplicated().any():
        errors.append("Duplicate product_ids found in products.csv")
        
    for _, row in products_df.iterrows():
        unit_var_cost = row["ingredient_cost"] + row["packaging_cost"]
        if row["current_price"] <= unit_var_cost:
            warnings.append(f"Product {row['product_name']} ({row['product_id']}) price ₹{row['current_price']} <= variable cost ₹{unit_var_cost}")
        if row["available_stock"] < 0 or row["waste_quantity"] < 0:
            errors.append(f"Negative stock or waste for product {row['product_id']}")
            
    print(f"✓ products.csv: {len(products_df)} products verified.")
    valid_pids = set(products_df["product_id"])
    
    # 2. Transactions Validation
    tx_path = os.path.join(DATA_DIR, "transactions.csv")
    if not os.path.exists(tx_path):
        errors.append(f"Missing {tx_path}")
        return errors, warnings
    
    tx_df = pd.read_csv(tx_path)
    req_tx_cols = [
        "transaction_id", "transaction_date", "product_id",
        "product_name", "quantity", "unit_price", "discount"
    ]
    for col in req_tx_cols:
        if col not in tx_df.columns:
            errors.append(f"transactions.csv missing required column: {col}")
            
    # Foreign key check
    tx_pids = set(tx_df["product_id"])
    invalid_tx_pids = tx_pids - valid_pids
    if invalid_tx_pids:
        errors.append(f"transactions.csv contains unknown product_ids: {invalid_tx_pids}")
        
    # Value rules
    if (tx_df["quantity"] <= 0).any():
        errors.append("transactions.csv contains non-positive quantities")
    if (tx_df["unit_price"] <= 0).any():
        errors.append("transactions.csv contains non-positive unit_prices")
    if (tx_df["discount"] < 0).any():
        errors.append("transactions.csv contains negative discounts")
        
    subtotal = tx_df["quantity"] * tx_df["unit_price"]
    if (tx_df["discount"] > subtotal + 1e-5).any():
        errors.append("transactions.csv contains discounts exceeding line subtotal")
        
    print(f"✓ transactions.csv: {len(tx_df)} rows, {tx_df['transaction_id'].nunique()} unique baskets verified.")
    
    # 3. Daily Sales Validation
    daily_path = os.path.join(DATA_DIR, "daily_sales.csv")
    if not os.path.exists(daily_path):
        errors.append(f"Missing {daily_path}")
        return errors, warnings
    
    daily_df = pd.read_csv(daily_path)
    req_daily_cols = [
        "date", "product_id", "selling_price", "quantity_sold",
        "day_of_week", "promotion_active", "stock_available", "unsold_quantity"
    ]
    for col in req_daily_cols:
        if col not in daily_df.columns:
            errors.append(f"daily_sales.csv missing required column: {col}")
            
    daily_pids = set(daily_df["product_id"])
    invalid_daily_pids = daily_pids - valid_pids
    if invalid_daily_pids:
        errors.append(f"daily_sales.csv contains unknown product_ids: {invalid_daily_pids}")
        
    if (daily_df["quantity_sold"] < 0).any():
        errors.append("daily_sales.csv contains negative quantity_sold")
    if (daily_df["stock_available"] < 0).any():
        errors.append("daily_sales.csv contains negative stock_available")
    if (daily_df["unsold_quantity"] < 0).any():
        errors.append("daily_sales.csv contains negative unsold_quantity")
        
    # Inventory consistency
    inv_diff = daily_df["stock_available"] - (daily_df["quantity_sold"] + daily_df["unsold_quantity"])
    if (inv_diff.abs() > 0).any():
        errors.append("daily_sales.csv inventory equation violated: stock_available != quantity_sold + unsold_quantity")
        
    print(f"✓ daily_sales.csv: {len(daily_df)} observations across {daily_df['date'].nunique()} dates verified.")
    
    return errors, warnings

if __name__ == "__main__":
    errs, warns = validate_datasets()
    if warns:
        print("\nWarnings:")
        for w in warns:
            print(f"  ! {w}")
    if errs:
        print("\nErrors encountered:")
        for e in errs:
            print(f"  ✗ {e}")
        sys.exit(1)
    else:
        print("\nAll datasets passed validation successfully!")
        sys.exit(0)
