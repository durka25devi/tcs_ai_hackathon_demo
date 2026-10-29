"""
Data generation script for AI-Powered Pricing and Combo Offer Support System.
Generates realistic, internally consistent synthetic datasets for a college canteen:
- products.csv: Catalog with costs, prices, stock, portion sizes, and waste
- transactions.csv: 3,000-5,000 basket transaction lines across 90 days
- daily_sales.csv: Daily product sales, prices, promotions, and inventory
"""

import os
import random
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
os.makedirs(DATA_DIR, exist_ok=True)

# 1. Products Definition
# Units:
# - current_price, ingredient_cost, packaging_cost in INR (₹)
# - portion_size_g in grams
# - available_stock represents daily average batch prepared or stocked in units/portions
# - waste_quantity represents daily average unsold/discarded units under standard operations
PRODUCTS = [
    {
        "product_id": "P101",
        "product_name": "Veg Sandwich",
        "category": "Snacks",
        "current_price": 40.0,
        "ingredient_cost": 16.0,
        "packaging_cost": 2.0,
        "portion_size_g": 180,
        "available_stock": 70,
        "waste_quantity": 4
    },
    {
        "product_id": "P102",
        "product_name": "Veg Burger",
        "category": "Snacks",
        "current_price": 50.0,
        "ingredient_cost": 22.0,
        "packaging_cost": 3.0,
        "portion_size_g": 210,
        "available_stock": 65,
        "waste_quantity": 3
    },
    {
        "product_id": "P103",
        "product_name": "French Fries",
        "category": "Sides",
        "current_price": 35.0,
        "ingredient_cost": 12.0,
        "packaging_cost": 3.0,
        "portion_size_g": 130,
        "available_stock": 80,
        "waste_quantity": 5
    },
    {
        "product_id": "P104",
        "product_name": "Lemon Juice",
        "category": "Beverages",
        "current_price": 20.0,
        "ingredient_cost": 5.0,
        "packaging_cost": 2.0,
        "portion_size_g": 250,
        "available_stock": 90,
        "waste_quantity": 4
    },
    {
        "product_id": "P105",
        "product_name": "Cold Coffee",
        "category": "Beverages",
        "current_price": 35.0,
        "ingredient_cost": 14.0,
        "packaging_cost": 3.0,
        "portion_size_g": 250,
        "available_stock": 60,
        "waste_quantity": 3
    },
    {
        "product_id": "P106",
        "product_name": "Samosa",
        "category": "Snacks",
        "current_price": 15.0,
        "ingredient_cost": 5.0,
        "packaging_cost": 1.0,
        "portion_size_g": 90,
        "available_stock": 120,
        "waste_quantity": 6
    },
    {
        "product_id": "P107",
        "product_name": "Paneer Roll",
        "category": "Snacks",
        "current_price": 55.0,
        "ingredient_cost": 25.0,
        "packaging_cost": 3.0,
        "portion_size_g": 200,
        "available_stock": 50,
        "waste_quantity": 3
    },
    {
        "product_id": "P108",
        "product_name": "Water Bottle",
        "category": "Beverages",
        "current_price": 15.0,
        "ingredient_cost": 9.0,
        "packaging_cost": 1.0,
        "portion_size_g": 500,
        "available_stock": 150,
        "waste_quantity": 1
    }
]

def generate_products_csv():
    df = pd.DataFrame(PRODUCTS)
    filepath = os.path.join(DATA_DIR, "products.csv")
    df.to_csv(filepath, index=False)
    print(f"Generated products.csv with {len(df)} products at {filepath}")
    return df

def generate_transactions_and_daily_sales():
    # 90-day simulation
    start_date = datetime(2026, 8, 1) # College semester opening
    num_days = 90
    
    prod_lookup = {p["product_id"]: p for p in PRODUCTS}
    
    # Natural affinities for college canteen baskets:
    # 1. Veg Burger + French Fries (classic fast food combo)
    # 2. Veg Sandwich + Lemon Juice (healthy morning / evening snack)
    # 3. Veg Burger + Cold Coffee (indulgence afternoon combo)
    # 4. Samosa + Water Bottle (quick cheap snack)
    # 5. Paneer Roll + Cold Coffee (heavy meal combo)
    # 6. Samosa alone (very common individual purchase)
    
    transactions = []
    transaction_counter = 10001
    
    # Daily sales aggregator: key = (date_str, product_id) -> stats
    daily_stats = {}
    for d in range(num_days):
        cur_date = start_date + timedelta(days=d)
        date_str = cur_date.strftime("%Y-%m-%d")
        dow = cur_date.strftime("%A")
        
        # Periodic promotions: e.g. Friday college festive discount or exam-prep boost
        is_promo_day = (d % 14 in [5, 6, 12]) # Approx 2 days every 2 weeks
        
        # College traffic pattern: higher Mon-Fri, lower on Sunday
        if dow == "Sunday":
            num_orders = random.randint(18, 30)
            traffic_mult = 0.5
        elif dow in ["Saturday"]:
            num_orders = random.randint(30, 45)
            traffic_mult = 0.75
        elif dow in ["Friday"]:
            num_orders = random.randint(55, 75)
            traffic_mult = 1.2
        else:
            num_orders = random.randint(45, 65)
            traffic_mult = 1.0
            
        # Select prices for products today (price variation on some days for elasticity estimation)
        # We simulate price tests: e.g., sandwich at ₹35, ₹40, ₹45 on specific weeks
        daily_prices = {}
        for p in PRODUCTS:
            pid = p["product_id"]
            base_p = p["current_price"]
            if pid in ["P101", "P102", "P103", "P105", "P107"]:
                # Varied across test periods
                period_phase = (d // 15) % 3
                if period_phase == 0:
                    effective_p = base_p
                elif period_phase == 1:
                    effective_p = base_p - 5.0 # Discounted test
                else:
                    effective_p = base_p + 5.0 # Premium test
            else:
                effective_p = base_p
            daily_prices[pid] = effective_p
            
        for _ in range(num_orders):
            tx_id = f"TX{transaction_counter}"
            transaction_counter += 1
            
            # Basket selection strategy
            basket_type = random.choices(
                ["single", "burger_fries", "sandwich_juice", "burger_coffee", "samosa_water", "roll_coffee", "mixed_trio"],
                weights=[0.30, 0.18, 0.16, 0.12, 0.10, 0.08, 0.06]
            )[0]
            
            items_to_add = []
            if basket_type == "single":
                p_choice = random.choices(
                    ["P106", "P101", "P104", "P108", "P102", "P105", "P107", "P103"],
                    weights=[0.30, 0.18, 0.12, 0.12, 0.10, 0.08, 0.05, 0.05]
                )[0]
                items_to_add.append(p_choice)
            elif basket_type == "burger_fries":
                items_to_add.extend(["P102", "P103"])
                if random.random() < 0.25: # Sometimes add cold beverage
                    items_to_add.append("P105")
            elif basket_type == "sandwich_juice":
                items_to_add.extend(["P101", "P104"])
                if random.random() < 0.15:
                    items_to_add.append("P103")
            elif basket_type == "burger_coffee":
                items_to_add.extend(["P102", "P105"])
            elif basket_type == "samosa_water":
                items_to_add.extend(["P106", "P108"])
            elif basket_type == "roll_coffee":
                items_to_add.extend(["P107", "P105"])
            elif basket_type == "mixed_trio":
                items_to_add.extend(["P101", "P103", "P104"])
                
            # If price was higher, slight drop in quantity probability
            for pid in set(items_to_add):
                prod_meta = prod_lookup[pid]
                unit_price = daily_prices[pid]
                
                # Quantity: mostly 1, occasionally 2, rarely 3
                qty = random.choices([1, 2, 3], weights=[0.85, 0.12, 0.03])[0]
                
                # Promotional line discount (e.g. ₹2-₹5 on promo days)
                item_subtotal = unit_price * qty
                if is_promo_day and random.random() < 0.35:
                    discount = round(min(item_subtotal * 0.15, random.choice([3.0, 5.0, 8.0])), 2)
                else:
                    discount = 0.0
                    
                transactions.append({
                    "transaction_id": tx_id,
                    "transaction_date": date_str,
                    "product_id": pid,
                    "product_name": prod_meta["product_name"],
                    "quantity": qty,
                    "unit_price": unit_price,
                    "discount": discount
                })
                
                # Accumulate daily sales
                key = (date_str, pid)
                if key not in daily_stats:
                    daily_stats[key] = {
                        "date": date_str,
                        "product_id": pid,
                        "selling_price": unit_price,
                        "quantity_sold": 0,
                        "day_of_week": dow,
                        "promotion_active": 1 if is_promo_day else 0,
                    }
                daily_stats[key]["quantity_sold"] += qty

    tx_df = pd.DataFrame(transactions)
    tx_filepath = os.path.join(DATA_DIR, "transactions.csv")
    tx_df.to_csv(tx_filepath, index=False)
    print(f"Generated transactions.csv with {len(tx_df)} rows and {tx_df['transaction_id'].nunique()} unique orders.")
    
    # 3. Generate daily_sales.csv
    # Ensure every product has a row for every date
    daily_records = []
    for d in range(num_days):
        cur_date = start_date + timedelta(days=d)
        date_str = cur_date.strftime("%Y-%m-%d")
        dow = cur_date.strftime("%A")
        is_promo_day = (d % 14 in [5, 6, 12])
        
        for p in PRODUCTS:
            pid = p["product_id"]
            key = (date_str, pid)
            base_stock = p["available_stock"]
            # Weekend stock adjustment
            if dow == "Sunday":
                stock_avail = int(base_stock * 0.55)
            elif dow == "Saturday":
                stock_avail = int(base_stock * 0.75)
            else:
                stock_avail = base_stock + random.randint(-5, 8)
                
            if key in daily_stats:
                stat = daily_stats[key]
                selling_price = stat["selling_price"]
                qty_sold = stat["quantity_sold"]
            else:
                # 0 sales on this day
                selling_price = p["current_price"]
                qty_sold = 0
                
            # Internal consistency: quantity_sold cannot exceed stock_available
            if qty_sold > stock_avail:
                stock_avail = qty_sold + random.randint(1, 4)
                
            unsold_quantity = stock_avail - qty_sold
            if unsold_quantity < 0:
                unsold_quantity = 0
                
            daily_records.append({
                "date": date_str,
                "product_id": pid,
                "selling_price": selling_price,
                "quantity_sold": qty_sold,
                "day_of_week": dow,
                "promotion_active": 1 if is_promo_day else 0,
                "stock_available": stock_avail,
                "unsold_quantity": unsold_quantity
            })
            
    daily_df = pd.DataFrame(daily_records)
    daily_filepath = os.path.join(DATA_DIR, "daily_sales.csv")
    daily_df.to_csv(daily_filepath, index=False)
    print(f"Generated daily_sales.csv with {len(daily_df)} product-day observations.")

if __name__ == "__main__":
    generate_products_csv()
    generate_transactions_and_daily_sales()
