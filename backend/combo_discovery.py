import pandas as pd
from typing import List, Dict, Any
from mlxtend.frequent_patterns import apriori, fpgrowth, association_rules
from backend.data_loader import data_loader

def discover_combos(
    algorithm: str = "fpgrowth",
    min_support: float = 0.03,
    min_confidence: float = 0.20,
    min_lift: float = 1.0
) -> List[Dict[str, Any]]:
    """
    Groups transaction records into baskets and executes Apriori or FP-Growth.
    Calculates Support, Confidence, Lift, combined prices, and variable costs.
    """
    tx_df = data_loader.transactions
    prods_df = data_loader.products
    prod_meta = prods_df.set_index("product_id").to_dict(orient="index")

    # Basket matrix: 1 if product in transaction_id, else 0
    # Distinct items per order
    basket_df = tx_df.groupby(["transaction_id", "product_id"])["quantity"].count().unstack().fillna(0)
    # Convert to bool
    basket_bool = basket_df.map(lambda x: True if x > 0 else False)

    # Frequent itemsets
    if algorithm.lower() == "apriori":
        frequent_itemsets = apriori(basket_bool, min_support=min_support, use_colnames=True)
    else:
        frequent_itemsets = fpgrowth(basket_bool, min_support=min_support, use_colnames=True)

    if frequent_itemsets.empty:
        return []

    # Association rules
    rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=min_confidence)
    if rules.empty:
        return []

    # Filter by minimum lift
    rules = rules[rules["lift"] >= min_lift]
    if rules.empty:
        return []

    # Format output rules
    discovered = []
    seen_combos = set()

    for idx, row in rules.iterrows():
        antecedents = list(row["antecedents"])
        consequents = list(row["consequents"])
        all_items = sorted(list(set(antecedents + consequents)))

        # Unique combo signature
        combo_sig = tuple(all_items)
        if combo_sig in seen_combos:
            continue
        seen_combos.add(combo_sig)

        item_names = [prod_meta.get(pid, {}).get("product_name", pid) for pid in all_items]
        sum_prices = sum(prod_meta.get(pid, {}).get("current_price", 0.0) for pid in all_items)
        sum_var_cost = sum(prod_meta.get(pid, {}).get("unit_variable_cost", 0.0) for pid in all_items)
        default_contrib = sum_prices - sum_var_cost

        discovered.append({
            "combo_id": f"COMBO_{'-'.join(all_items)}",
            "antecedents": antecedents,
            "consequents": consequents,
            "items": all_items,
            "item_names": item_names,
            "support": round(float(row["support"]), 4),
            "confidence": round(float(row["confidence"]), 4),
            "lift": round(float(row["lift"]), 3),
            "sum_individual_prices": round(sum_prices, 2),
            "total_variable_cost": round(sum_var_cost, 2),
            "default_combo_contribution": round(default_contrib, 2)
        })

    # Sort by lift descending
    discovered.sort(key=lambda x: x["lift"], reverse=True)
    return discovered
