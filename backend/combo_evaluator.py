from typing import List, Dict, Any
from backend.data_loader import data_loader

def evaluate_combo_offer(
    combo_items: List[str],
    proposed_price: float,
    assumed_daily_volume: int = 25,
    min_contribution_margin: float = 0.25,
    student_budget_limit: float = 80.0
) -> Dict[str, Any]:
    """
    Evaluates financial and operational feasibility of a candidate combo offer.
    Constraints:
      - combo_price > 0
      - combo_price <= sum of individual prices (for discounted offers)
      - contribution margin >= min_contribution_margin
      - student budget constraint
      - inventory bottleneck constraint
    Clearly distinguishes unit contribution vs hypothetical scenario contribution.
    """
    if not combo_items:
        raise ValueError("Combo must contain at least one product.")

    products_meta = []
    sum_individual_prices = 0.0
    combo_variable_cost = 0.0
    stock_levels = {}

    for pid in combo_items:
        p = data_loader.get_product(pid)
        products_meta.append(p)
        sum_individual_prices += float(p["current_price"])
        combo_variable_cost += float(p["unit_variable_cost"])
        stock_levels[p["product_name"]] = int(p["available_stock"])

    customer_savings = sum_individual_prices - proposed_price
    customer_savings_pct = (customer_savings / sum_individual_prices * 100.0) if sum_individual_prices > 0 else 0.0

    contribution_per_combo = proposed_price - combo_variable_cost
    margin_pct = (contribution_per_combo / proposed_price * 100.0) if proposed_price > 0 else 0.0

    # Bottleneck stock is the minimum stock among participating items
    bottleneck_product = min(stock_levels, key=stock_levels.get)
    max_bottleneck_stock = stock_levels[bottleneck_product]

    # Constraints verification
    warnings = []
    passes_budget = bool(proposed_price <= student_budget_limit)
    if not passes_budget:
        warnings.append(
            f"Proposed combo price ₹{proposed_price:.1f} exceeds student budget limit of ₹{student_budget_limit:.1f}."
        )

    passes_margin = bool((margin_pct / 100.0) >= min_contribution_margin)
    if not passes_margin:
        warnings.append(
            f"Combo contribution margin {margin_pct:.1f}% is below minimum target {(min_contribution_margin*100):.1f}%."
        )

    passes_stock = bool(assumed_daily_volume <= max_bottleneck_stock)
    if not passes_stock:
        warnings.append(
            f"Assumed volume ({assumed_daily_volume} units) exceeds stock capacity of limiting component '{bottleneck_product}' ({max_bottleneck_stock} units)."
        )

    if proposed_price > sum_individual_prices:
        warnings.append(
            f"Proposed combo price ₹{proposed_price:.1f} is higher than individual sum ₹{sum_individual_prices:.1f} (negative customer savings)."
        )

    if contribution_per_combo <= 0:
        warnings.append("Proposed combo generates zero or negative contribution per unit sold.")

    hypothetical_rev = proposed_price * assumed_daily_volume
    hypothetical_contrib = contribution_per_combo * assumed_daily_volume

    return {
        "included_products": products_meta,
        "sum_individual_prices": round(sum_individual_prices, 2),
        "proposed_combo_price": round(proposed_price, 2),
        "customer_savings": round(customer_savings, 2),
        "customer_savings_pct": round(customer_savings_pct, 1),
        "combo_variable_cost": round(combo_variable_cost, 2),
        "contribution_per_combo": round(contribution_per_combo, 2),
        "contribution_margin_pct": round(margin_pct, 1),
        "assumed_daily_volume": assumed_daily_volume,
        "hypothetical_daily_revenue": round(hypothetical_rev, 2),
        "hypothetical_daily_contribution": round(hypothetical_contrib, 2),
        "max_bottleneck_stock": max_bottleneck_stock,
        "stock_bottleneck_product": bottleneck_product,
        "passes_budget": passes_budget,
        "passes_margin": passes_margin,
        "passes_stock": passes_stock,
        "constraint_status": {
            "within_student_budget": passes_budget,
            "meets_min_margin": passes_margin,
            "within_stock_limit": passes_stock,
            "positive_contribution": contribution_per_combo > 0,
            "valid_discount": customer_savings >= 0
        },
        "warnings": warnings,
        "caveat_notes": (
            "Hypothetical scenario: Assumes the specified daily volume without estimating cannibalization "
            "or cross-elasticity from individual items. Total contribution is illustrative."
        )
    }
