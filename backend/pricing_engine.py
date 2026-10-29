from typing import List, Dict, Any, Optional
from backend.data_loader import data_loader
from backend.demand_model import demand_estimator

def evaluate_price_scenarios(
    product_id: str,
    candidate_prices: Optional[List[float]] = None,
    day_of_week: str = "Wednesday",
    promotion_active: int = 0
) -> Dict[str, Any]:
    """
    Evaluates financial scenarios across candidate prices for a single product.
    Calculates:
      Revenue = candidate_price * estimated_quantity
      Unit variable cost = ingredient_cost + packaging_cost
      Variable cost = unit variable cost * estimated_quantity
      Contribution = revenue - variable cost
      Contribution change = candidate contribution - baseline contribution
    """
    prod = data_loader.get_product(product_id)
    current_price = float(prod["current_price"])
    unit_var_cost = float(prod["unit_variable_cost"])
    stock = int(prod["available_stock"])

    if not candidate_prices:
        # Generate 5 realistic price candidates around current price
        step = 5.0
        candidate_prices = [
            max(unit_var_cost + 1.0, current_price - 2 * step),
            max(unit_var_cost + 1.0, current_price - step),
            current_price,
            current_price + step,
            current_price + 2 * step
        ]
        # Deduplicate and sort
        candidate_prices = sorted(list(set([round(p, 1) for p in candidate_prices])))

    # 1. Evaluate baseline at current_price
    baseline_qty = demand_estimator.predict_demand(
        product_id=product_id,
        selling_price=current_price,
        day_of_week=day_of_week,
        promotion_active=promotion_active,
        stock_available=stock
    )
    baseline_rev = current_price * baseline_qty
    baseline_var_cost = unit_var_cost * baseline_qty
    baseline_contrib = baseline_rev - baseline_var_cost

    scenario_items = []
    for price in candidate_prices:
        est_qty = demand_estimator.predict_demand(
            product_id=product_id,
            selling_price=price,
            day_of_week=day_of_week,
            promotion_active=promotion_active,
            stock_available=stock
        )
        rev = price * est_qty
        v_cost = unit_var_cost * est_qty
        contrib = rev - v_cost
        delta_contrib = contrib - baseline_contrib
        delta_rev = rev - baseline_rev

        scenario_items.append({
            "candidate_price": round(price, 2),
            "is_baseline": bool(abs(price - current_price) < 1e-4),
            "estimated_daily_quantity": est_qty,
            "estimated_daily_revenue": round(rev, 2),
            "unit_variable_cost": round(unit_var_cost, 2),
            "estimated_daily_variable_cost": round(v_cost, 2),
            "estimated_daily_contribution": round(contrib, 2),
            "contribution_change_vs_baseline": round(delta_contrib, 2),
            "revenue_change_vs_baseline": round(delta_rev, 2)
        })

    return {
        "product": prod,
        "day_of_week": day_of_week,
        "promotion_active": promotion_active,
        "scenarios": scenario_items,
        "model_notes": (
            "Illustrative synthetic estimates. Demand responds dynamically to price. "
            "Contribution is calculated before fixed canteen overheads. "
            "Validation with real experimental or sales data is recommended before operational adjustments."
        )
    }
