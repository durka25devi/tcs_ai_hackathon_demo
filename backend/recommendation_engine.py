from typing import List, Dict, Any
from backend.data_loader import data_loader
from backend.combo_discovery import discover_combos
from backend.pricing_engine import evaluate_price_scenarios

def generate_recommendations(
    max_student_budget: float = 75.0,
    min_contribution_margin: float = 0.25,
    stock_buffer_safety_pct: float = 0.85
) -> Dict[str, Any]:
    """
    Evaluates feasible pricing optimizations and combo deals.
    Objective: Maximize contribution while respecting student budget, margin, and stock constraints.
    Returns:
      - Pricing recommendations for single products
      - Combo recommendations based on association rules
      - Rationales and uncertainty warnings
    """
    prods_df = data_loader.products
    pricing_recs = []

    # 1. Product Price Recommendations
    for _, row in prods_df.iterrows():
        pid = row["product_id"]
        pname = row["product_name"]
        cur_p = float(row["current_price"])

        # Evaluate candidate variations: ±₹5, ±₹10
        candidates = [
            max(row["unit_variable_cost"] + 2.0, cur_p - 10.0),
            max(row["unit_variable_cost"] + 2.0, cur_p - 5.0),
            cur_p,
            cur_p + 5.0,
            cur_p + 10.0
        ]
        candidates = sorted(list(set([round(c, 1) for c in candidates])))

        scenarios = evaluate_price_scenarios(
            product_id=pid,
            candidate_prices=candidates,
            day_of_week="Wednesday",
            promotion_active=0
        )["scenarios"]

        baseline_scenario = next(s for s in scenarios if s["is_baseline"])
        baseline_contrib = baseline_scenario["estimated_daily_contribution"]

        # Filter feasible candidates: candidate_price <= max_student_budget and contribution margin >= min_margin
        feasible = []
        for s in scenarios:
            margin = (s["candidate_price"] - s["unit_variable_cost"]) / s["candidate_price"]
            if s["candidate_price"] <= max_student_budget and margin >= min_contribution_margin:
                feasible.append(s)

        if not feasible:
            feasible = [baseline_scenario]

        # Select candidate maximizing contribution
        best_scenario = max(feasible, key=lambda s: s["estimated_daily_contribution"])
        gain = best_scenario["estimated_daily_contribution"] - baseline_contrib
        p_diff = best_scenario["candidate_price"] - cur_p

        if abs(p_diff) < 1e-4:
            status = "MAINTAIN"
            rationale = (
                f"Current price ₹{cur_p:.0f} is already optimal under current demand estimates, "
                f"yielding estimated daily contribution of ₹{baseline_contrib:.1f}."
            )
        elif p_diff > 0:
            status = "INCREASE"
            rationale = (
                f"Slight price elasticity allows raising price from ₹{cur_p:.0f} to ₹{best_scenario['candidate_price']:.0f}, "
                f"projected to increase daily contribution by +₹{gain:.1f} (+{(gain/baseline_contrib*100):.1f}%)."
            )
        else:
            status = "REDUCE"
            rationale = (
                f"Lowering price from ₹{cur_p:.0f} to ₹{best_scenario['candidate_price']:.0f} stimulates higher volume "
                f"({best_scenario['estimated_daily_quantity']:.1f} units), projected to add +₹{gain:.1f} contribution."
            )

        pricing_recs.append({
            "product_id": pid,
            "product_name": pname,
            "current_price": cur_p,
            "recommended_price": best_scenario["candidate_price"],
            "price_change": round(p_diff, 1),
            "baseline_contribution": round(baseline_contrib, 2),
            "recommended_contribution": round(best_scenario["estimated_daily_contribution"], 2),
            "projected_contribution_gain": round(gain, 2),
            "projected_daily_demand": best_scenario["estimated_daily_quantity"],
            "status": status,
            "rationale": rationale
        })

    # Sort pricing recs by potential gain descending
    pricing_recs.sort(key=lambda x: x["projected_contribution_gain"], reverse=True)

    # 2. Combo Recommendations from Discovered Rules
    discovered = discover_combos(algorithm="fpgrowth", min_support=0.03, min_confidence=0.20, min_lift=1.2)
    combo_recs = []

    for rule in discovered:
        sum_p = rule["sum_individual_prices"]
        var_c = rule["total_variable_cost"]
        item_pids = rule["items"]

        # Find bottleneck stock among items
        bottleneck_stock = min(
            int(prods_df[prods_df["product_id"] == pid]["available_stock"].values[0])
            for pid in item_pids
        )
        safe_capacity = int(bottleneck_stock * stock_buffer_safety_pct)

        # Candidate combo discount: 10% or 15% discount
        for discount_rate in [0.12, 0.15, 0.10]:
            candidate_combo_p = round((sum_p * (1.0 - discount_rate)) / 5.0) * 5.0 # Round to nearest ₹5
            contrib_combo = candidate_combo_p - var_c
            margin_pct = (contrib_combo / candidate_combo_p) if candidate_combo_p > 0 else 0

            # Constraints check
            if candidate_combo_p <= max_student_budget and margin_pct >= min_contribution_margin and candidate_combo_p < sum_p:
                savings = sum_p - candidate_combo_p
                combo_recs.append({
                    "combo_name": " + ".join(rule["item_names"]),
                    "product_ids": item_pids,
                    "individual_total_price": sum_p,
                    "recommended_combo_price": candidate_combo_p,
                    "customer_savings": round(savings, 2),
                    "customer_savings_pct": round(savings / sum_p * 100.0, 1),
                    "contribution_per_combo": round(contrib_combo, 2),
                    "contribution_margin_pct": round(margin_pct * 100.0, 1),
                    "daily_bottleneck_capacity": safe_capacity,
                    "support": rule["support"],
                    "confidence": rule["confidence"],
                    "lift": rule["lift"],
                    "rationale": (
                        f"Strong empirical co-purchase pattern (Lift: {rule['lift']}x, Confidence: {rule['confidence']*100:.1f}%). "
                        f"Attractive student savings of ₹{savings:.0f} ({(savings/sum_p*100):.0f}% off) while maintaining healthy "
                        f"{margin_pct*100:.1f}% contribution margin under the ₹{max_student_budget:.0f} student budget."
                    )
                })
                break # Only top pricing tier per rule

    return {
        "objective": "Maximize daily contribution while respecting student budget and inventory limits",
        "student_budget_limit": max_student_budget,
        "min_margin_threshold": min_contribution_margin,
        "pricing_recommendations": pricing_recs,
        "combo_recommendations": combo_recs,
        "general_caveats": [
            "Decision support only: Actual price changes require canteen manager review and approval.",
            "Synthetic data baseline: Elasticity estimates must be validated with real-world A/B menu testing.",
            "Combo substitution effect: Assumes partial incrementality without full consumer cannibalization modeling."
        ]
    }
