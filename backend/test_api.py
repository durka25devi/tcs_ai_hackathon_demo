"""
Integration tests for backend API routes:
1. /api/overview
2. /api/products
3. /api/combos
4. /api/pricing/evaluate
5. /api/combos/evaluate
6. /api/recommendations
7. /api/demand-model/metrics
8. /api/llm/status
9. /api/chat
"""

import os
import sys

# Ensure repository root is on sys.path so test runs cleanly without manual PYTHONPATH
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from fastapi.testclient import TestClient
from backend.main import app

client = TestClient(app)

def test_health():
    res = client.get("/api/health")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "healthy"
    assert data["products_count"] == 8
    assert "llm_model" in data

def test_overview():
    res = client.get("/api/overview")
    assert res.status_code == 200
    data = res.json()
    assert data["total_transactions"] > 1000
    assert data["total_revenue"] > 0
    assert data["total_estimated_contribution"] > 0
    assert len(data["best_selling_products"]) == 8

def test_products():
    res = client.get("/api/products")
    assert res.status_code == 200
    prods = res.json()
    assert len(prods) == 8
    p = prods[0]
    assert "product_id" in p
    assert p["current_price"] > p["unit_variable_cost"]

def test_combos_discovery():
    res = client.get("/api/combos?min_support=0.03&min_confidence=0.20&min_lift=1.0")
    assert res.status_code == 200
    rules = res.json()
    assert len(rules) > 0
    first_rule = rules[0]
    assert "support" in first_rule
    assert "confidence" in first_rule
    assert "lift" in first_rule
    assert first_rule["lift"] >= 1.0

def test_demand_model_metrics():
    res = client.get("/api/demand-model/metrics")
    assert res.status_code == 200
    metrics = res.json()
    assert "baseline_mae" in metrics
    assert "model_mae" in metrics
    assert metrics["baseline_mae"] > 0
    assert metrics["model_mae"] > 0

def test_pricing_evaluation():
    payload = {
        "product_id": "P101", # Veg Sandwich
        "candidate_prices": [30.0, 35.0, 40.0, 45.0, 50.0],
        "day_of_week": "Wednesday",
        "promotion_active": 0
    }
    res = client.post("/api/pricing/evaluate", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert len(data["scenarios"]) == 5
    baseline = next(s for s in data["scenarios"] if s["is_baseline"])
    assert baseline["candidate_price"] == 40.0
    assert baseline["contribution_change_vs_baseline"] == 0.0

def test_combo_evaluation():
    payload = {
        "combo_items": ["P102", "P103"], # Veg Burger + French Fries
        "proposed_price": 75.0,
        "assumed_daily_volume": 20,
        "min_contribution_margin": 0.25,
        "student_budget_limit": 80.0
    }
    res = client.post("/api/combos/evaluate", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert data["sum_individual_prices"] == 85.0
    assert data["customer_savings"] == 10.0
    assert data["passes_budget"] is True
    assert data["passes_margin"] is True
    assert data["passes_stock"] is True

def test_recommendations():
    payload = {
        "max_student_budget": 75.0,
        "min_contribution_margin": 0.25,
        "stock_buffer_safety_pct": 0.85
    }
    res = client.post("/api/recommendations", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert len(data["pricing_recommendations"]) == 8
    assert len(data["combo_recommendations"]) > 0

def test_llm_status():
    res = client.get("/api/llm/status")
    assert res.status_code == 200
    data = res.json()
    assert "configured" in data
    assert "model" in data
    assert data["model"] == "qwen-2.5-32b"

def test_chat():
    payload = {
        "message": "Why should we reduce the price of French fries?"
    }
    res = client.post("/api/chat", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert "reply" in data
    assert "evidence" in data
    assert data["intent"] == "price_recommendation"
    assert data["evidence"]["headline"] is not None

def test_chat_empty():
    res = client.post("/api/chat", json={"message": "   "})
    assert res.status_code == 400

if __name__ == "__main__":
    test_health()
    test_overview()
    test_products()
    test_combos_discovery()
    test_demand_model_metrics()
    test_pricing_evaluation()
    test_combo_evaluation()
    test_recommendations()
    test_llm_status()
    test_chat()
    test_chat_empty()
    print("All backend integration tests (including Groq Qwen chat & LLM status) passed successfully!")
