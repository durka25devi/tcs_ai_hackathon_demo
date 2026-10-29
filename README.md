# AI-Powered Pricing and Combo Offer Support System

Essential working prototype of an **AI-Powered Pricing and Combo Offer Support System** designed for a college canteen, integrated with **Groq-hosted Qwen LLM** for intelligent reasoning and natural-language explanations.

The system helps a canteen manager:
1. **Discover food combinations** from empirical order baskets using FP-Growth and Apriori association rule mining (`mlxtend`).
2. **Estimate product demand** dynamically responsive to candidate selling prices, day-of-week, and promotions using Scikit-Learn (`RandomForestRegressor`).
3. **Simulate pricing scenarios** calculating revenue, variable costs (ingredients + packaging), and contribution before fixed costs.
4. **Evaluate candidate combo offers** assessing customer savings, combined margins, student affordability limits, and stock bottlenecks.
5. **Generate transparent recommendations** to optimize contribution while enforcing budget and operational constraints.
6. **Provide AI Reasoning via Groq Qwen** (`backend/llm_service.py`) explaining the financial and operational "why" behind pricing and combos grounded strictly in verified calculations.

---

## 1. Project Architecture

```
TCS_demo/
├── backend/
│   ├── main.py                  # FastAPI server with REST endpoints & static frontend mount
│   ├── llm_service.py           # Groq Qwen LLM integration with grounded evidence retrieval
│   ├── data_loader.py           # Loads CSVs and calculates catalog metrics
│   ├── combo_discovery.py       # Association rule mining (Apriori & FP-Growth)
│   ├── demand_model.py          # Random Forest demand estimation & chronological split
│   ├── pricing_engine.py        # Single product candidate price scenario evaluation
│   ├── combo_evaluator.py       # Combo offer economics, savings & bottleneck checks
│   ├── recommendation_engine.py # Constrained optimization & recommendation generation
│   ├── schemas.py               # Pydantic request/response validation schemas
│   ├── test_api.py              # Automated integration tests for all API endpoints
│   ├── test_chat_llm.py         # Unit & mock tests for Groq Qwen service and /api/chat
│   ├── requirements.txt         # Backend Python dependencies
│   ├── .env.example             # Template for Groq API credentials
│   └── .env                     # Local environment variables (git-ignored)
├── data/
│   ├── products.csv             # 8 products with cost structure, prices & stock
│   ├── transactions.csv         # 8,372 transaction lines across 4,531 order baskets (90 days)
│   └── daily_sales.csv          # 720 product-day sales, prices, promotions & inventory
├── frontend/
│   └── static/
│       └── index.html           # Interactive React 18 + Tailwind CSS + Chart.js dashboard & AI Copilot
├── scripts/
│   ├── generate_data.py         # Reproducible synthetic data generation script (seed=42)
│   └── validate_data.py         # Dataset validation & integrity check script
├── .gitignore                   # Ignores .env, *.env, node_modules, __pycache__
└── README.md
```

---

## 2. LLM Reasoning Integration (Groq Qwen)

### Architecture Data Flow:
```
User Question
     │
     ▼
FastAPI (/api/chat)
     │
     ▼
Intent Identification & Entity Extraction
     │
     ▼
Existing Deterministic Python Engines
(Random Forest Demand Model, Pricing Engine, Combo Evaluator, Catalog)
     │
     ▼
Structured Verified Evidence
     │
     ▼
Groq Qwen Reasoning (qwen-2.5-32b)
     │
     ▼
Grounded Explanation + Numerical Evidence Card
     │
     ▼
React Dashboard (AI Copilot Tab)
```

### Safety & Grounding Principles:
- **Zero Numerical Hallucination**: Qwen is strictly instructed to explain computed facts from the evidence and never fabricate prices, costs, margins, or demand figures.
- **Clear Distinction**: The interface clearly distinguishes model-generated reasoning (`✨ Groq Qwen 2.5-32B AI Reasoning`) from deterministic calculations (`📊 Verified Engine Evidence`).
- **High-Reliability Fallback**: If `GROQ_API_KEY` is not configured or an API error/rate-limit occurs, the system continues running smoothly in deterministic fallback mode without crashing.
- **Prompt Injection Defense**: All user messages are treated strictly as untrusted input.

---

## 3. Dataset Schema & Assumptions

All datasets are **synthetic** college canteen records created for prototype development.

### A. `products.csv`
- `product_id`: Unique identifier (`P101` to `P108`).
- `product_name`: Food / beverage item (Veg Sandwich, Veg Burger, French Fries, Lemon Juice, Cold Coffee, Samosa, Paneer Roll, Water Bottle).
- `category`: Menu grouping (`Snacks`, `Beverages`, `Sides`).
- `current_price`: Baseline selling price in Indian Rupees (₹).
- `ingredient_cost`: Raw food input cost in ₹.
- `packaging_cost`: Disposables / packaging cost in ₹.
- `portion_size_g`: Weight / volume (grams or ml).
- `available_stock`: Daily standard prepared batch capacity (units).
- `waste_quantity`: Daily average unsold/discarded units under standard operations.

### B. `transactions.csv`
- `transaction_id`: Basket identifier (`TX10001` ...).
- `transaction_date`: Date of purchase (90 days from August 2026 to October 2026).
- `product_id` & `product_name`: Item purchased.
- `quantity`: Quantity in line item.
- `unit_price`: Price charged per unit.
- `discount`: Line-level promotional discount (₹) non-negative and `<= subtotal`.

### C. `daily_sales.csv`
- `date`: Daily observation date.
- `product_id`: Product identifier.
- `selling_price`: Tested price on that day.
- `quantity_sold`: Total units sold.
- `day_of_week`: Day of week (Monday through Sunday).
- `promotion_active`: Binary indicator (`1` = promotion/event day, `0` = normal).
- `stock_available`: Daily available inventory.
- `unsold_quantity`: `stock_available - quantity_sold`.

---

## 4. Financial & Operational Formulas

- **Unit Variable Cost**:
  $$\text{Unit Variable Cost} = \text{Ingredient Cost} + \text{Packaging Cost}$$
- **Revenue**:
  $$\text{Revenue} = \text{Candidate Price} \times \text{Estimated Quantity}$$
- **Contribution**:
  $$\text{Contribution} = \text{Revenue} - (\text{Unit Variable Cost} \times \text{Estimated Quantity})$$
  *(Note: Labeled strictly as contribution before fixed canteen rent, electricity, and staff overheads).*
- **Combo Savings**:
  $$\text{Customer Savings} = \sum(\text{Individual Current Prices}) - \text{Proposed Combo Price}$$
- **Combo Unit Contribution**:
  $$\text{Contribution per Combo} = \text{Proposed Combo Price} - \sum(\text{Component Variable Costs})$$
- **Stock Bottleneck**:
  $$\text{Max Combo Capacity} = \min_{i \in \text{Combo}}(\text{Available Stock}_i)$$

---

## 5. Quickstart & How to Run

### Step 1: Install Dependencies
```bash
pip install -r backend/requirements.txt
```

### Step 2: Configure Groq API Key
Copy `backend/.env.example` to `backend/.env` and add your Groq API key:
```bash
cp backend/.env.example backend/.env
```
Edit `backend/.env`:
```env
GROQ_API_KEY=gsk_your_actual_groq_api_key_here
GROQ_MODEL=qwen-2.5-32b
GROQ_TEMPERATURE=0.2
GROQ_MAX_TOKENS=1024
GROQ_TIMEOUT_SECONDS=20
```
*(If no API key is set, the application operates seamlessly in deterministic fallback mode).*

### Step 3: Run Automated Tests
```bash
# Unit & Mock tests for LLM Reasoning Service and Chat Endpoint
PYTHONPATH=. python3 backend/test_chat_llm.py

# End-to-end API Integration tests
PYTHONPATH=. python3 backend/test_api.py
```

### Step 4: Launch Application Server
```bash
PYTHONPATH=. python3 -m uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
```

Open your browser to:
- **Interactive UI Dashboard & AI Copilot**: [http://localhost:8000/](http://localhost:8000/)
- **Interactive API Documentation (Swagger)**: [http://localhost:8000/docs](http://localhost:8000/docs)

---

## 6. API Endpoints Summary

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/api/health` | Healthcheck, model status, and Groq Qwen configuration |
| `GET` | `/api/llm/status` | Current LLM model, configuration, and readiness status |
| `POST` | `/api/chat` | Conversational pricing assistant with Qwen reasoning & structured evidence |
| `GET` | `/api/overview` | Total transactions, revenue, contribution, top & slow products |
| `GET` | `/api/products` | Complete product catalog with variable costs and stock |
| `GET` | `/api/combos` | Discovered association rules with default parameters |
| `POST` | `/api/combos/discover` | Discovers combinations with custom support, confidence, lift |
| `GET` | `/api/demand-model/metrics` | Model MAE, baseline comparison, and feature importance |
| `POST` | `/api/pricing/evaluate` | Evaluates price candidates for revenue & contribution deltas |
| `POST` | `/api/combos/evaluate` | Evaluates combo pricing, customer savings, and constraints |
| `POST` | `/api/recommendations` | Suggests optimal prices and combos under budget & stock limits |

---

## 7. Sample Questions Supported by AI Copilot

- *"Why should we reduce the price of French fries?"*
- *"Which combo provides the best contribution while remaining affordable?"*
- *"What happens if we reduce the sandwich price by ₹5?"*
- *"Explain contribution versus revenue using our canteen data."*
- *"Which products have stock bottlenecks?"*
- *"Compare two pricing scenarios."*
