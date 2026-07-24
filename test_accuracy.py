"""
test_accuracy.py — Graph RAG Model Evaluation & Benchmark Test Suite
Run this script to test model routing accuracy, ground truth answers, and retrieval quality.

Usage:
    python test_accuracy.py
"""
import requests
import json

BASE_URL = "http://127.0.0.1:5000"

# Standard Test Suite with Ground Truth Answers & Expected Retriever Routes
TEST_SUITE = [
    {
        "id": 1,
        "question": "What plan is Acme Corp on?",
        "expected_route": "vector",
        "expected_facts": ["enterprise", "plan"],
        "category": "Fact Lookup"
    },
    {
        "id": 2,
        "question": "Who leads the Search team?",
        "expected_route": "vector",
        "expected_facts": ["Priya Nair"],
        "category": "Fact Lookup"
    },
    {
        "id": 3,
        "question": "Which customer is affected by the outage on billing-service?",
        "expected_route": "local",
        "expected_facts": ["Acme Corp"],
        "category": "Multi-hop Graph Chain"
    },
    {
        "id": 4,
        "question": "Which database does the service led by Priya Nair run on?",
        "expected_route": "local",
        "expected_facts": ["index-db"],
        "category": "Multi-hop Graph Chain"
    },
    {
        "id": 5,
        "question": "What is the most common root cause of outages across all services?",
        "expected_route": "global",
        "expected_facts": ["auth-service", "dependency"],
        "category": "Global Dataset Aggregate"
    },
    {
        "id": 6,
        "question": "What platform service creates a shared vulnerability between Payments and Search?",
        "expected_route": "global",
        "expected_facts": ["auth-service", "shared-db"],
        "category": "Global Dataset Aggregate"
    }
]

def evaluate_model():
    print("=" * 70)
    print("🚀 GRAPH RAG MODEL ACCURACY & BENCHMARK TEST SUITE")
    print("=" * 70)
    
    passed_routes = 0
    passed_answers = 0
    total = len(TEST_SUITE)
    
    for item in TEST_SUITE:
        q = item["question"]
        exp_route = item["expected_route"]
        exp_facts = item["expected_facts"]
        
        try:
            res = requests.post(f"{BASE_URL}/api/query/routed", json={"question": q}, timeout=30).json()
            pred_route = res.get("retriever_used", "unknown").lower()
            ans = res.get("answer", "")
            reasoning = res.get("classification", {}).get("reasoning", "")
            
            # Check route match
            route_ok = (pred_route == exp_route) or (pred_route in ("local", "global") and exp_route in ("local", "global"))
            if route_ok:
                passed_routes += 1
                
            # Check fact accuracy in answer
            facts_ok = any(fact.lower() in ans.lower() for fact in exp_facts)
            if facts_ok:
                passed_answers += 1
                
            status_symbol = "✅ PASS" if (route_ok and facts_ok) else "⚠️ MISROUTE / PARTIAL"
            
            print(f"\n[Test {item['id']}] Category: {item['category']}")
            print(f"❓ Question      : {q}")
            print(f"📌 Expected Route: {exp_route.upper()} | Predicted Route: {pred_route.upper()}")
            print(f"💡 LLM Reasoning : {reasoning}")
            print(f"💬 Answer        : {ans[:140]}...")
            print(f"📊 Test Result   : {status_symbol}")
            
        except Exception as exc:
            print(f"\n[Test {item['id']}] FAILED to execute: {exc}")
            
    print("\n" + "=" * 70)
    print("📈 FINAL BENCHMARK SCORE CARD")
    print("=" * 70)
    print(f"🎯 Router Classification Accuracy : {passed_routes}/{total} ({passed_routes/total*100:.1f}%)")
    print(f"🧠 Answer Fact Accuracy          : {passed_answers}/{total} ({passed_answers/total*100:.1f}%)")
    print("=" * 70)

if __name__ == "__main__":
    evaluate_model()
