"""
Automated unit & integration tests for Groq Qwen LLM Reasoning Service
and /api/chat conversational pricing assistant endpoint.

Mocks the Groq API so tests run reliably without requiring a live API key or network.
Tests:
1. Missing credentials handling
2. Empty and invalid requests
3. Intent recognition & deterministic evidence gathering
4. Successful Groq Qwen mock response
5. AuthenticationError, RateLimitError, APITimeoutError, and ConnectionError fallbacks
6. /api/llm/status and /api/chat endpoint contracts
"""

import os
import sys

# Ensure repository root is on sys.path so test runs cleanly without PYTHONPATH=.
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import unittest
from unittest.mock import MagicMock, patch
from fastapi import HTTPException

from backend.schemas import ChatRequest, ChatMessage, ChatResponse
from backend.llm_service import LLMService, GROQ_AVAILABLE
from backend.main import post_chat, get_llm_status, health_check


class TestLLMService(unittest.TestCase):

    def setUp(self):
        # Service instance without credentials
        self.service_no_key = LLMService(api_key="", model="qwen-2.5-32b")

    def test_missing_credentials_fallback(self):
        """When API key is missing, service must return deterministic grounded explanation."""
        self.assertFalse(self.service_no_key.is_configured())
        res = self.service_no_key.explain("Why should we reduce the price of French fries?")
        
        self.assertEqual(res["intent"], "price_recommendation")
        self.assertEqual(res["status"], "missing_credentials")
        self.assertFalse(res["is_llm_generated"])
        self.assertIsNotNone(res["evidence"])
        self.assertIn("French Fries", res["evidence"].headline)
        self.assertIn("deterministic fallback", res["reply"].lower())
        self.assertIn("P103", res["evidence"].metrics["product_id"])

    def test_empty_or_whitespace_message(self):
        """Empty messages should return invalid_request status."""
        res = self.service_no_key.explain("   ")
        self.assertEqual(res["status"], "invalid_request")
        self.assertEqual(res["intent"], "empty_query")

    def test_intent_price_simulation_delta(self):
        """'What happens if we reduce the sandwich price by ₹5?' should simulate ₹35 price."""
        res = self.service_no_key.explain("What happens if we reduce the sandwich price by ₹5?")
        self.assertEqual(res["intent"], "price_simulation")
        self.assertIsNotNone(res["evidence"])
        m = res["evidence"].metrics
        self.assertEqual(m["product_id"], "P101")
        self.assertEqual(m["current_price"], 40.0)
        self.assertEqual(m["candidate_price"], 35.0)
        self.assertIn("revenue_delta", m)
        self.assertIn("contribution_delta", m)

    def test_intent_combo_recommendation(self):
        """'Which combo provides the best contribution while remaining affordable?'"""
        res = self.service_no_key.explain("Which combo provides the best contribution while remaining affordable?")
        self.assertEqual(res["intent"], "combo_recommendation")
        self.assertIsNotNone(res["evidence"])
        self.assertIn("student_budget_limit", res["evidence"].metrics)
        self.assertGreater(len(res["evidence"].table), 0)

    def test_intent_contribution_vs_revenue(self):
        """'Explain contribution versus revenue using our canteen data.'"""
        res = self.service_no_key.explain("Explain contribution versus revenue using our canteen data.")
        self.assertEqual(res["intent"], "contribution_vs_revenue")
        self.assertIsNotNone(res["evidence"])
        m = res["evidence"].metrics
        self.assertIn("total_revenue", m)
        self.assertIn("total_variable_cost", m)
        self.assertIn("total_estimated_contribution", m)
        self.assertGreater(m["total_revenue"], m["total_estimated_contribution"])

    def test_intent_stock_bottlenecks(self):
        """'Which products have stock bottlenecks?'"""
        res = self.service_no_key.explain("Which products have stock bottlenecks?")
        self.assertEqual(res["intent"], "stock_bottlenecks")
        self.assertIsNotNone(res["evidence"])
        self.assertIn("lowest_stock_items", res["evidence"].metrics)
        self.assertGreater(len(res["evidence"].table), 0)

    def test_intent_compare_scenarios(self):
        """'Compare two pricing scenarios.'"""
        res = self.service_no_key.explain("Compare two pricing scenarios for burger.")
        self.assertEqual(res["intent"], "compare_scenarios")
        self.assertIsNotNone(res["evidence"])
        self.assertIn("current_price", res["evidence"].metrics)
        self.assertIn("optimal_price", res["evidence"].metrics)

    @patch("backend.llm_service.Groq")
    def test_mock_successful_groq_qwen_call(self, mock_groq_cls):
        """Verify successful call through official Groq client with Qwen model."""
        # Setup mock client
        mock_client = MagicMock()
        mock_groq_cls.return_value = mock_client
        
        mock_completion = MagicMock()
        mock_choice = MagicMock()
        mock_choice.message.content = (
            "### AI Pricing Analysis\n"
            "Based on the empirical demand model, French Fries has high price elasticity. "
            "Reducing price from ₹35 to ₹30 increases demand from 32.1 to 41.5 units, "
            "adding +₹18.4 in daily contribution before fixed canteen rent."
        )
        mock_completion.choices = [mock_choice]
        mock_client.chat.completions.create.return_value = mock_completion

        # Instantiate service with mocked client
        service = LLMService(api_key="gsk_mock_valid_key_12345", model="qwen-2.5-32b")
        service._client = mock_client

        res = service.explain("Why should we reduce the price of French fries?")

        self.assertEqual(res["status"], "success")
        self.assertTrue(res["is_llm_generated"])
        self.assertEqual(res["model_used"], "qwen-2.5-32b")
        self.assertIn("AI Pricing Analysis", res["reply"])
        self.assertEqual(res["intent"], "price_recommendation")
        self.assertIsNotNone(res["evidence"])

        # Check call arguments
        mock_client.chat.completions.create.assert_called_once()
        kwargs = mock_client.chat.completions.create.call_args[1]
        self.assertEqual(kwargs["model"], "qwen-2.5-32b")
        self.assertEqual(len(kwargs["messages"]), 2) # system + user
        self.assertEqual(kwargs["messages"][0]["role"], "system")
        self.assertIn("CanteenIQ Copilot", kwargs["messages"][0]["content"])

    def test_mock_groq_authentication_error(self):
        """AuthenticationError should trigger fallback without leaking keys or throwing 500."""
        from groq import AuthenticationError
        
        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = AuthenticationError(
            message="Invalid API Key",
            response=MagicMock(status_code=401),
            body={"error": {"message": "Invalid API Key"}}
        )

        service = LLMService(api_key="gsk_invalid_key", model="qwen-2.5-32b")
        service._client = mock_client

        res = service.explain("Why should we reduce the price of French fries?")
        self.assertEqual(res["status"], "authentication_error")
        self.assertFalse(res["is_llm_generated"])
        self.assertIn("deterministic fallback mode", res["reply"].lower())
        self.assertNotIn("gsk_invalid_key", res["reply"])

    def test_mock_groq_rate_limit_error(self):
        """RateLimitError should fall back gracefully with exact calculations."""
        from groq import RateLimitError
        
        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = RateLimitError(
            message="Rate limit reached: 30 requests per minute",
            response=MagicMock(status_code=429),
            body={"error": {"message": "Rate limit reached"}}
        )

        service = LLMService(api_key="gsk_valid_key", model="qwen-2.5-32b")
        service._client = mock_client

        res = service.explain("Which combo provides the best contribution while remaining affordable?")
        self.assertEqual(res["status"], "rate_limited")
        self.assertFalse(res["is_llm_generated"])
        self.assertIn("deterministic fallback mode", res["reply"].lower())
        self.assertIsNotNone(res["evidence"])

    def test_mock_groq_timeout_error(self):
        """APITimeoutError should fall back gracefully."""
        from groq import APITimeoutError
        
        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = APITimeoutError(request=MagicMock())

        service = LLMService(api_key="gsk_valid_key", model="qwen-2.5-32b")
        service._client = mock_client

        res = service.explain("What happens if we reduce the sandwich price by ₹5?")
        self.assertEqual(res["status"], "timeout")
        self.assertFalse(res["is_llm_generated"])
        self.assertIn("deterministic fallback mode", res["reply"].lower())

    def test_mock_generic_api_failure(self):
        """Generic unexpected exception should not crash the service."""
        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = RuntimeError("Network socket disconnect")

        service = LLMService(api_key="gsk_valid_key", model="qwen-2.5-32b")
        service._client = mock_client

        res = service.explain("Explain contribution versus revenue using our canteen data.")
        self.assertEqual(res["status"], "api_error")
        self.assertFalse(res["is_llm_generated"])
        self.assertIn("deterministic fallback mode", res["reply"].lower())


class TestChatEndpoints(unittest.TestCase):

    def test_health_check_includes_llm_status(self):
        data = health_check()
        self.assertIn("llm_service_configured", data)
        self.assertIn("llm_model", data)
        self.assertEqual(data["llm_model"], "qwen-2.5-32b")

    def test_get_llm_status_endpoint(self):
        status = get_llm_status()
        self.assertIn("configured", status)
        self.assertIn("model", status)
        self.assertIn("temperature", status)
        self.assertIn("timeout_seconds", status)

    def test_post_chat_valid_request(self):
        req = ChatRequest(
            message="Why should we reduce the price of French fries?",
            history=[ChatMessage(role="user", content="Hello")]
        )
        res = post_chat(req)
        self.assertIsInstance(res, dict)
        self.assertIn("reply", res)
        self.assertIn("evidence", res)
        self.assertEqual(res["intent"], "price_recommendation")

    def test_post_chat_empty_message_raises_400(self):
        req = ChatRequest(message="   ")
        with self.assertRaises(HTTPException) as ctx:
            post_chat(req)
        self.assertEqual(ctx.exception.status_code, 400)


if __name__ == "__main__":
    unittest.main()
