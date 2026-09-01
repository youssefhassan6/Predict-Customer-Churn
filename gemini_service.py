"""
gemini_service.py
-----------------
Encapsulates all Google Gemini API interactions using the official google-genai SDK.
The ML model owns the churn prediction; Gemini only generates
business/retention recommendations based on ML output.
"""

import os
import json
import re
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

_client = None  # lazily initialised


def _get_client():
    """Return a cached Gemini client, or None if unavailable."""
    global _client
    if _client is not None:
        return _client

    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    if not api_key or api_key == "YOUR_API_KEY_HERE":
        return None

    try:
        from google import genai
        _client = genai.Client(api_key=api_key)
    except Exception:
        _client = None

    return _client


def is_gemini_available() -> bool:
    """Return True if a valid API key is configured and SDK is importable."""
    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    if not api_key or api_key == "YOUR_API_KEY_HERE":
        return False
    try:
        from google import genai  # noqa: F401
        return True
    except ImportError:
        return False


# Fields that may contain PII – strip before sending to Gemini
_PII_FIELDS = {
    "customerid", "email", "phone", "address", "name",
    "firstname", "lastname", "first_name", "last_name",
    "customer name", "customername", "churn"
}


def _strip_pii(customer_dict: dict) -> dict:
    """Remove known PII keys before sending data to an external API."""
    return {
        k: v for k, v in customer_dict.items()
        if k.lower().strip() not in _PII_FIELDS
    }


# ── Prompt ────────────────────────────────────────────────────────────────────

_SYSTEM_CONTEXT = (
    "You are a customer retention specialist at a telecommunications company. "
    "An ML model has already predicted the churn probability for a customer. "
    "Your role is ONLY to generate actionable business recommendations based on "
    "that prediction and the customer's profile. "
    "Do NOT re-predict churn. Do NOT contradict the ML model output. "
    "Do NOT invent facts about the customer. "
    "Keep recommendations practical, concise, and written for a retention team."
)

_RESPONSE_SCHEMA = """\
Return your answer as a single valid JSON object with exactly these keys:
{
  "recommended_action": "<one-sentence action for the retention team>",
  "reason": "<brief explanation grounded in the customer profile and ML result>",
  "priority": "<Low | Medium | High>",
  "retention_strategy": "<specific retention approach>",
  "suggested_next_step": "<concrete next step within 48 hours>"
}
Do not include any text outside the JSON object."""


def _build_prompt(customer_attrs: dict, churn_prob: float,
                  risk_level: str, prediction: str) -> str:
    safe_attrs = _strip_pii(customer_attrs)
    attrs_text = "\n".join(f"  - {k}: {v}" for k, v in safe_attrs.items())
    return (
        f"{_SYSTEM_CONTEXT}\n\n"
        f"Customer Profile:\n{attrs_text}\n\n"
        f"ML Model Output:\n"
        f"  - Churn Probability: {churn_prob:.1%}\n"
        f"  - Risk Level: {risk_level}\n"
        f"  - Prediction: {prediction}\n\n"
        f"{_RESPONSE_SCHEMA}"
    )


# ── Parsing ───────────────────────────────────────────────────────────────────

_REQUIRED_KEYS = {
    "recommended_action", "reason", "priority",
    "retention_strategy", "suggested_next_step"
}

_VALID_PRIORITIES = {"low", "medium", "high"}


def _parse_gemini_json(raw_text: str) -> dict | None:
    """
    Extract and validate the JSON object from a Gemini response.
    Returns None if parsing or validation fails.
    """
    text = re.sub(r"```(?:json)?", "", raw_text, flags=re.IGNORECASE).strip()
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return None
    try:
        data = json.loads(match.group())
    except json.JSONDecodeError:
        return None

    if not _REQUIRED_KEYS.issubset(data.keys()):
        return None
    if data.get("priority", "").strip().lower() not in _VALID_PRIORITIES:
        data["priority"] = "Medium"

    return data


# ── Public API ────────────────────────────────────────────────────────────────

_FALLBACK = {
    "recommended_action": "Manual review recommended",
    "reason": "AI recommendation unavailable. Review customer profile and ML output directly.",
    "priority": "Medium",
    "retention_strategy": "Apply standard retention protocol based on risk level.",
    "suggested_next_step": "Contact the customer via the next scheduled touchpoint.",
}


def get_recommendation(
    customer_attrs: dict,
    churn_prob: float,
    risk_level: str,
    prediction: str,
) -> dict:
    """
    Generate a business recommendation for a single customer.

    Parameters
    ----------
    customer_attrs : dict   – raw customer feature dict (PII stripped internally)
    churn_prob     : float  – churn probability from ML model (0–1)
    risk_level     : str    – "Low Risk" | "Medium Risk" | "High Risk"
    prediction     : str    – "Likely to Churn" | "Not Likely to Churn"

    Returns
    -------
    dict with keys: recommended_action, reason, priority,
                    retention_strategy, suggested_next_step
    """
    client = _get_client()
    if client is None:
        return _FALLBACK.copy()

    prompt = _build_prompt(customer_attrs, churn_prob, risk_level, prediction)
    try:
        from google.genai import types
        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=prompt,
        )
        raw = response.text
        parsed = _parse_gemini_json(raw)
        if parsed is None:
            return _FALLBACK.copy()
        return parsed
    except Exception:
        return _FALLBACK.copy()


def get_batch_recommendations(customers: list) -> list:
    """
    Generate recommendations for a list of customer dicts.
    Each dict must contain: customer_attrs, churn_prob, risk_level, prediction.
    Falls back gracefully per-customer if the API call fails.
    """
    results = []
    for c in customers:
        rec = get_recommendation(
            customer_attrs=c.get("customer_attrs", {}),
            churn_prob=c.get("churn_prob", 0.0),
            risk_level=c.get("risk_level", "Unknown"),
            prediction=c.get("prediction", "Unknown"),
        )
        results.append(rec)
    return results
