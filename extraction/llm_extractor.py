import json
import os

from openai import OpenAI
from pydantic import BaseModel, field_validator

_client: OpenAI | None = None

_SYSTEM = """You are a food ingredient classification expert for the CPG industry.
Given a raw text about an ingredient, extract structured facts.
Respond ONLY with valid JSON matching the schema. No prose, no markdown."""

_SCHEMA = """{
  "functional_class": "string — one of: emulsifier, sweetener, preservative, colorant, flavor, thickener, antioxidant, acidulant, bulking-agent, nutrient, fat, protein, carbohydrate, mineral, vitamin, enzyme, stabilizer, humectant, solvent, other",
  "allergens": ["array of strings — only: milk, eggs, fish, shellfish, tree-nuts, peanuts, wheat, soybeans, sesame — empty if none"],
  "vegan": "boolean or null if unknown",
  "kosher": "boolean or null if unknown",
  "halal": "boolean or null if unknown",
  "e_number": "string like E471 or null",
  "description": "1-sentence description of what this ingredient does in a food product",
  "confidence": "float 0.0-1.0 — how confident you are given the source text"
}"""


class IngredientProfile(BaseModel):
    name: str
    functional_class: str = "other"
    allergens: list[str] = []
    vegan: bool | None = None
    kosher: bool | None = None
    halal: bool | None = None
    e_number: str | None = None
    description: str = ""
    raw_text: str = ""
    sources: list[str] = []
    confidence: float = 0.5

    @field_validator("functional_class")
    @classmethod
    def normalize_class(cls, v: str) -> str:
        valid = {
            "emulsifier", "sweetener", "preservative", "colorant", "flavor",
            "thickener", "antioxidant", "acidulant", "bulking-agent", "nutrient",
            "fat", "protein", "carbohydrate", "mineral", "vitamin", "enzyme",
            "stabilizer", "humectant", "solvent", "other",
        }
        return v if v in valid else "other"

    @field_validator("allergens")
    @classmethod
    def normalize_allergens(cls, v: list[str]) -> list[str]:
        valid = {"milk", "eggs", "fish", "shellfish", "tree-nuts", "peanuts", "wheat", "soybeans", "sesame"}
        return [a.lower() for a in v if a.lower() in valid]

    @field_validator("confidence")
    @classmethod
    def clamp_confidence(cls, v: float) -> float:
        return max(0.0, min(1.0, v))


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    return _client


def extract(name: str, raw_text: str, sources: list[str] | None = None) -> IngredientProfile:
    if raw_text.strip():
        prompt = f"""Ingredient name: {name}

Source text (may be partial or indirect):
{raw_text[:2500]}

Return JSON matching this schema:
{_SCHEMA}"""
        used_sources = sources or []
    else:
        prompt = f"""Ingredient name: {name}

No external source text is available. Use your knowledge about food ingredients, CPG manufacturing, and food chemistry.

Return JSON matching this schema:
{_SCHEMA}

Set confidence lower (0.4-0.7) since this is based on general knowledge, not verified source data."""
        used_sources = ["llm_knowledge"]

    try:
        resp = _get_client().chat.completions.create(
            model="o4-mini",
            messages=[
                {"role": "system", "content": _SYSTEM},
                {"role": "user", "content": prompt},
            ],
            response_format={"type": "json_object"},
        )
        text = resp.choices[0].message.content.strip()
        parsed = json.loads(text)
        return IngredientProfile(name=name, raw_text=raw_text, sources=used_sources, **parsed)
    except Exception:
        return IngredientProfile(name=name, raw_text=raw_text, sources=used_sources, confidence=0.1)
