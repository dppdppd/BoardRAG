from __future__ import annotations


def openai_requires_default_temperature(model_name: str) -> bool:
    """Return True if this OpenAI model doesn't accept custom temperature (must use default).

    Covers o4 reasoning and future reasoning-style identifiers.
    """
    m = (model_name or "").lower()
    # Models that disallow setting temperature explicitly
    if m.startswith("o4") or "-reasoning" in m or m.startswith("gpt-5"):
        return True
    # o3 family (e.g., o3-mini) does not accept temperature
    if m.startswith("o3"):
        return True
    return False


