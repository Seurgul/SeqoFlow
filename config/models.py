"""LLM endpoint profiles from models.json (gitignored); models.example.json is the template."""
import json

from config.config import HOME

MODELS_FILE = HOME / "models.json"


def model_args(name: str) -> dict:
    """VLLMWrapper keyword arguments (model_name, url, api_key, timeout) of profile `name`."""
    if not MODELS_FILE.exists():
        raise FileNotFoundError(f"{MODELS_FILE} not found: copy models.example.json to models.json "
                                "and fill in the model names, endpoints and API keys")
    profiles = json.loads(MODELS_FILE.read_text())
    if name not in profiles:
        raise KeyError(f"model profile '{name}' not in {MODELS_FILE}; available: {sorted(profiles)}")
    return dict(profiles[name])
