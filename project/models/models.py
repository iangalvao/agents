import json
import os


def load_models(provider: str = "") -> dict:
    """
    Load models from the models.json file in the current directory.
    This function is used to load model configurations for various LLMs.
    """

    models_file = "models.json"
    if not os.path.exists(models_file):
        raise FileNotFoundError(f"{models_file} not found in the current directory")

    with open(models_file, "r") as f:
        models = json.load(f)
    if provider:
        # Filter models by provider if specified
        models = models.get(provider, {})
    return models
