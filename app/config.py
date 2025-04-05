import yaml
import os
import logging
from typing import Dict, Any, Optional

# Get a logger for this module
logger = logging.getLogger(__name__)

# Default configuration values (used if config.yaml is missing or incomplete)
DEFAULT_CONFIG = {
    "features": {
        "llm_agent": True,
        "ocr": {"enabled": True},
        "database": {"enabled": True},
        "scheduling": {"enabled": True},
        "email": {"enabled": True},
        "conversation_analysis": {"enabled": True},
        "progressive_memory": {"enabled": False},
    },
    "llm": {
        "primary_provider": "anthropic",
        "anthropic_model": "claude-3-haiku-20240307",
        "openai_model": "gpt-4o-mini",
        "temperature": 0.1,
    },
    "logging": {
        "level": "INFO",
    },
}

def load_config(config_path: str = "config.yaml") -> Dict[str, Any]:
    """Loads configuration from a YAML file, merging with defaults."""
    config = DEFAULT_CONFIG.copy() # Start with defaults

    if os.path.exists(config_path):
        try:
            with open(config_path, 'r') as f:
                user_config = yaml.safe_load(f)
            if user_config:
                # Deep merge user config into defaults (simple merge for now)
                # A more robust solution might use a recursive merge function
                for section, settings in user_config.items():
                    if section in config and isinstance(config[section], dict) and isinstance(settings, dict):
                        config[section].update(settings)
                    else:
                        config[section] = settings
            logger.info(f"Loaded configuration from {config_path}")
        except yaml.YAMLError as e:
            logger.warning(f"Error parsing {config_path}: {e}. Using default configuration.", exc_info=True)
        except Exception as e:
            logger.warning(f"Could not load {config_path}: {e}. Using default configuration.", exc_info=True)
    else:
        logger.warning(f"{config_path} not found. Using default configuration.")

    return config

# Load configuration globally on import
# This makes it accessible via `from app.config import settings`
settings = load_config()

# Helper functions to easily check feature flags
def is_feature_enabled(feature_key: str) -> bool:
    """Checks if a specific feature is enabled in the config."""
    # Handles nested keys like 'ocr.enabled'
    keys = feature_key.split('.')
    value = settings.get("features", {})
    try:
        for key in keys:
            value = value[key]
        return bool(value)
    except (KeyError, TypeError):
        # Fallback to default if key path is invalid or missing
        default_value = DEFAULT_CONFIG.get("features", {})
        try:
            for key in keys:
                default_value = default_value[key]
            return bool(default_value)
        except (KeyError, TypeError):
             return False # Feature is unknown or structure differs significantly

def get_llm_config() -> Dict[str, Any]:
    """Returns the LLM configuration section."""
    return settings.get("llm", DEFAULT_CONFIG["llm"])

def get_logging_config() -> Dict[str, Any]:
    """Returns the logging configuration section."""
    return settings.get("logging", DEFAULT_CONFIG["logging"])


DEFAULT_COMPANY_CONFIG = {
    "default": {
        "display_name": "Our Company",
        "company_info": "A leading provider of innovative solutions.",
        "products_info": "We offer a range of cutting-edge products.",
        "required_fields": ["contact_name", "email", "interest_level"],
        "optional_fields": ["company_name", "role"],
        "conversation_goal": "Gauge initial interest and collect basic contact information."
    }
}

def load_company_configs(config_path: str = "company_configs.yaml") -> Dict[str, Any]:
    """Loads company-specific configurations from a YAML file."""
    if os.path.exists(config_path):
        try:
            with open(config_path, 'r') as f:
                company_configs = yaml.safe_load(f)
            if company_configs and isinstance(company_configs, dict):
                # Ensure 'default' exists if other configs are present
                if 'default' not in company_configs:
                     logger.warning(f"'default' configuration missing in {config_path}. Adding a basic default.")
                     company_configs['default'] = DEFAULT_COMPANY_CONFIG['default']
                logger.info(f"Loaded company configurations from {config_path}")
                return company_configs
            else:
                 logger.warning(f"{config_path} is empty or not a valid dictionary. Using default company config.")
                 return DEFAULT_COMPANY_CONFIG.copy()
        except yaml.YAMLError as e:
            logger.warning(f"Error parsing {config_path}: {e}. Using default company configuration.", exc_info=True)
            return DEFAULT_COMPANY_CONFIG.copy()
        except Exception as e:
            logger.warning(f"Could not load {config_path}: {e}. Using default company configuration.", exc_info=True)
            return DEFAULT_COMPANY_CONFIG.copy()
    else:
        logger.warning(f"{config_path} not found. Using default company configuration.")
        return DEFAULT_COMPANY_CONFIG.copy()


if __name__ == "__main__":
    # Example of accessing settings - Configure basic logging for direct script execution
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger.info("--- Running config.py directly for testing ---")
    logger.info("--- Current Configuration ---")
    import json
    logger.info(json.dumps(settings, indent=2))

    logger.info("\n--- Feature Checks ---")
    logger.info(f"OCR Enabled: {is_feature_enabled('ocr.enabled')}")
    logger.info(f"Database Enabled: {is_feature_enabled('database.enabled')}")
    logger.info(f"Email Enabled: {is_feature_enabled('email.enabled')}")
    logger.info(f"Progressive Memory Enabled: {is_feature_enabled('progressive_memory.enabled')}")
    logger.info(f"Unknown Feature Enabled: {is_feature_enabled('unknown.feature')}") # Example of non-existent key

    logger.info("\n--- LLM Config ---")
    logger.info(get_llm_config())
