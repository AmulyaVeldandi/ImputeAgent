"""Configuration file validation."""

import logging
from typing import Dict, Any, List

logger = logging.getLogger(__name__)


def validate_config(cfg: Dict[str, Any]) -> bool:
    """
    Validate the main configuration file.

    Args:
        cfg: Configuration dictionary loaded from YAML

    Returns:
        True if valid, raises ValueError otherwise
    """
    required_sections = ["data", "output", "missingness", "evaluation"]

    for section in required_sections:
        if section not in cfg:
            raise ValueError(f"Missing required config section: {section}")

    # Validate data section
    if "numeric" not in cfg["data"] or "categorical" not in cfg["data"]:
        raise ValueError("Config 'data' section must have 'numeric' and 'categorical' lists")

    if not isinstance(cfg["data"]["numeric"], list):
        raise ValueError("Config 'data.numeric' must be a list")

    if not isinstance(cfg["data"]["categorical"], list):
        raise ValueError("Config 'data.categorical' must be a list")

    # Validate output section
    if "dir" not in cfg["output"]:
        raise ValueError("Config 'output' section must have 'dir' field")

    # Validate missingness section
    if "types" not in cfg["missingness"] or "fractions" not in cfg["missingness"]:
        raise ValueError("Config 'missingness' section must have 'types' and 'fractions' lists")

    valid_miss_types = ["MCAR", "MAR", "MNAR"]
    for miss_type in cfg["missingness"]["types"]:
        if miss_type not in valid_miss_types:
            raise ValueError(
                f"Invalid missingness type '{miss_type}'. Must be one of: {valid_miss_types}"
            )

    for frac in cfg["missingness"]["fractions"]:
        if not isinstance(frac, (int, float)) or not (0.0 <= frac <= 1.0):
            raise ValueError(
                f"Invalid missingness fraction '{frac}'. Must be a number between 0.0 and 1.0"
            )

    # Validate evaluation section
    if "downstream_model" not in cfg["evaluation"]:
        raise ValueError("Config 'evaluation' section must have 'downstream_model' field")

    valid_models = ["logistic", "rf", "xgboost"]
    if cfg["evaluation"]["downstream_model"] not in valid_models:
        logger.warning(
            f"Downstream model '{cfg['evaluation']['downstream_model']}' not in recommended list: {valid_models}"
        )

    # Validate seed (optional)
    if "seed" in cfg:
        if not isinstance(cfg["seed"], int):
            raise ValueError("Config 'seed' must be an integer")

    logger.info("Configuration validation passed")
    return True


def validate_decider_config(dcfg: Dict[str, Any]) -> bool:
    """
    Validate the decider configuration file.

    Args:
        dcfg: Decider configuration dictionary loaded from YAML

    Returns:
        True if valid, raises ValueError otherwise
    """
    if "decider" not in dcfg:
        raise ValueError("Decider config must have 'decider' section")

    decider = dcfg["decider"]

    # Validate optional fields with type checking
    if "default_confidence" in decider:
        conf = decider["default_confidence"]
        if not isinstance(conf, (int, float)) or not (0.0 <= conf <= 1.0):
            raise ValueError(
                f"Invalid default_confidence '{conf}'. Must be a number between 0.0 and 1.0"
            )

    if "llm_probe_size" in decider:
        size = decider["llm_probe_size"]
        if not isinstance(size, int) or size <= 0:
            raise ValueError(f"Invalid llm_probe_size '{size}'. Must be a positive integer")

    logger.info("Decider configuration validation passed")
    return True
