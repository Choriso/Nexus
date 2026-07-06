"""
Обратная совместимость: все настройки перенесены в корневой config.py.
"""

from config import (  # noqa: F401
    Config,
    DevelopmentConfig,
    ProductionConfig,
    TestingConfig,
    config,
    config_by_name,
    get_config,
)

__all__ = [
    "Config",
    "DevelopmentConfig",
    "ProductionConfig",
    "TestingConfig",
    "config",
    "config_by_name",
    "get_config",
]
