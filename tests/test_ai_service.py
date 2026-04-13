from src.services.ai_service import (
    BaseAIService,
    DisabledAIService,
    ExternalAIService,
    LocalAIService,
    get_ai_service,
)


def test_disabled_service_interface() -> None:
    service: BaseAIService = DisabledAIService()
    reply = service.generate_reply("ping")
    assert "AI отключен" in reply


def test_factory_returns_base_interface() -> None:
    service = get_ai_service()
    assert isinstance(service, BaseAIService)

