# tests/unit/api/test_chat.py
# AI-Mind © 2025 by Martin Bukowski is licensed under CC BY-NC-SA 4.0

import pytest
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient

from aim_server.serverapi import ServerApi
from aim.llm.models import LanguageModelV2, ModelCategory


@pytest.fixture
def mock_chat_manager():
    """Create a mock ChatManager."""
    chat = MagicMock()
    chat.cvm = MagicMock()
    chat.current_document = None
    chat.current_workspace = None
    return chat


@pytest.fixture
def mock_model():
    """Create a mock LanguageModel."""
    model = MagicMock(spec=LanguageModelV2)
    model.name = "test-model"
    model.max_tokens = 4096
    model.max_output_tokens = 2048
    model.category = {ModelCategory.INSTRUCT}
    model.llm_factory = MagicMock()
    return model


@pytest.fixture
def mock_llm_provider():
    """Create a mock LLM provider."""
    provider = MagicMock()
    provider.stream_turns = MagicMock(return_value=iter(["Hello ", "from ", "AI"]))
    return provider


@pytest.fixture
def mock_persona():
    """Create a mock persona."""
    persona = MagicMock()
    persona.get_wakeup.return_value = "Wake up message"
    persona.xml_decorator = MagicMock(side_effect=lambda x, **kwargs: x)
    persona.default_location = "Home"
    return persona


@pytest.fixture
def sample_chat_request():
    """Sample chat completion request."""
    return {
        "model": "test-model",
        "messages": [
            {"role": "user", "content": "Hello"}
        ],
        "metadata": {
            "user_id": "user1",
            "persona_id": "Andi"
        },
        "temperature": 0.7,
        "stream": False
    }


class TestChatModule:
    """Tests for ChatModule endpoints."""

    def test_chat_completion_success(self, mock_chat_manager, mock_model, mock_llm_provider, mock_persona, sample_chat_request):
        """Test successful chat completion."""
        with patch('aim_server.serverapi.ChatManager.from_config_with_roster', return_value=mock_chat_manager):
            with patch('aim.llm.models.LanguageModelV2.index_models', return_value={"test-model": mock_model}):
                with patch('aim.chat.chat_strategy_for') as mock_strategy_factory:
                    mock_strategy = MagicMock()
                    mock_strategy.chat_turns_for = MagicMock(return_value=[
                        {"role": "user", "content": "Hello"}
                    ])
                    mock_strategy_factory.return_value = mock_strategy

                    mock_model.llm_factory.return_value = mock_llm_provider

                    with patch('aim.utils.turns.validate_turns'):
                        server = ServerApi()
                        server.shared_roster.personas['Andi'] = mock_persona
                        client = TestClient(server.app)

                        response = client.post(
                            "/v1/chat/completions",
                            json=sample_chat_request,
                            headers={"Authorization": "Bearer test_token"}
                        )

                        assert response.status_code == 200
                        data = response.json()
                        assert data["model"] == "test-model"
                        assert data["choices"][0]["message"]["role"] == "assistant"
                        assert data["choices"][0]["message"]["content"] == "Hello from AI"

    def test_chat_completion_with_stream(self, mock_chat_manager, mock_model, mock_llm_provider, mock_persona, sample_chat_request):
        """Test chat completion with streaming enabled."""
        sample_chat_request["stream"] = True

        with patch('aim_server.serverapi.ChatManager.from_config_with_roster', return_value=mock_chat_manager):
            with patch('aim.llm.models.LanguageModelV2.index_models', return_value={"test-model": mock_model}):
                with patch('aim.chat.chat_strategy_for') as mock_strategy_factory:
                    mock_strategy = MagicMock()
                    mock_strategy.chat_turns_for = MagicMock(return_value=[
                        {"role": "user", "content": "Hello"}
                    ])
                    mock_strategy_factory.return_value = mock_strategy

                    mock_model.llm_factory.return_value = mock_llm_provider

                    with patch('aim.utils.turns.validate_turns'):
                        server = ServerApi()
                        server.shared_roster.personas['Andi'] = mock_persona
                        client = TestClient(server.app)

                        response = client.post(
                            "/v1/chat/completions",
                            json=sample_chat_request,
                            headers={"Authorization": "Bearer test_token"}
                        )

                        assert response.status_code == 200
                        assert response.headers["content-type"] == "text/event-stream; charset=utf-8"

    def test_invalid_model(self, sample_chat_request):
        """Test that invalid model returns 400."""
        sample_chat_request["model"] = "invalid-model"

        with patch('aim_server.serverapi.ChatManager.from_config_with_roster'):
            with patch('aim.llm.models.LanguageModelV2.index_models', return_value={}):
                server = ServerApi()
                client = TestClient(server.app)

                response = client.post(
                    "/v1/chat/completions",
                    json=sample_chat_request,
                    headers={"Authorization": "Bearer test_token"}
                )

                assert response.status_code == 400
                assert "Invalid model" in response.json()["detail"]

    def test_invalid_persona_id(self, mock_model, sample_chat_request):
        """Test that invalid persona_id returns 400."""
        sample_chat_request["metadata"]["persona_id"] = "InvalidPersona"

        with patch('aim_server.serverapi.ChatManager.from_config_with_roster'):
            with patch('aim.llm.models.LanguageModelV2.index_models', return_value={"test-model": mock_model}):
                server = ServerApi()
                client = TestClient(server.app)

                response = client.post(
                    "/v1/chat/completions",
                    json=sample_chat_request,
                    headers={"Authorization": "Bearer test_token"}
                )

                assert response.status_code == 400
                assert "Invalid persona" in response.json()["detail"]

    def test_missing_metadata(self, mock_model):
        """Test that missing metadata returns 400."""
        request = {
            "model": "test-model",
            "messages": [{"role": "user", "content": "Hello"}],
            "metadata": None
        }

        with patch('aim_server.serverapi.ChatManager.from_config_with_roster'):
            with patch('aim.llm.models.LanguageModelV2.index_models', return_value={"test-model": mock_model}):
                server = ServerApi()
                client = TestClient(server.app)

                response = client.post(
                    "/v1/chat/completions",
                    json=request,
                    headers={"Authorization": "Bearer test_token"}
                )

                assert response.status_code == 400
                assert "No metadata provided" in response.json()["detail"]

    def test_empty_messages(self, mock_model, sample_chat_request):
        """Test that empty messages list returns 400."""
        sample_chat_request["messages"] = []

        with patch('aim_server.serverapi.ChatManager.from_config_with_roster'):
            with patch('aim.llm.models.LanguageModelV2.index_models', return_value={"test-model": mock_model}):
                server = ServerApi()
                client = TestClient(server.app)

                response = client.post(
                    "/v1/chat/completions",
                    json=sample_chat_request,
                    headers={"Authorization": "Bearer test_token"}
                )

                assert response.status_code == 400
                assert "No messages provided" in response.json()["detail"]

    def test_chat_with_active_document(self, mock_chat_manager, mock_model, mock_llm_provider, mock_persona, sample_chat_request):
        """Test chat completion with active document in metadata."""
        sample_chat_request["metadata"]["active_document"] = "doc123"

        with patch('aim_server.serverapi.ChatManager.from_config_with_roster', return_value=mock_chat_manager):
            with patch('aim.llm.models.LanguageModelV2.index_models', return_value={"test-model": mock_model}):
                with patch('aim.chat.chat_strategy_for') as mock_strategy_factory:
                    mock_strategy = MagicMock()
                    mock_strategy.chat_turns_for = MagicMock(return_value=[
                        {"role": "user", "content": "Hello"}
                    ])
                    mock_strategy_factory.return_value = mock_strategy

                    mock_model.llm_factory.return_value = mock_llm_provider

                    with patch('aim.utils.turns.validate_turns'):
                        server = ServerApi()
                        server.shared_roster.personas['Andi'] = mock_persona
                        client = TestClient(server.app)

                        response = client.post(
                            "/v1/chat/completions",
                            json=sample_chat_request,
                            headers={"Authorization": "Bearer test_token"}
                        )

                        assert response.status_code == 200
                        # Verify active_document was set on chat manager
                        assert mock_chat_manager.current_document == "doc123"

    def test_chat_with_pinned_messages(self, mock_chat_manager, mock_model, mock_llm_provider, mock_persona, sample_chat_request):
        """Test chat completion with pinned messages in metadata."""
        sample_chat_request["metadata"]["pinned_messages"] = ["msg1", "msg2"]

        # Create the mock strategy with proper setup
        mock_strategy = MagicMock()
        mock_strategy.chat_turns_for = MagicMock(return_value=[
            {"role": "user", "content": "Hello"}
        ])
        mock_strategy.clear_pinned = MagicMock()
        mock_strategy.pin_message = MagicMock()

        with patch('aim_server.serverapi.ChatManager.from_config_with_roster', return_value=mock_chat_manager):
            with patch('aim.llm.models.LanguageModelV2.index_models', return_value={"test-model": mock_model}):
                with patch('aim_server.modules.chat.route.chat_strategy_for', return_value=mock_strategy) as mock_strategy_factory:
                    mock_model.llm_factory.return_value = mock_llm_provider

                    with patch('aim.utils.turns.validate_turns'):
                        server = ServerApi()
                        server.shared_roster.personas['Andi'] = mock_persona
                        client = TestClient(server.app)

                        response = client.post(
                            "/v1/chat/completions",
                            json=sample_chat_request,
                            headers={"Authorization": "Bearer test_token"}
                        )

                        assert response.status_code == 200
                        # Verify pinned messages were set
                        mock_strategy.clear_pinned.assert_called_once()
                        assert mock_strategy.pin_message.call_count == 2

    def test_get_models(self):
        """Test get models endpoint."""
        # Create a real LanguageModelV2 instance for serialization
        from aim.llm.models import LanguageModelV2, ModelProvider, ModelCategory, SamplerConfig

        real_model = LanguageModelV2(
            name="test-model",
            provider=ModelProvider.OPENAI,
            architecture="transformer",
            size="medium",
            nsfw=False,
            category={ModelCategory.INSTRUCT},
            sampler=SamplerConfig(),
            max_tokens=4096,
            max_output_tokens=2048
        )

        models = {
            "test-model": real_model,
        }

        with patch('aim_server.serverapi.ChatManager.from_config_with_roster'):
            with patch('aim.llm.models.LanguageModelV2.index_models', return_value=models):
                server = ServerApi()
                client = TestClient(server.app)

                response = client.get("/v1/chat/models")

                assert response.status_code == 200
                data = response.json()
                assert "categories" in data
                assert "models" in data
                # Models are serialized, so check count
                assert len(data["models"]) >= 1

    def test_lazy_cache_different_personas(self, mock_model, mock_llm_provider, mock_persona):
        """Test that different personas get different ChatManager instances."""
        with patch('aim_server.serverapi.ChatManager') as MockChatManager:
            mock_chat1 = MagicMock()
            mock_chat1.current_document = None
            mock_chat1.current_workspace = None
            mock_chat2 = MagicMock()
            mock_chat2.current_document = None
            mock_chat2.current_workspace = None
            MockChatManager.from_config_with_roster.side_effect = [mock_chat1, mock_chat2]

            with patch('aim.llm.models.LanguageModelV2.index_models', return_value={"test-model": mock_model}):
                with patch('aim.chat.chat_strategy_for') as mock_strategy_factory:
                    mock_strategy = MagicMock()
                    mock_strategy.chat_turns_for = MagicMock(return_value=[
                        {"role": "user", "content": "Hello"}
                    ])
                    mock_strategy_factory.return_value = mock_strategy

                    mock_model.llm_factory.return_value = mock_llm_provider

                    with patch('aim.utils.turns.validate_turns'):
                        server = ServerApi()
                        server.shared_roster.personas['Andi'] = mock_persona
                        server.shared_roster.personas['Nova'] = mock_persona
                        client = TestClient(server.app)

                        # Request for Andi
                        request1 = {
                            "model": "test-model",
                            "messages": [{"role": "user", "content": "Hello"}],
                            "metadata": {"user_id": "user1", "persona_id": "Andi"},
                            "stream": False
                        }
                        response1 = client.post(
                            "/v1/chat/completions",
                            json=request1,
                            headers={"Authorization": "Bearer test_token"}
                        )
                        assert response1.status_code == 200

                        # Request for Nova
                        request2 = {
                            "model": "test-model",
                            "messages": [{"role": "user", "content": "Hello"}],
                            "metadata": {"user_id": "user1", "persona_id": "Nova"},
                            "stream": False
                        }
                        response2 = client.post(
                            "/v1/chat/completions",
                            json=request2,
                            headers={"Authorization": "Bearer test_token"}
                        )
                        assert response2.status_code == 200

                        # Should have created two different ChatManagers
                        assert MockChatManager.from_config_with_roster.call_count == 2
