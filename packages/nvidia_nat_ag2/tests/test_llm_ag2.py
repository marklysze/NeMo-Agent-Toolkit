# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Test LLM for AG2."""

from unittest.mock import Mock
from unittest.mock import patch

import pytest
from pydantic import Field

from nat.builder.builder import Builder
from nat.data_models.llm import LLMBaseConfig
from nat.data_models.retry_mixin import RetryMixin
from nat.data_models.thinking_mixin import ThinkingMixin
from nat.llm.aws_bedrock_llm import AWSBedrockModelConfig
from nat.llm.azure_openai_llm import AzureOpenAIModelConfig
from nat.llm.litellm_llm import LiteLlmModelConfig
from nat.llm.nim_llm import NIMModelConfig
from nat.llm.openai_llm import OpenAIModelConfig
from nat.plugins.ag2.llm import _patch_ag2_client_based_on_config


class MockRetryConfig(LLMBaseConfig, RetryMixin):
    """Mock config with retry mixin."""

    num_retries: int = 3
    retry_on_status_codes: list[int | str] = Field(default_factory=lambda: [500, 502, 503])
    retry_on_errors: list[str] | None = Field(default_factory=lambda: ["timeout"])


class MockThinkingConfig(LLMBaseConfig, ThinkingMixin):
    """Mock config with thinking mixin."""

    model_name: str = "nvidia/nvidia-nemotron-test"


class MockRetryThinkingConfig(LLMBaseConfig, RetryMixin, ThinkingMixin):
    """Mock config with both retry and thinking mixins."""

    model_name: str = "nvidia/nvidia-nemotron-test"
    num_retries: int = 3
    retry_on_status_codes: list[int | str] = Field(default_factory=lambda: [500, 502, 503])
    retry_on_errors: list[str] | None = Field(default_factory=lambda: ["timeout"])


class TestPatchAG2Client:
    """Tests for _patch_ag2_client_based_on_config."""

    def test_no_mixin(self):
        """Test patching without any mixins."""
        client = Mock()
        config = Mock(spec=LLMBaseConfig)
        result = _patch_ag2_client_based_on_config(client, config)
        assert result == client

    @patch("nat.plugins.ag2.llm.patch_with_retry")
    def test_retry_mixin(self, mock_patch_retry):
        """Test patching with retry mixin."""
        client = Mock()
        config = MockRetryConfig()
        mock_patch_retry.return_value = client

        _patch_ag2_client_based_on_config(client, config)

        mock_patch_retry.assert_called_once_with(
            client,
            retries=3,
            retry_codes=[500, 502, 503],
            retry_on_messages=["timeout"],
        )

    @patch("nat.plugins.ag2.llm.patch_with_thinking")
    def test_thinking_mixin(self, mock_patch_thinking):
        """Test patching with thinking mixin enabled on a Nemotron model."""
        client = Mock()
        config = MockThinkingConfig(thinking=True)
        mock_patch_thinking.return_value = client

        _patch_ag2_client_based_on_config(client, config)

        mock_patch_thinking.assert_called_once()

    def test_thinking_mixin_disabled(self):
        """Test patching with thinking mixin disabled does nothing."""
        client = Mock()
        config = MockThinkingConfig(thinking=None)

        result = _patch_ag2_client_based_on_config(client, config)
        assert result == client

    @patch("nat.plugins.ag2.llm.patch_with_thinking")
    @patch("nat.plugins.ag2.llm.patch_with_retry")
    def test_both_mixins(self, mock_patch_retry, mock_patch_thinking):
        """Test patching with both retry and thinking mixins."""
        client = Mock()
        config = MockRetryThinkingConfig(thinking=True)
        mock_patch_retry.return_value = client
        mock_patch_thinking.return_value = client

        _patch_ag2_client_based_on_config(client, config)

        mock_patch_retry.assert_called_once()
        mock_patch_thinking.assert_called_once()


class TestOpenAIAG2:
    """Tests for openai_ag2 LLM wrapper."""

    @pytest.fixture(name="mock_builder")
    def fixture_mock_builder(self):
        """Create a mock builder."""
        return Mock(spec=Builder)

    @patch("autogen.LLMConfig")
    async def test_openai_basic(self, mock_llm_config_cls, mock_builder):
        """Test basic OpenAI configuration."""
        mock_config = Mock()
        mock_llm_config_cls.return_value = mock_config

        from nat.plugins.ag2.llm import openai_ag2

        llm_config = OpenAIModelConfig(model_name="gpt-4", api_key="test-key")

        async with openai_ag2(llm_config, mock_builder) as result:
            assert result is not None

        mock_llm_config_cls.assert_called_once()
        call_args = mock_llm_config_cls.call_args
        config_entry = call_args[0][0]
        assert config_entry["model"] == "gpt-4"
        assert config_entry["api_key"] == "test-key"

    @patch("autogen.LLMConfig")
    async def test_openai_with_base_url(self, mock_llm_config_cls, mock_builder):
        """Test OpenAI configuration with custom base URL."""
        mock_config = Mock()
        mock_llm_config_cls.return_value = mock_config

        from nat.plugins.ag2.llm import openai_ag2

        llm_config = OpenAIModelConfig(model_name="gpt-4", api_key="test-key", base_url="https://custom.api.com/v1")

        async with openai_ag2(llm_config, mock_builder):
            pass

        call_args = mock_llm_config_cls.call_args
        config_entry = call_args[0][0]
        assert config_entry["base_url"] == "https://custom.api.com/v1"

    @patch("autogen.LLMConfig")
    async def test_openai_with_params(self, mock_llm_config_cls, mock_builder):
        """Test OpenAI configuration with extra parameters."""
        mock_config = Mock()
        mock_llm_config_cls.return_value = mock_config

        from nat.plugins.ag2.llm import openai_ag2

        llm_config = OpenAIModelConfig(model_name="gpt-4", api_key="test-key", temperature=0.7, top_p=0.9)

        async with openai_ag2(llm_config, mock_builder):
            pass

        call_args = mock_llm_config_cls.call_args
        assert call_args[1]["temperature"] == 0.7
        assert call_args[1]["top_p"] == 0.9


class TestNIMAG2:
    """Tests for nim_ag2 LLM wrapper."""

    @pytest.fixture(name="mock_builder")
    def fixture_mock_builder(self):
        """Create a mock builder."""
        return Mock(spec=Builder)

    @patch("autogen.LLMConfig")
    async def test_nim_default_base_url(self, mock_llm_config_cls, mock_builder):
        """Test NIM uses default base URL when none provided."""
        mock_config = Mock()
        mock_llm_config_cls.return_value = mock_config

        from nat.plugins.ag2.llm import nim_ag2

        llm_config = NIMModelConfig(model_name="meta/llama-3.1-70b-instruct", api_key="test-key")

        async with nim_ag2(llm_config, mock_builder):
            pass

        call_args = mock_llm_config_cls.call_args
        config_entry = call_args[0][0]
        assert config_entry["base_url"] == "https://integrate.api.nvidia.com/v1"

    @patch("autogen.LLMConfig")
    async def test_nim_custom_base_url(self, mock_llm_config_cls, mock_builder):
        """Test NIM with custom base URL and dummy key."""
        mock_config = Mock()
        mock_llm_config_cls.return_value = mock_config

        from nat.plugins.ag2.llm import nim_ag2

        llm_config = NIMModelConfig(model_name="custom-model", base_url="http://localhost:8000/v1")

        async with nim_ag2(llm_config, mock_builder):
            pass

        call_args = mock_llm_config_cls.call_args
        config_entry = call_args[0][0]
        assert config_entry["base_url"] == "http://localhost:8000/v1"
        assert config_entry["api_key"] == "dummy-api-key"


class TestAzureOpenAIAG2:
    """Tests for azure_openai_ag2 LLM wrapper."""

    @pytest.fixture(name="mock_builder")
    def fixture_mock_builder(self):
        """Create a mock builder."""
        return Mock(spec=Builder)

    @patch("autogen.LLMConfig")
    async def test_azure_basic(self, mock_llm_config_cls, mock_builder):
        """Test basic Azure OpenAI configuration."""
        mock_config = Mock()
        mock_llm_config_cls.return_value = mock_config

        from nat.plugins.ag2.llm import azure_openai_ag2

        llm_config = AzureOpenAIModelConfig(
            azure_deployment="gpt-4-deployment",
            azure_endpoint="https://myresource.openai.azure.com",
            api_key="test-key",
            api_version="2024-02-01",
        )

        async with azure_openai_ag2(llm_config, mock_builder):
            pass

        call_args = mock_llm_config_cls.call_args
        config_entry = call_args[0][0]
        assert config_entry["model"] == "gpt-4-deployment"
        assert config_entry["api_type"] == "azure"
        assert config_entry["api_version"] == "2024-02-01"
        assert "openai/deployments/gpt-4-deployment" in config_entry["base_url"]


class TestLiteLlmAG2:
    """Tests for litellm_ag2 LLM wrapper."""

    @pytest.fixture(name="mock_builder")
    def fixture_mock_builder(self):
        """Create a mock builder."""
        return Mock(spec=Builder)

    @patch("autogen.LLMConfig")
    async def test_litellm_basic(self, mock_llm_config_cls, mock_builder):
        """Test basic LiteLLM configuration."""
        mock_config = Mock()
        mock_llm_config_cls.return_value = mock_config

        from nat.plugins.ag2.llm import litellm_ag2

        llm_config = LiteLlmModelConfig(model_name="claude-3-opus", api_key="test-key")

        async with litellm_ag2(llm_config, mock_builder):
            pass

        call_args = mock_llm_config_cls.call_args
        config_entry = call_args[0][0]
        assert config_entry["model"] == "claude-3-opus"
        assert config_entry["api_key"] == "test-key"


class TestBedrockAG2:
    """Tests for bedrock_ag2 LLM wrapper."""

    @pytest.fixture(name="mock_builder")
    def fixture_mock_builder(self):
        """Create a mock builder."""
        return Mock(spec=Builder)

    @patch("autogen.LLMConfig")
    async def test_bedrock_basic(self, mock_llm_config_cls, mock_builder):
        """Test basic AWS Bedrock configuration."""
        mock_config = Mock()
        mock_llm_config_cls.return_value = mock_config

        from nat.plugins.ag2.llm import bedrock_ag2

        llm_config = AWSBedrockModelConfig(
            model_name="anthropic.claude-3-sonnet-20240229-v1:0",
            region_name="us-east-1",
        )

        async with bedrock_ag2(llm_config, mock_builder):
            pass

        call_args = mock_llm_config_cls.call_args
        config_entry = call_args[0][0]
        assert config_entry["model"] == "anthropic.claude-3-sonnet-20240229-v1:0"
        assert config_entry["api_type"] == "bedrock"
        assert config_entry["aws_region"] == "us-east-1"

    @patch("autogen.LLMConfig")
    async def test_bedrock_none_region(self, mock_llm_config_cls, mock_builder):
        """Test Bedrock with None region uses AWS default."""
        mock_config = Mock()
        mock_llm_config_cls.return_value = mock_config

        from nat.plugins.ag2.llm import bedrock_ag2

        llm_config = AWSBedrockModelConfig(
            model_name="anthropic.claude-3-sonnet-20240229-v1:0",
            region_name=None,
        )

        async with bedrock_ag2(llm_config, mock_builder):
            pass

        call_args = mock_llm_config_cls.call_args
        config_entry = call_args[0][0]
        assert "aws_region" not in config_entry
