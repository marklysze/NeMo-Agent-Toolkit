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
"""Test AG2 ConversableAgent workflow."""

from unittest.mock import AsyncMock
from unittest.mock import Mock
from unittest.mock import patch

import pytest

from nat.plugins.ag2.agent.register import AG2ConversableAgentConfig
from nat.plugins.ag2.agent.register import _extract_final_response
from nat.plugins.ag2.agent.register import _extract_last_user_message


class TestAG2ConversableAgentConfig:
    """Test cases for AG2ConversableAgentConfig."""

    def test_default_config(self):
        """Test default configuration values."""
        config = AG2ConversableAgentConfig(llm_name="test_llm")
        assert config.description == "AG2 Conversable Agent Workflow"
        assert config.tool_names == []
        assert config.max_turns == 10
        assert config.system_prompt is None
        assert config.human_input_mode == "NEVER"
        assert config.is_termination_msg is None

    def test_custom_config(self):
        """Test custom configuration values."""
        config = AG2ConversableAgentConfig(
            llm_name="gpt4_llm",
            description="Custom agent",
            max_turns=5,
            system_prompt="You are a coding assistant.",
            human_input_mode="TERMINATE",
            is_termination_msg="DONE",
        )
        assert config.description == "Custom agent"
        assert config.max_turns == 5
        assert config.system_prompt == "You are a coding assistant."
        assert config.human_input_mode == "TERMINATE"
        assert config.is_termination_msg == "DONE"


class TestExtractLastUserMessage:
    """Test cases for _extract_last_user_message."""

    def test_extracts_last_user_message(self):
        """Test extracting the last user message from a ChatRequest."""
        mock_request = Mock()
        msg1 = Mock()
        msg1.role = "user"
        msg1.content = "First message"
        msg2 = Mock()
        msg2.role = "assistant"
        msg2.content = "Response"
        msg3 = Mock()
        msg3.role = "user"
        msg3.content = "Second message"
        mock_request.messages = [msg1, msg2, msg3]

        result = _extract_last_user_message(mock_request)
        assert result == "Second message"

    def test_fallback_to_last_message(self):
        """Test fallback when no user message found."""
        mock_request = Mock()
        msg = Mock()
        msg.role = "system"
        msg.content = "System prompt"
        mock_request.messages = [msg]

        result = _extract_last_user_message(mock_request)
        assert result == "System prompt"

    def test_empty_messages(self):
        """Test with empty messages list."""
        mock_request = Mock()
        mock_request.messages = []

        result = _extract_last_user_message(mock_request)
        assert result == ""


class TestExtractFinalResponse:
    """Test cases for _extract_final_response."""

    def test_uses_summary_when_available(self):
        """Test that summary is preferred."""
        mock_result = Mock()
        mock_result.summary = "This is the summary"
        mock_result.chat_history = []

        result = _extract_final_response(mock_result)
        assert result == "This is the summary"

    def test_strips_terminate_keyword(self):
        """Test that TERMINATE keyword is stripped from response."""
        mock_result = Mock()
        mock_result.summary = ""
        mock_result.chat_history = [
            {"role": "assistant", "content": "The answer is 42. TERMINATE"},
        ]

        result = _extract_final_response(mock_result)
        assert result == "The answer is 42."

    def test_falls_back_to_last_assistant_message(self):
        """Test fallback to last assistant message when no summary."""
        mock_result = Mock()
        mock_result.summary = ""
        mock_result.chat_history = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
            {"role": "user", "content": "Thanks"},
            {"role": "assistant", "content": "You're welcome!"},
        ]

        result = _extract_final_response(mock_result)
        assert result == "You're welcome!"

    def test_falls_back_to_last_message(self):
        """Test fallback to last message when no assistant message."""
        mock_result = Mock()
        mock_result.summary = ""
        mock_result.chat_history = [
            {"role": "user", "content": "Hello"},
        ]

        result = _extract_final_response(mock_result)
        assert result == "Hello"

    def test_empty_history(self):
        """Test with empty chat history and no summary."""
        mock_result = Mock()
        mock_result.summary = ""
        mock_result.chat_history = []

        result = _extract_final_response(mock_result)
        assert result == ""

    def test_strips_think_tags(self):
        """Test that <think>...</think> blocks are stripped from response."""
        mock_result = Mock()
        mock_result.summary = ""
        mock_result.chat_history = [
            {"role": "assistant", "content": "<think>\nLet me reason about this.\n</think>\n\nThe answer is 42. TERMINATE"},
        ]

        result = _extract_final_response(mock_result)
        assert result == "The answer is 42."


class TestAG2ConversableAgentWorkflow:
    """Test cases for the ag2_conversable_agent_workflow function."""

    @pytest.fixture(name="mock_builder")
    def fixture_mock_builder(self):
        """Create a mock builder."""
        mock = Mock()
        mock.get_llm = AsyncMock(return_value=Mock())
        mock.get_tools = AsyncMock(return_value=[])
        return mock

    @patch("nat.plugins.ag2.agent.register.GlobalTypeConverter")
    @patch("autogen.ConversableAgent")
    @patch("autogen.AssistantAgent")
    async def test_workflow_creates_agents(self, mock_assistant_cls, mock_conversable_cls, mock_converter,
                                           mock_builder):
        """Test that the workflow creates the correct AG2 agents."""
        from nat.plugins.ag2.agent.register import ag2_conversable_agent_workflow

        config = AG2ConversableAgentConfig(
            llm_name="test_llm",
            system_prompt="Be helpful.",
        )

        async with ag2_conversable_agent_workflow(config, mock_builder) as function_info:
            assert function_info is not None
            assert function_info.single_fn is not None
            assert function_info.stream_fn is not None

        mock_builder.get_llm.assert_called_once()
        mock_assistant_cls.assert_called_once()

        # Verify system message was passed (with TERMINATE instruction appended)
        call_kwargs = mock_assistant_cls.call_args[1]
        assert "Be helpful." in call_kwargs["system_message"]

    @patch("nat.plugins.ag2.agent.register.GlobalTypeConverter")
    @patch("autogen.ConversableAgent")
    @patch("autogen.AssistantAgent")
    async def test_workflow_registers_tools(self, mock_assistant_cls, mock_conversable_cls, mock_converter,
                                            mock_builder):
        """Test that tools are registered on the assistant."""
        from nat.plugins.ag2.agent.register import ag2_conversable_agent_workflow

        mock_tool1 = Mock()
        mock_tool2 = Mock()
        mock_builder.get_tools = AsyncMock(return_value=[mock_tool1, mock_tool2])

        config = AG2ConversableAgentConfig(
            llm_name="test_llm",
            tool_names=["tool1", "tool2"],
        )

        async with ag2_conversable_agent_workflow(config, mock_builder) as function_info:
            assert function_info is not None

        # Each tool should have register_for_llm and register_for_execution called
        mock_tool1.register_for_llm.assert_called_once()
        mock_tool1.register_for_execution.assert_called_once()
        mock_tool2.register_for_llm.assert_called_once()
        mock_tool2.register_for_execution.assert_called_once()

    @patch("nat.plugins.ag2.agent.register.GlobalTypeConverter")
    @patch("autogen.ConversableAgent")
    @patch("autogen.AssistantAgent")
    async def test_workflow_with_termination_msg(self, mock_assistant_cls, mock_conversable_cls, mock_converter,
                                                 mock_builder):
        """Test that is_termination_msg configures termination on user_proxy."""
        from nat.plugins.ag2.agent.register import ag2_conversable_agent_workflow

        config = AG2ConversableAgentConfig(
            llm_name="test_llm",
            is_termination_msg="DONE",
        )

        async with ag2_conversable_agent_workflow(config, mock_builder) as function_info:
            assert function_info is not None

        # The user_proxy should have been created with a termination function
        call_kwargs = mock_conversable_cls.call_args[1]
        term_fn = call_kwargs["is_termination_msg"]
        assert term_fn is not None
        assert term_fn({"content": "Task is DONE"}) is True
        assert term_fn({"content": "Continue working"}) is False

    @patch("nat.plugins.ag2.agent.register.GlobalTypeConverter")
    @patch("autogen.ConversableAgent")
    @patch("autogen.AssistantAgent")
    async def test_workflow_default_system_prompt(self, mock_assistant_cls, mock_conversable_cls, mock_converter,
                                                  mock_builder):
        """Test that default system prompt includes TERMINATE instruction."""
        from nat.plugins.ag2.agent.register import ag2_conversable_agent_workflow

        config = AG2ConversableAgentConfig(llm_name="test_llm")

        async with ag2_conversable_agent_workflow(config, mock_builder) as function_info:
            assert function_info is not None

        call_kwargs = mock_assistant_cls.call_args[1]
        assert "You are a helpful AI assistant." in call_kwargs["system_message"]
        assert "TERMINATE" in call_kwargs["system_message"]

    @patch("nat.plugins.ag2.agent.register.GlobalTypeConverter")
    @patch("autogen.ConversableAgent")
    @patch("autogen.AssistantAgent")
    async def test_workflow_default_termination(self, mock_assistant_cls, mock_conversable_cls, mock_converter,
                                                mock_builder):
        """Test that default termination detects TERMINATE keyword."""
        from nat.plugins.ag2.agent.register import ag2_conversable_agent_workflow

        config = AG2ConversableAgentConfig(llm_name="test_llm")

        async with ag2_conversable_agent_workflow(config, mock_builder) as function_info:
            assert function_info is not None

        # user_proxy should detect TERMINATE
        call_kwargs = mock_conversable_cls.call_args[1]
        term_fn = call_kwargs["is_termination_msg"]
        assert term_fn({"content": "Here is the answer. TERMINATE"}) is True
        assert term_fn({"content": "Let me use a tool"}) is False
