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
"""Test AG2 callback handler."""

from unittest.mock import MagicMock
from unittest.mock import Mock
from unittest.mock import patch

import pytest

from nat.plugins.ag2.callback_handler import AG2ProfilerHandler


class TestAG2ProfilerHandler:
    """Test cases for AG2ProfilerHandler."""

    @pytest.fixture(name="mock_context")
    def fixture_mock_context(self):
        """Create a mock context with step manager."""
        mock_step_manager = MagicMock()
        mock_ctx = MagicMock()
        mock_ctx.intermediate_step_manager = mock_step_manager
        return mock_ctx

    @pytest.fixture(name="handler")
    def fixture_handler(self, mock_context):
        """Create an AG2ProfilerHandler with mocked context."""
        with patch("nat.plugins.ag2.callback_handler.Context") as mock_context_cls:
            mock_context_cls.get.return_value = mock_context
            handler = AG2ProfilerHandler()
        return handler

    def test_init(self, handler):
        """Test handler initialization."""
        assert handler._instrumented is False
        assert handler._original_create is None
        assert handler._original_execute_function is None
        assert handler._original_a_execute_function is None

    @patch("nat.plugins.ag2.callback_handler.Context")
    def test_instrument_patches_create(self, mock_context_cls):
        """Test that instrument patches OpenAIWrapper.create."""
        mock_ctx = MagicMock()
        mock_context_cls.get.return_value = mock_ctx

        handler = AG2ProfilerHandler()

        with patch.dict("sys.modules", {
            "autogen": MagicMock(),
            "autogen.oai": MagicMock(),
            "autogen.oai.client": MagicMock(),
            "autogen.agentchat": MagicMock(),
            "autogen.agentchat.conversable_agent": MagicMock(),
        }):
            handler.instrument()

        assert handler._instrumented is True

    @patch("nat.plugins.ag2.callback_handler.Context")
    def test_instrument_idempotent(self, mock_context_cls):
        """Test that calling instrument twice is safe."""
        mock_ctx = MagicMock()
        mock_context_cls.get.return_value = mock_ctx

        handler = AG2ProfilerHandler()
        handler._instrumented = True

        handler.instrument()
        assert handler._original_create is None

    def test_extract_model_name(self, handler):
        """Test model name extraction from OpenAIWrapper."""
        mock_wrapper = Mock()
        mock_wrapper._config_list = [{"model": "gpt-4"}]

        result = handler._extract_model_name(mock_wrapper)
        assert result == "gpt-4"

    def test_extract_model_name_empty(self, handler):
        """Test model name extraction with empty config."""
        mock_wrapper = Mock()
        mock_wrapper._config_list = []

        result = handler._extract_model_name(mock_wrapper)
        assert result == "unknown_model"

    def test_extract_model_name_no_attr(self, handler):
        """Test model name extraction with no _config_list."""
        mock_wrapper = Mock(spec=[])

        result = handler._extract_model_name(mock_wrapper)
        assert result == "unknown_model"

    def test_extract_input_text_dict_messages(self, handler):
        """Test extracting input text from dict-style messages."""
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there"},
        ]

        result = handler._extract_input_text(messages)
        assert "Hello" in result
        assert "Hi there" in result

    def test_extract_input_text_object_messages(self, handler):
        """Test extracting input text from object-style messages."""
        msg = Mock()
        msg.content = "Test message"

        result = handler._extract_input_text([msg])
        assert "Test message" in result

    def test_extract_input_text_list_content(self, handler):
        """Test extracting input text with list content."""
        messages = [{"role": "user", "content": [{"text": "part1"}, {"text": "part2"}]}]

        result = handler._extract_input_text(messages)
        assert "part1" in result
        assert "part2" in result

    def test_extract_output_text(self, handler):
        """Test extracting output text from response."""
        mock_response = Mock()
        mock_choice = Mock()
        mock_message = Mock()
        mock_message.content = "Response text"
        mock_choice.message = mock_message
        mock_response.choices = [mock_choice]

        result = handler._extract_output_text(mock_response)
        assert result == "Response text"

    def test_extract_output_text_no_content(self, handler):
        """Test extracting output text when no content."""
        mock_response = Mock()
        mock_response.choices = []

        result = handler._extract_output_text(mock_response)
        assert result == ""

    def test_extract_usage(self, handler):
        """Test extracting usage from response."""
        mock_response = Mock()
        mock_response.usage.prompt_tokens = 10
        mock_response.usage.completion_tokens = 20
        mock_response.usage.total_tokens = 30

        result = handler._extract_usage(mock_response)
        assert result == {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30}

    def test_extract_usage_none(self, handler):
        """Test extracting usage when None."""
        mock_response = Mock()
        mock_response.usage = None

        result = handler._extract_usage(mock_response)
        assert result == {}

    def test_llm_wrapper_success(self, handler):
        """Test LLM wrapper emits START and END events on success."""
        original_func = Mock(return_value=Mock(
            choices=[Mock(message=Mock(content="output"))],
            usage=Mock(prompt_tokens=5, completion_tokens=10, total_tokens=15),
        ))

        wrapped = handler._create_llm_wrapper(original_func)

        mock_wrapper = Mock()
        mock_wrapper._config_list = [{"model": "test-model"}]

        result = wrapped(mock_wrapper, messages=[{"role": "user", "content": "hello"}])

        assert result is not None
        original_func.assert_called_once()
        assert handler.step_manager.push_intermediate_step.call_count == 2

        calls = handler.step_manager.push_intermediate_step.call_args_list
        assert calls[0][0][0].event_type.value == "LLM_START"
        assert calls[1][0][0].event_type.value == "LLM_END"

    def test_llm_wrapper_error(self, handler):
        """Test LLM wrapper emits START and error END on failure."""
        original_func = Mock(side_effect=RuntimeError("API error"))

        wrapped = handler._create_llm_wrapper(original_func)

        mock_wrapper = Mock()
        mock_wrapper._config_list = [{"model": "test-model"}]

        with pytest.raises(RuntimeError, match="API error"):
            wrapped(mock_wrapper, messages=[])

        assert handler.step_manager.push_intermediate_step.call_count == 2
        calls = handler.step_manager.push_intermediate_step.call_args_list
        assert calls[1][0][0].metadata.error == "API error"

    def test_tool_wrapper_success(self, handler):
        """Test tool wrapper emits START and END events on success."""
        original_func = Mock(return_value=(True, {"name": "my_tool", "role": "function", "content": "result"}))

        wrapped = handler._create_tool_wrapper(original_func)

        func_call = {"name": "my_tool", "arguments": '{"x": 1}'}
        result = wrapped(Mock(), func_call, call_id="123")

        assert result == (True, {"name": "my_tool", "role": "function", "content": "result"})
        assert handler.step_manager.push_intermediate_step.call_count == 2

        calls = handler.step_manager.push_intermediate_step.call_args_list
        assert calls[0][0][0].event_type.value == "TOOL_START"
        assert calls[0][0][0].name == "my_tool"
        assert calls[1][0][0].event_type.value == "TOOL_END"

    def test_tool_wrapper_error(self, handler):
        """Test tool wrapper emits START and error END on failure."""
        original_func = Mock(side_effect=ValueError("tool failed"))

        wrapped = handler._create_tool_wrapper(original_func)

        func_call = {"name": "bad_tool", "arguments": "{}"}
        with pytest.raises(ValueError, match="tool failed"):
            wrapped(Mock(), func_call)

        assert handler.step_manager.push_intermediate_step.call_count == 2
        calls = handler.step_manager.push_intermediate_step.call_args_list
        assert calls[1][0][0].metadata.error == "tool failed"

    @pytest.mark.asyncio
    async def test_async_tool_wrapper_success(self, handler):
        """Test async tool wrapper emits START and END events on success."""

        async def original_func(agent_self, func_call, call_id=None, verbose=False):
            return (True, {"name": "async_tool", "role": "function", "content": "async_result"})

        wrapped = handler._create_async_tool_wrapper(original_func)

        func_call = {"name": "async_tool", "arguments": '{"y": 2}'}
        result = await wrapped(Mock(), func_call, call_id="456")

        assert result == (True, {"name": "async_tool", "role": "function", "content": "async_result"})
        assert handler.step_manager.push_intermediate_step.call_count == 2

        calls = handler.step_manager.push_intermediate_step.call_args_list
        assert calls[0][0][0].event_type.value == "TOOL_START"
        assert calls[1][0][0].event_type.value == "TOOL_END"

    @pytest.mark.asyncio
    async def test_async_tool_wrapper_error(self, handler):
        """Test async tool wrapper emits START and error END on failure."""

        async def original_func(agent_self, func_call, call_id=None, verbose=False):
            raise RuntimeError("async tool failed")

        wrapped = handler._create_async_tool_wrapper(original_func)

        func_call = {"name": "failing_tool", "arguments": "{}"}
        with pytest.raises(RuntimeError, match="async tool failed"):
            await wrapped(Mock(), func_call)

        assert handler.step_manager.push_intermediate_step.call_count == 2

    def test_uninstrument(self, handler):
        """Test uninstrument resets state."""
        handler._instrumented = True
        handler._original_create = Mock()
        handler._original_execute_function = None
        handler._original_a_execute_function = None

        with patch.dict("sys.modules", {
            "autogen": MagicMock(),
            "autogen.oai": MagicMock(),
            "autogen.oai.client": MagicMock(),
        }):
            handler.uninstrument()

        assert handler._instrumented is False
        assert handler._original_create is None
