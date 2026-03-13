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
"""Test tool_wrapper.py file."""

import typing
from unittest.mock import AsyncMock
from unittest.mock import Mock

import pytest
from autogen.tools import Tool
from pydantic import BaseModel

from nat.builder.builder import Builder
from nat.builder.function import Function
from nat.plugins.ag2.tool_wrapper import _resolve_type
from nat.plugins.ag2.tool_wrapper import ag2_tool_wrapper


class MockInputSchema(BaseModel):
    """Mock input schema for tool wrapper."""

    param1: str
    param2: int
    param3: float = 3.14


class TestResolveType:
    """Test cases for _resolve_type function."""

    def test_resolve_union_type(self):
        """Test resolving Union types."""
        union_type = str | None
        result = _resolve_type(union_type)
        assert result is str

    def test_resolve_non_union_type(self):
        """Test resolving non-union types."""
        result = _resolve_type(int)
        assert result is int

    def test_resolve_complex_union(self):
        """Test resolving union with multiple non-None types."""
        union_type = str | int | None
        result = _resolve_type(union_type)
        result_args = typing.get_args(result)
        assert set(result_args) == {str, int}


class TestAG2ToolWrapper:
    """Test cases for ag2_tool_wrapper function."""

    @pytest.fixture(name="mock_function")
    def fixture_mock_function(self):
        """Create a mock NAT function."""
        mock_fn = Mock(spec=Function)
        mock_fn.description = "Test function description"
        mock_fn.input_schema = MockInputSchema
        mock_fn.has_streaming_output = False
        mock_fn.has_single_output = True
        mock_fn.acall_invoke = AsyncMock(return_value="test_result")
        mock_fn.acall_stream = AsyncMock()
        return mock_fn

    @pytest.fixture(name="mock_builder")
    def fixture_mock_builder(self):
        """Create a mock builder."""
        return Mock(spec=Builder)

    def test_ag2_tool_wrapper_basic(self, mock_function, mock_builder):
        """Test basic tool wrapper returns an AG2 Tool."""
        result = ag2_tool_wrapper("test_tool", mock_function, mock_builder)

        assert isinstance(result, Tool)
        assert result.name == "test_tool"
        assert result.description == "Test function description"

    def test_ag2_tool_wrapper_streaming(self, mock_function, mock_builder):
        """Test tool wrapper with streaming output."""
        mock_function.has_streaming_output = True
        mock_function.has_single_output = False

        result = ag2_tool_wrapper("test_tool", mock_function, mock_builder)

        assert isinstance(result, Tool)
        assert result.name == "test_tool"

    def test_ag2_tool_wrapper_no_description(self, mock_function, mock_builder):
        """Test tool wrapper with no description."""
        mock_function.description = None

        result = ag2_tool_wrapper("test_tool", mock_function, mock_builder)

        assert isinstance(result, Tool)
        assert result.description == "No description provided."

    async def test_callable_ainvoke(self, mock_function, mock_builder):
        """Test the async invoke callable."""
        ag2_tool_wrapper("test_tool", mock_function, mock_builder)

        # The tool's underlying function should delegate to fn.acall_invoke
        invoke_result = await mock_function.acall_invoke(param1="arg1", param2=42)
        assert invoke_result == "test_result"
        mock_function.acall_invoke.assert_called_once_with(param1="arg1", param2=42)

    async def test_callable_astream(self, mock_function, mock_builder):
        """Test the async stream callable."""
        mock_function.has_streaming_output = True
        mock_function.has_single_output = False

        async def mock_stream(**kwargs):
            yield "item1"
            yield "item2"

        mock_function.acall_stream = mock_stream

        ag2_tool_wrapper("test_tool", mock_function, mock_builder)

        items = []
        async for item in mock_function.acall_stream():
            items.append(item)

        assert items == ["item1", "item2"]
