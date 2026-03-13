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
"""Test register.py file."""


class TestRegisterModule:
    """Test cases for register module."""

    def test_llm_module_functions(self):
        """Test that LLM module has expected functions."""
        from nat.plugins.ag2 import llm

        expected_functions = ['openai_ag2', 'azure_openai_ag2', 'nim_ag2', 'litellm_ag2', 'bedrock_ag2']

        for func_name in expected_functions:
            assert hasattr(llm, func_name), f"Function {func_name} not found in llm module"

    def test_tool_wrapper_module_functions(self):
        """Test that tool_wrapper module has expected functions."""
        from nat.plugins.ag2 import tool_wrapper

        expected_functions = ['ag2_tool_wrapper']

        for func_name in expected_functions:
            assert hasattr(tool_wrapper, func_name), f"Function {func_name} not found in tool_wrapper module"

    def test_callback_handler_module(self):
        """Test that callback_handler module has expected class."""
        from nat.plugins.ag2 import callback_handler

        assert hasattr(callback_handler, 'AG2ProfilerHandler')

    def test_agent_module_functions(self):
        """Test that agent module has expected functions."""
        from nat.plugins.ag2.agent import register

        assert hasattr(register, 'ag2_conversable_agent_workflow')
        assert hasattr(register, 'AG2ConversableAgentConfig')
