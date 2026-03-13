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
"""AG2 ConversableAgent workflow registration for NeMo Agent Toolkit.

Registers an AG2 ConversableAgent as a NAT workflow function, enabling it to
be configured and composed via YAML workflow definitions.
"""

import logging
import re
from collections.abc import AsyncGenerator

from pydantic import Field

from nat.builder.builder import Builder
from nat.builder.framework_enum import LLMFrameworkEnum
from nat.builder.function_info import FunctionInfo
from nat.cli.register_workflow import register_function
from nat.data_models.agent import AgentBaseConfig
from nat.data_models.api_server import ChatRequest
from nat.data_models.api_server import ChatRequestOrMessage
from nat.data_models.component_ref import FunctionGroupRef
from nat.data_models.component_ref import FunctionRef
from nat.utils.type_converter import GlobalTypeConverter

logger = logging.getLogger(__name__)

_AG2_LOG_PREFIX = "[AG2 Agent]"


class AG2ConversableAgentConfig(AgentBaseConfig, name="ag2_conversable_agent"):
    """Configuration for an AG2 ConversableAgent workflow.

    This agent wraps AG2's ``ConversableAgent`` as a NAT workflow function,
    providing tool-calling capabilities via AG2's native agent loop.
    """

    description: str = Field(
        default="AG2 Conversable Agent Workflow", description="Description of this function's use.")
    tool_names: list[FunctionRef | FunctionGroupRef] = Field(
        default_factory=list, description="The list of tools to provide to the agent.")
    max_turns: int = Field(default=10, description="Maximum number of conversation turns before stopping.")
    system_prompt: str | None = Field(default=None, description="System prompt for the agent.")
    human_input_mode: str = Field(
        default="NEVER",
        description="Human input mode: 'NEVER', 'ALWAYS', or 'TERMINATE'.")
    is_termination_msg: str | None = Field(
        default=None,
        description="If set, the agent terminates when the response contains this string.")


@register_function(config_type=AG2ConversableAgentConfig, framework_wrappers=[LLMFrameworkEnum.AG2])
async def ag2_conversable_agent_workflow(config: AG2ConversableAgentConfig, builder: Builder):
    """Build and register an AG2 ConversableAgent as a NAT workflow.

    Creates an ``AssistantAgent`` (the LLM-powered agent) and a ``UserProxyAgent``
    (the tool executor) and orchestrates them via ``a_initiate_chat``.

    Args:
        config: Agent configuration from the workflow definition.
        builder: NAT Builder instance for resolving LLM and tool references.

    Yields:
        FunctionInfo with single and streaming response functions.
    """
    from autogen import AssistantAgent
    from autogen import ConversableAgent
    from autogen.agentchat.chat import ChatResult

    # Suppress AG2's event output (e.g. "TERMINATING RUN") when not in verbose mode.
    # When verbose=True, leave the logger alone so users get the full AG2 conversation trace.
    if not config.verbose:
        logging.getLogger("ag2.event.processor").setLevel(logging.ERROR)

    llm_config = await builder.get_llm(config.llm_name, wrapper_type=LLMFrameworkEnum.AG2)
    tools = await builder.get_tools(tool_names=config.tool_names, wrapper_type=LLMFrameworkEnum.AG2)

    # Termination: by default, the assistant appends "TERMINATE" when done
    # and the user_proxy detects it to end the conversation.
    _TERMINATE_KEYWORD = "TERMINATE"

    if config.is_termination_msg:

        def _is_termination(msg: dict) -> bool:
            content = msg.get("content", "") or ""
            return config.is_termination_msg in content
    else:

        def _is_termination(msg: dict) -> bool:
            content = msg.get("content", "") or ""
            return content.rstrip().endswith(_TERMINATE_KEYWORD)

    # Append TERMINATE instruction to system prompt unless user provided
    # their own termination string.
    system_prompt = config.system_prompt or "You are a helpful AI assistant."
    if not config.is_termination_msg:
        system_prompt += (
            f"\n\nWhen the task is fully complete, provide your final answer "
            f"followed by the word {_TERMINATE_KEYWORD} on the same line. "
            f"For example: 'The answer is 42. {_TERMINATE_KEYWORD}'"
        )

    # Build the assistant agent (LLM-powered)
    assistant = AssistantAgent(
        name="assistant",
        llm_config=llm_config,
        system_message=system_prompt,
    )

    # Build the user proxy (orchestrator / tool executor).
    # is_termination_msg is set on user_proxy so it stops when the
    # assistant's final answer arrives.
    user_proxy = ConversableAgent(
        name="user_proxy",
        human_input_mode=config.human_input_mode,
        is_termination_msg=_is_termination,
        llm_config=False,
    )

    # Register tools: LLM schema on assistant, execution on user_proxy.
    # In AG2's two-agent pattern, the assistant recommends tool calls and
    # the user_proxy executes them.
    for tool in tools:
        tool.register_for_llm(assistant)
        tool.register_for_execution(user_proxy)

    async def _response_fn(chat_request_or_message: ChatRequestOrMessage) -> str:
        """Invoke the AG2 agent and return the final response.

        Args:
            chat_request_or_message: Input message or chat request.

        Returns:
            The agent's final response as a string.
        """
        try:
            message = GlobalTypeConverter.get().convert(chat_request_or_message, to_type=ChatRequest)
            user_message = _extract_last_user_message(message)

            logger.debug("%s Starting chat with message: %s", _AG2_LOG_PREFIX, user_message[:100])

            result: ChatResult = await user_proxy.a_initiate_chat(
                recipient=assistant,
                message=user_message,
                max_turns=config.max_turns,
                silent=not config.verbose,
            )

            return _extract_final_response(result)
        except Exception as ex:
            logger.error("%s Agent failed with exception: %s", _AG2_LOG_PREFIX, ex)
            raise

    async def _stream_fn(chat_request_or_message: ChatRequestOrMessage) -> AsyncGenerator[str]:
        """Invoke the AG2 agent and stream the final response.

        AG2's ConversableAgent does not natively support token-level streaming.
        This yields the complete response as a single chunk.

        Args:
            chat_request_or_message: Input message or chat request.

        Yields:
            The agent's final response.
        """
        result = await _response_fn(chat_request_or_message)
        yield result

    yield FunctionInfo.create(single_fn=_response_fn, stream_fn=_stream_fn, description=config.description)


def _extract_last_user_message(chat_request: ChatRequest) -> str:
    """Extract the last user message from a ChatRequest.

    Args:
        chat_request: The chat request containing messages.

    Returns:
        The content of the last user message, or the last message if no user message found.
    """
    if chat_request.messages:
        # Find last user message
        for msg in reversed(chat_request.messages):
            if msg.role == "user":
                return msg.content
        # Fallback to last message
        return chat_request.messages[-1].content
    return ""


def _extract_final_response(result) -> str:
    """Extract the final response from a ChatResult.

    Args:
        result: AG2 ChatResult from a_initiate_chat.

    Returns:
        The summary or last assistant message from the chat.
    """
    response = ""

    if result.summary:
        response = result.summary
    else:
        # Walk chat history backwards to find last assistant response
        for msg in reversed(result.chat_history):
            role = msg.get("role", "")
            content = msg.get("content", "")
            if role == "assistant" and content:
                response = content
                break

    if not response:
        # Fallback: last message content
        if result.chat_history:
            response = str(result.chat_history[-1].get("content", ""))

    # Strip <think>...</think> blocks (chain-of-thought reasoning from some models)
    response = re.sub(r"<think>.*?</think>\s*", "", response, flags=re.DOTALL)

    # Strip the TERMINATE keyword from the response
    cleaned = response.rstrip().removesuffix("TERMINATE").rstrip()

    # If stripping TERMINATE left nothing, search deeper for a substantive response
    if not cleaned and result.chat_history:
        for msg in reversed(result.chat_history):
            content = msg.get("content", "") or ""
            # Skip empty messages, tool results, and bare TERMINATE
            stripped = content.rstrip().removesuffix("TERMINATE").rstrip()
            if stripped and msg.get("role") == "assistant":
                return stripped

    return cleaned
