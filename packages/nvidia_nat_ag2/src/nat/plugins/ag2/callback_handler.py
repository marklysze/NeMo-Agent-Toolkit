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
"""AG2 callback handler for usage statistics collection.

This module provides profiling instrumentation for AG2 agents by monkey-patching
LLM client and tool execution methods to collect telemetry data.

Supported Targets
-----------------
- ``OpenAIWrapper.create``: All LLM completions (OpenAI, NIM, Azure, Bedrock, etc.)
- ``ConversableAgent.execute_function``: Synchronous tool executions
- ``ConversableAgent.a_execute_function``: Asynchronous tool executions
"""

import copy
import logging
import threading
import time
from collections.abc import Callable
from typing import Any

from nat.builder.context import Context
from nat.builder.framework_enum import LLMFrameworkEnum
from nat.data_models.intermediate_step import IntermediateStepPayload
from nat.data_models.intermediate_step import IntermediateStepType
from nat.data_models.intermediate_step import StreamEventData
from nat.data_models.intermediate_step import TraceMetadata
from nat.data_models.intermediate_step import UsageInfo
from nat.data_models.profiler_callback import BaseProfilerCallback
from nat.data_models.token_usage import TokenUsageBaseModel

logger = logging.getLogger(__name__)

_FRAMEWORK = LLMFrameworkEnum.AG2


class AG2ProfilerHandler(BaseProfilerCallback):
    """Callback handler for AG2 that intercepts LLM and tool calls for profiling.

    This handler monkey-patches AG2's ``OpenAIWrapper.create`` and
    ``ConversableAgent.execute_function`` / ``a_execute_function`` to collect
    usage statistics including token usage, inputs, outputs, and timing.
    """

    def __init__(self) -> None:
        super().__init__()
        self._lock = threading.Lock()
        self.last_call_ts = time.time()
        self.step_manager = Context.get().intermediate_step_manager
        self._instrumented = False
        self._original_create: Callable[..., Any] | None = None
        self._original_execute_function: Callable[..., Any] | None = None
        self._original_a_execute_function: Callable[..., Any] | None = None

    def instrument(self) -> None:
        """Monkey-patch AG2 methods with usage-stat collection logic.

        Patches:
            - ``OpenAIWrapper.create`` for LLM call profiling
            - ``ConversableAgent.execute_function`` for sync tool profiling
            - ``ConversableAgent.a_execute_function`` for async tool profiling

        Does nothing if already instrumented or if imports fail.
        """
        if self._instrumented:
            logger.debug("AG2ProfilerHandler already instrumented; skipping.")
            return

        # Patch OpenAIWrapper.create
        try:
            from autogen.oai.client import OpenAIWrapper

            self._original_create = getattr(OpenAIWrapper, "create", None)
            if self._original_create:
                OpenAIWrapper.create = self._create_llm_wrapper(self._original_create)
                logger.debug("Patched OpenAIWrapper.create")
        except ImportError:
            logger.debug("autogen.oai.client not available; skipping LLM instrumentation")

        # Patch ConversableAgent.execute_function and a_execute_function
        try:
            from autogen.agentchat.conversable_agent import ConversableAgent

            self._original_execute_function = getattr(ConversableAgent, "execute_function", None)
            if self._original_execute_function:
                ConversableAgent.execute_function = self._create_tool_wrapper(self._original_execute_function)
                logger.debug("Patched ConversableAgent.execute_function")

            self._original_a_execute_function = getattr(ConversableAgent, "a_execute_function", None)
            if self._original_a_execute_function:
                ConversableAgent.a_execute_function = self._create_async_tool_wrapper(
                    self._original_a_execute_function)
                logger.debug("Patched ConversableAgent.a_execute_function")
        except ImportError:
            logger.debug("autogen.agentchat.conversable_agent not available; skipping tool instrumentation")

        self._instrumented = True
        logger.debug("AG2ProfilerHandler instrumentation applied successfully.")

    def uninstrument(self) -> None:
        """Restore original AG2 methods."""
        try:
            if self._original_create:
                from autogen.oai.client import OpenAIWrapper
                OpenAIWrapper.create = self._original_create
                logger.debug("Restored OpenAIWrapper.create")

            if self._original_execute_function or self._original_a_execute_function:
                from autogen.agentchat.conversable_agent import ConversableAgent
                if self._original_execute_function:
                    ConversableAgent.execute_function = self._original_execute_function
                if self._original_a_execute_function:
                    ConversableAgent.a_execute_function = self._original_a_execute_function
                logger.debug("Restored ConversableAgent methods")

            self._original_create = None
            self._original_execute_function = None
            self._original_a_execute_function = None
            self._instrumented = False
            logger.debug("AG2ProfilerHandler uninstrumented successfully.")
        except Exception:
            logger.exception("Failed to uninstrument AG2ProfilerHandler")

    def _extract_model_name(self, wrapper: Any) -> str:
        """Extract model name from an OpenAIWrapper instance.

        Args:
            wrapper: OpenAIWrapper instance.

        Returns:
            Model name string or ``'unknown_model'``.
        """
        try:
            config_list = getattr(wrapper, "_config_list", [])
            if config_list:
                return str(config_list[0].get("model", "unknown_model"))
        except Exception:
            logger.debug("Failed to extract model from _config_list")

        return "unknown_model"

    def _extract_input_text(self, messages: list[Any]) -> str:
        """Extract text content from a message list.

        Args:
            messages: List of message dictionaries.

        Returns:
            Concatenated text content.
        """
        model_input = ""
        try:
            for message in messages:
                if isinstance(message, dict):
                    content = message.get("content", "")
                elif hasattr(message, "content"):
                    content = message.content
                else:
                    content = str(message)

                if isinstance(content, list):
                    for part in content:
                        if isinstance(part, dict):
                            model_input += str(part.get("text", ""))
                        else:
                            model_input += str(part)
                else:
                    model_input += str(content) if content else ""
        except Exception:
            logger.debug("Error extracting input text from messages")
        return model_input

    def _extract_output_text(self, response: Any) -> str:
        """Extract text content from an LLM response.

        Args:
            response: LLM response conforming to ``ModelClientResponseProtocol``.

        Returns:
            Extracted text content.
        """
        try:
            choices = getattr(response, "choices", [])
            if choices:
                message = getattr(choices[0], "message", None)
                if message:
                    content = getattr(message, "content", None)
                    if content:
                        return str(content)
        except Exception:
            logger.debug("Error extracting output text from response")
        return ""

    def _extract_usage(self, response: Any) -> dict[str, int]:
        """Extract token usage from an LLM response.

        Args:
            response: LLM response object.

        Returns:
            Dictionary with ``prompt_tokens``, ``completion_tokens``, ``total_tokens``.
        """
        try:
            usage = getattr(response, "usage", None)
            if usage is not None:
                return {
                    "prompt_tokens": int(getattr(usage, "prompt_tokens", 0) or 0),
                    "completion_tokens": int(getattr(usage, "completion_tokens", 0) or 0),
                    "total_tokens": int(getattr(usage, "total_tokens", 0) or 0),
                }
        except Exception:
            logger.debug("Error extracting usage from response")
        return {}

    def _create_llm_wrapper(self, original_func: Callable[..., Any]) -> Callable[..., Any]:
        """Create wrapper for ``OpenAIWrapper.create``.

        Args:
            original_func: Original ``create`` method.

        Returns:
            Wrapped function with profiling.
        """
        handler = self

        def wrapped_create(wrapper_self: Any, **config: Any) -> Any:
            now = time.time()
            with handler._lock:
                seconds_between_calls = int(now - handler.last_call_ts)

            model_name = handler._extract_model_name(wrapper_self)
            messages = config.get("messages", [])
            model_input = handler._extract_input_text(messages)

            start_payload = IntermediateStepPayload(
                event_type=IntermediateStepType.LLM_START,
                framework=_FRAMEWORK,
                name=model_name,
                data=StreamEventData(input=model_input),
                metadata=TraceMetadata(chat_inputs=copy.deepcopy(messages)),
                usage_info=UsageInfo(
                    token_usage=TokenUsageBaseModel(),
                    num_llm_calls=1,
                    seconds_between_calls=seconds_between_calls,
                ),
            )
            start_uuid = start_payload.UUID
            handler.step_manager.push_intermediate_step(start_payload)

            try:
                response = original_func(wrapper_self, **config)
            except Exception as e:
                logger.error("Error during AG2 LLM call: %s", e)
                handler.step_manager.push_intermediate_step(
                    IntermediateStepPayload(
                        event_type=IntermediateStepType.LLM_END,
                        span_event_timestamp=time.time(),
                        framework=_FRAMEWORK,
                        name=model_name,
                        data=StreamEventData(input=model_input, output=str(e)),
                        metadata=TraceMetadata(error=str(e)),
                        usage_info=UsageInfo(token_usage=TokenUsageBaseModel()),
                        UUID=start_uuid,
                    ))
                with handler._lock:
                    handler.last_call_ts = time.time()
                raise

            model_output = handler._extract_output_text(response)
            usage_payload = handler._extract_usage(response)

            end_time = time.time()
            handler.step_manager.push_intermediate_step(
                IntermediateStepPayload(
                    event_type=IntermediateStepType.LLM_END,
                    span_event_timestamp=end_time,
                    framework=_FRAMEWORK,
                    name=model_name,
                    data=StreamEventData(input=model_input, output=model_output),
                    metadata=TraceMetadata(chat_responses=model_output),
                    usage_info=UsageInfo(
                        token_usage=TokenUsageBaseModel(**usage_payload),
                        num_llm_calls=1,
                        seconds_between_calls=seconds_between_calls,
                    ),
                    UUID=start_uuid,
                ))

            with handler._lock:
                handler.last_call_ts = end_time

            return response

        return wrapped_create

    def _create_tool_wrapper(self, original_func: Callable[..., Any]) -> Callable[..., Any]:
        """Create wrapper for ``ConversableAgent.execute_function``.

        Args:
            original_func: Original ``execute_function`` method.

        Returns:
            Wrapped function with profiling.
        """
        handler = self

        def wrapped_execute(agent_self: Any, func_call: dict[str, Any], call_id: str | None = None,
                            verbose: bool = False) -> Any:
            now = time.time()
            with handler._lock:
                seconds_between_calls = int(now - handler.last_call_ts)

            func_name = func_call.get("name", "unknown_tool")
            tool_input = func_call.get("arguments", "{}")

            start_payload = IntermediateStepPayload(
                event_type=IntermediateStepType.TOOL_START,
                framework=_FRAMEWORK,
                name=func_name,
                data=StreamEventData(input=str(tool_input)),
                metadata=TraceMetadata(tool_inputs={"arguments": tool_input}),
                usage_info=UsageInfo(
                    token_usage=TokenUsageBaseModel(),
                    num_llm_calls=0,
                    seconds_between_calls=seconds_between_calls,
                ),
            )
            start_uuid = start_payload.UUID
            handler.step_manager.push_intermediate_step(start_payload)

            try:
                result = original_func(agent_self, func_call, call_id, verbose)
            except Exception as e:
                logger.error("AG2 tool execution failed: %s", e)
                handler.step_manager.push_intermediate_step(
                    IntermediateStepPayload(
                        event_type=IntermediateStepType.TOOL_END,
                        span_event_timestamp=time.time(),
                        framework=_FRAMEWORK,
                        name=func_name,
                        data=StreamEventData(input=str(tool_input), output=str(e)),
                        metadata=TraceMetadata(error=str(e)),
                        usage_info=UsageInfo(token_usage=TokenUsageBaseModel()),
                        UUID=start_uuid,
                    ))
                with handler._lock:
                    handler.last_call_ts = time.time()
                raise

            # result is (is_exec_success, result_dict)
            is_success, result_dict = result
            tool_output = str(result_dict.get("content", ""))

            end_time = time.time()
            handler.step_manager.push_intermediate_step(
                IntermediateStepPayload(
                    event_type=IntermediateStepType.TOOL_END,
                    span_event_timestamp=end_time,
                    framework=_FRAMEWORK,
                    name=func_name,
                    data=StreamEventData(input=str(tool_input), output=tool_output),
                    metadata=TraceMetadata(tool_outputs={"result": tool_output, "success": is_success}),
                    usage_info=UsageInfo(token_usage=TokenUsageBaseModel()),
                    UUID=start_uuid,
                ))

            with handler._lock:
                handler.last_call_ts = end_time

            return result

        return wrapped_execute

    def _create_async_tool_wrapper(self, original_func: Callable[..., Any]) -> Callable[..., Any]:
        """Create wrapper for ``ConversableAgent.a_execute_function``.

        Args:
            original_func: Original ``a_execute_function`` method.

        Returns:
            Wrapped async function with profiling.
        """
        handler = self

        async def wrapped_a_execute(agent_self: Any, func_call: dict[str, Any], call_id: str | None = None,
                                    verbose: bool = False) -> Any:
            now = time.time()
            with handler._lock:
                seconds_between_calls = int(now - handler.last_call_ts)

            func_name = func_call.get("name", "unknown_tool")
            tool_input = func_call.get("arguments", "{}")

            start_payload = IntermediateStepPayload(
                event_type=IntermediateStepType.TOOL_START,
                framework=_FRAMEWORK,
                name=func_name,
                data=StreamEventData(input=str(tool_input)),
                metadata=TraceMetadata(tool_inputs={"arguments": tool_input}),
                usage_info=UsageInfo(
                    token_usage=TokenUsageBaseModel(),
                    num_llm_calls=0,
                    seconds_between_calls=seconds_between_calls,
                ),
            )
            start_uuid = start_payload.UUID
            handler.step_manager.push_intermediate_step(start_payload)

            try:
                result = await original_func(agent_self, func_call, call_id, verbose)
            except Exception as e:
                logger.error("AG2 async tool execution failed: %s", e)
                handler.step_manager.push_intermediate_step(
                    IntermediateStepPayload(
                        event_type=IntermediateStepType.TOOL_END,
                        span_event_timestamp=time.time(),
                        framework=_FRAMEWORK,
                        name=func_name,
                        data=StreamEventData(input=str(tool_input), output=str(e)),
                        metadata=TraceMetadata(error=str(e)),
                        usage_info=UsageInfo(token_usage=TokenUsageBaseModel()),
                        UUID=start_uuid,
                    ))
                with handler._lock:
                    handler.last_call_ts = time.time()
                raise

            is_success, result_dict = result
            tool_output = str(result_dict.get("content", ""))

            end_time = time.time()
            handler.step_manager.push_intermediate_step(
                IntermediateStepPayload(
                    event_type=IntermediateStepType.TOOL_END,
                    span_event_timestamp=end_time,
                    framework=_FRAMEWORK,
                    name=func_name,
                    data=StreamEventData(input=str(tool_input), output=tool_output),
                    metadata=TraceMetadata(tool_outputs={"result": tool_output, "success": is_success}),
                    usage_info=UsageInfo(token_usage=TokenUsageBaseModel()),
                    UUID=start_uuid,
                ))

            with handler._lock:
                handler.last_call_ts = end_time

            return result

        return wrapped_a_execute
