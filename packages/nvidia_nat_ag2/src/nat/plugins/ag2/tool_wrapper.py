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
"""Tool wrapper for AG2 integration with NeMo Agent Toolkit.

Converts NeMo Agent Toolkit ``Function`` objects into AG2 ``Tool`` objects that can be
registered with AG2 ``ConversableAgent`` instances for LLM-driven tool calling.
"""

import inspect
import logging
from collections.abc import AsyncIterator
from typing import Any

from pydantic import BaseModel

from nat.builder.builder import Builder
from nat.builder.framework_enum import LLMFrameworkEnum
from nat.builder.function import Function
from nat.cli.register_workflow import register_tool_wrapper
from nat.utils.type_utils import DecomposedType

logger = logging.getLogger(__name__)


def _resolve_type(t: Any) -> Any:
    """Return the non-None member of a Union/PEP 604 union; otherwise return the type unchanged.

    Args:
        t: The type to resolve.

    Returns:
        The resolved type.
    """
    resolved = DecomposedType(t)
    if resolved.is_optional:
        return resolved.get_optional_type().type
    return resolved.type


@register_tool_wrapper(wrapper_type=LLMFrameworkEnum.AG2)
def ag2_tool_wrapper(
    name: str,
    fn: Function,
    _builder: Builder,
) -> Any:
    """Wrap a NeMo Agent Toolkit ``Function`` as an AG2 ``Tool``.

    This creates an AG2 ``Tool`` instance from a NAT function by:

    1. Extracting the input schema from the NAT function's Pydantic model.
    2. Building a Python function with the correct signature and type annotations.
    3. Wrapping the NAT function's async invocation/streaming in the AG2 tool callable.

    Args:
        name: The name of the tool.
        fn: The NAT function to wrap.
        _builder: The NAT workflow builder (unused).

    Returns:
        An AG2 ``Tool`` instance wrapping the NAT function.
    """
    from autogen.tools import Tool

    async def callable_ainvoke(**kwargs: Any) -> Any:
        """Async function to invoke the NAT function.

        Args:
            **kwargs: Keyword arguments to pass to the NAT function.

        Returns:
            The result of invoking the NAT function.
        """
        return await fn.acall_invoke(**kwargs)

    async def callable_astream(**kwargs: Any) -> AsyncIterator[Any]:
        """Async generator to stream results from the NAT function.

        Args:
            **kwargs: Keyword arguments to pass to the NAT function.

        Yields:
            Streamed items from the NAT function.
        """
        async for item in fn.acall_stream(**kwargs):
            yield item

    # Choose the appropriate callable based on streaming capability
    if fn.has_streaming_output and not fn.has_single_output:
        logger.debug("Creating streaming AG2 Tool for: %s", name)
        tool_func = callable_astream
    else:
        logger.debug("Creating non-streaming AG2 Tool for: %s", name)
        tool_func = callable_ainvoke

    # Build a proper function signature from the input schema so AG2 can generate
    # the correct tool schema for the LLM
    input_schema = fn.input_schema
    if input_schema is not None and issubclass(input_schema, BaseModel):
        params: list[inspect.Parameter] = []
        annotations: dict[str, Any] = {}
        model_fields = getattr(input_schema, "model_fields", {})

        for param_name, model_field in model_fields.items():
            resolved_type = _resolve_type(model_field.annotation)
            default = inspect.Parameter.empty if model_field.is_required() else model_field.default
            params.append(
                inspect.Parameter(param_name,
                                  inspect.Parameter.POSITIONAL_OR_KEYWORD,
                                  annotation=resolved_type,
                                  default=default))
            annotations[param_name] = resolved_type

        tool_func.__name__ = name
        tool_func.__doc__ = fn.description or name
        tool_func.__signature__ = inspect.Signature(parameters=params)
        tool_func.__annotations__ = annotations

    return Tool(
        name=name,
        description=fn.description or "No description provided.",
        func_or_tool=tool_func,
    )
