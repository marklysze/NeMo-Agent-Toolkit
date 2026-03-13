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
"""AG2 LLM client registrations for NeMo Agent Toolkit.

This module provides AG2-compatible LLM configuration wrappers for the following providers:

Supported Providers
-------------------
- **OpenAI**: Direct OpenAI API integration via AG2's ``LLMConfig``
- **Azure OpenAI**: Azure-hosted OpenAI models
- **NVIDIA NIM**: OpenAI-compatible endpoints for NVIDIA models
- **LiteLLM**: Unified interface to multiple LLM providers
- **AWS Bedrock**: Amazon Bedrock models via AG2's native Bedrock support

Each wrapper:

- Converts NeMo Agent Toolkit LLM configurations to AG2 ``LLMConfig`` objects
- Patches clients with NeMo Agent Toolkit retry logic from ``RetryMixin``
- Injects chain-of-thought prompts when ``ThinkingMixin`` is configured

See the AG2 documentation at https://docs.ag2.ai for model provider details.
"""

import logging
import os
from collections.abc import AsyncGenerator
from typing import Any
from typing import TypeVar

from nat.builder.builder import Builder
from nat.builder.framework_enum import LLMFrameworkEnum
from nat.cli.register_workflow import register_llm_client
from nat.data_models.common import get_secret_value
from nat.data_models.llm import LLMBaseConfig
from nat.data_models.retry_mixin import RetryMixin
from nat.data_models.thinking_mixin import ThinkingMixin
from nat.llm.aws_bedrock_llm import AWSBedrockModelConfig
from nat.llm.azure_openai_llm import AzureOpenAIModelConfig
from nat.llm.litellm_llm import LiteLlmModelConfig
from nat.llm.nim_llm import NIMModelConfig
from nat.llm.openai_llm import OpenAIModelConfig
from nat.llm.utils.thinking import BaseThinkingInjector
from nat.llm.utils.thinking import FunctionArgumentWrapper
from nat.llm.utils.thinking import patch_with_thinking
from nat.utils.exception_handlers.automatic_retries import patch_with_retry
from nat.utils.type_utils import override

logger = logging.getLogger(__name__)

ModelType = TypeVar("ModelType")

_EXTRA_PARAM_FIELDS = {
    "temperature": "temperature",
    "max_tokens": "max_tokens",
    "top_p": "top_p",
    "request_timeout": "timeout",
    "seed": "seed",
}


def _extract_extra_params(llm_config: LLMBaseConfig) -> dict[str, Any]:
    """Extract LLMConfig-level parameters from a NAT config using getattr.

    Only includes parameters that exist on the config and are not None.

    Args:
        llm_config: NAT LLM configuration.

    Returns:
        Dictionary of extra parameters for AG2 LLMConfig.
    """
    params: dict[str, Any] = {}
    for nat_field, ag2_key in _EXTRA_PARAM_FIELDS.items():
        value = getattr(llm_config, nat_field, None)
        if value is not None:
            params[ag2_key] = value
    return params


def _build_ag2_llm_config(config_entry: dict[str, Any], **kwargs: Any) -> Any:
    """Build an AG2 LLMConfig from a config entry dictionary.

    Args:
        config_entry: Dictionary of LLM configuration parameters.
        **kwargs: Additional LLMConfig-level parameters (temperature, max_tokens, etc.).

    Returns:
        AG2 LLMConfig object.
    """
    from autogen import LLMConfig

    # Suppress AG2's "Model not found" cost warning for custom/NIM models
    # by providing a default price when none is specified.
    config_entry.setdefault("price", [0.0, 0.0])

    return LLMConfig(config_entry, **kwargs)


def _patch_ag2_client_based_on_config(llm_config_obj: ModelType, nat_config: LLMBaseConfig) -> ModelType:
    """Patch an AG2 LLMConfig with NeMo Agent Toolkit mixins (retry, thinking).

    Args:
        llm_config_obj: The AG2 LLMConfig object to patch.
        nat_config: The NeMo Agent Toolkit LLM configuration containing mixin settings.

    Returns:
        The patched AG2 LLMConfig object.
    """

    class AG2ThinkingInjector(BaseThinkingInjector):
        """Thinking injector for AG2 message format.

        Injects a system message at the start of the message list to enable
        chain-of-thought prompting for supported models.
        """

        @override
        def inject(self, messages: list, *args: Any, **kwargs: Any) -> FunctionArgumentWrapper:
            """Inject thinking system prompt into AG2 messages.

            Args:
                messages: List of AG2 message dictionaries.
                *args: Additional positional arguments.
                **kwargs: Additional keyword arguments.

            Returns:
                Wrapper containing modified args and kwargs.
            """
            thinking_prompt = self.system_prompt
            if not thinking_prompt:
                return FunctionArgumentWrapper(messages, *args, **kwargs)

            system_message = {"role": "system", "content": thinking_prompt}
            new_messages = [system_message] + messages
            return FunctionArgumentWrapper(new_messages, *args, **kwargs)

    if isinstance(nat_config, RetryMixin):
        llm_config_obj = patch_with_retry(llm_config_obj,
                                          retries=nat_config.num_retries,
                                          retry_codes=nat_config.retry_on_status_codes,
                                          retry_on_messages=nat_config.retry_on_errors)

    if isinstance(nat_config, ThinkingMixin) and nat_config.thinking_system_prompt is not None:
        llm_config_obj = patch_with_thinking(
            llm_config_obj,
            AG2ThinkingInjector(
                system_prompt=nat_config.thinking_system_prompt,
                function_names=[
                    "create",
                ],
            ))

    return llm_config_obj


@register_llm_client(config_type=OpenAIModelConfig, wrapper_type=LLMFrameworkEnum.AG2)
async def openai_ag2(llm_config: OpenAIModelConfig, _builder: Builder) -> AsyncGenerator[Any, None]:
    """Build an AG2 LLMConfig from an OpenAI NeMo Agent Toolkit configuration.

    Args:
        llm_config: OpenAI configuration declared in the workflow.
        _builder: Builder instance provided by the workflow factory (unused).

    Yields:
        AG2 ``LLMConfig`` objects ready for use with AG2 agents.
    """
    config_entry: dict[str, Any] = {"model": llm_config.model_name}

    if (api_key := get_secret_value(llm_config.api_key) or os.getenv("OPENAI_API_KEY")):
        config_entry["api_key"] = api_key
    if (base_url := llm_config.base_url or os.getenv("OPENAI_BASE_URL")):
        config_entry["base_url"] = base_url

    ag2_config = _build_ag2_llm_config(config_entry, **_extract_extra_params(llm_config))
    yield _patch_ag2_client_based_on_config(ag2_config, llm_config)


@register_llm_client(config_type=AzureOpenAIModelConfig, wrapper_type=LLMFrameworkEnum.AG2)
async def azure_openai_ag2(llm_config: AzureOpenAIModelConfig, _builder: Builder) -> AsyncGenerator[Any, None]:
    """Build an AG2 LLMConfig from an Azure OpenAI NeMo Agent Toolkit configuration.

    Args:
        llm_config: Azure OpenAI configuration declared in the workflow.
        _builder: Builder instance provided by the workflow factory (unused).

    Yields:
        AG2 ``LLMConfig`` objects configured for Azure OpenAI deployments.
    """
    config_entry: dict[str, Any] = {
        "model": llm_config.azure_deployment,
        "api_type": "azure",
        "api_key": get_secret_value(llm_config.api_key),
        "base_url": f"{llm_config.azure_endpoint}/openai/deployments/{llm_config.azure_deployment}",
        "api_version": llm_config.api_version,
    }

    ag2_config = _build_ag2_llm_config(config_entry, **_extract_extra_params(llm_config))
    yield _patch_ag2_client_based_on_config(ag2_config, llm_config)


@register_llm_client(config_type=NIMModelConfig, wrapper_type=LLMFrameworkEnum.AG2)
async def nim_ag2(llm_config: NIMModelConfig, _builder: Builder) -> AsyncGenerator[Any, None]:
    """Build an AG2 LLMConfig for NVIDIA NIM endpoints.

    NIM endpoints are OpenAI-compatible, so AG2 connects via its standard OpenAI provider.

    Args:
        llm_config: NIM model configuration from the workflow.
        _builder: Builder instance supplied during workflow construction (unused).

    Yields:
        AG2 ``LLMConfig`` objects configured for NVIDIA NIM endpoints.
    """
    base_url = llm_config.base_url or "https://integrate.api.nvidia.com/v1"
    api_key = get_secret_value(llm_config.api_key) or os.getenv("NVIDIA_API_KEY")

    # Use a dummy key for custom NIM endpoints without authentication
    if llm_config.base_url and llm_config.base_url.strip() and api_key is None:
        api_key = "dummy-api-key"

    config_entry: dict[str, Any] = {
        "model": llm_config.model_name,
        "api_key": api_key,
        "base_url": base_url,
    }

    ag2_config = _build_ag2_llm_config(config_entry, **_extract_extra_params(llm_config))
    yield _patch_ag2_client_based_on_config(ag2_config, llm_config)


@register_llm_client(config_type=LiteLlmModelConfig, wrapper_type=LLMFrameworkEnum.AG2)
async def litellm_ag2(llm_config: LiteLlmModelConfig, _builder: Builder) -> AsyncGenerator[Any, None]:
    """Build an AG2 LLMConfig for LiteLLM providers.

    LiteLLM provides a unified interface to multiple LLM providers. AG2 connects
    via its standard OpenAI-compatible interface.

    Args:
        llm_config: LiteLLM model configuration from the workflow.
        _builder: Builder instance supplied during workflow construction (unused).

    Yields:
        AG2 ``LLMConfig`` objects configured for LiteLLM providers.
    """
    config_entry: dict[str, Any] = {"model": llm_config.model_name}

    if llm_config.api_key is not None:
        config_entry["api_key"] = get_secret_value(llm_config.api_key)
    if llm_config.base_url is not None:
        config_entry["base_url"] = llm_config.base_url

    ag2_config = _build_ag2_llm_config(config_entry, **_extract_extra_params(llm_config))
    yield _patch_ag2_client_based_on_config(ag2_config, llm_config)


@register_llm_client(config_type=AWSBedrockModelConfig, wrapper_type=LLMFrameworkEnum.AG2)
async def bedrock_ag2(llm_config: AWSBedrockModelConfig, _builder: Builder) -> AsyncGenerator[Any, None]:
    """Build an AG2 LLMConfig for AWS Bedrock models.

    AG2 supports Bedrock through its built-in provider support. Credentials are
    loaded in the following priority:

    1. Explicit values from ``credentials_profile_name`` in the AWS profile.
    2. Standard environment variables (``AWS_ACCESS_KEY_ID``, ``AWS_SECRET_ACCESS_KEY``).
    3. Ambient credentials from the compute environment (IAM role).

    Args:
        llm_config: AWS Bedrock model configuration from the workflow.
        _builder: Builder instance supplied during workflow construction (unused).

    Yields:
        AG2 ``LLMConfig`` objects configured for AWS Bedrock endpoints.
    """
    config_entry: dict[str, Any] = {
        "model": llm_config.model_name,
        "api_type": "bedrock",
    }

    if llm_config.region_name not in (None, "None"):
        config_entry["aws_region"] = llm_config.region_name
    if llm_config.credentials_profile_name is not None:
        config_entry["aws_profile_name"] = llm_config.credentials_profile_name
    if llm_config.base_url is not None:
        config_entry["base_url"] = llm_config.base_url

    ag2_config = _build_ag2_llm_config(config_entry, **_extract_extra_params(llm_config))
    yield _patch_ag2_client_based_on_config(ag2_config, llm_config)
