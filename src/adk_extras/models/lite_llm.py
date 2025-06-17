"""
LiteLlmWithStreaming - A custom wrapper for LiteLlm that fixes streaming issues.

This module provides a drop-in replacement for the LiteLlm class that properly
implements async streaming using acompletion instead of the synchronous completion method.
"""

from typing import AsyncGenerator, Dict, Any

from google.adk.models.lite_llm import (
    LiteLlm,
    _model_response_to_chunk,
    _message_to_generate_content_response,
    _get_completion_inputs,
    _model_response_to_generate_content_response,
)
from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse
from google.genai import types
from litellm import (
    ChatCompletionAssistantMessage,
    ChatCompletionMessageToolCall,
    Function,
)


class LiteLlmWithStreaming(LiteLlm):
    """
    Custom wrapper that fixes the streaming issue in LiteLlm.

    This class inherits from LiteLlm and overrides the generate_content_async
    method to use proper async streaming instead of blocking synchronous calls.
    """

    async def generate_content_async(
        self, llm_request: LlmRequest, stream: bool = False
    ) -> AsyncGenerator[LlmResponse, None]:
        """Generates content asynchronously with proper streaming support.

        This method overrides the original implementation to use async iteration
        with acompletion instead of synchronous completion for streaming.

        Args:
            llm_request: LlmRequest, the request to send to the LiteLlm model.
            stream: bool = False, whether to do streaming call.

        Yields:
            LlmResponse: The model response.
        """

        self._maybe_append_user_content(llm_request)

        messages, tools, response_format = _get_completion_inputs(llm_request)

        completion_args = {
            "model": self.model,
            "messages": messages,
            "tools": tools,
            "response_format": response_format,
        }
        completion_args.update(self._additional_args)

        if stream:
            text = ""
            # Track function calls by index
            function_calls = {}  # index -> {name, args, id}
            completion_args["stream"] = True
            aggregated_llm_response = None
            aggregated_llm_response_with_tool_call = None
            usage_metadata = None
            fallback_index = 0

            # THE KEY FIX: Use async iteration with acompletion instead of synchronous completion
            async for part in await self.llm_client.acompletion(**completion_args):
                for chunk, finish_reason in _model_response_to_chunk(part):
                    if (
                        hasattr(chunk, "id")
                        and hasattr(chunk, "name")
                        and hasattr(chunk, "args")
                    ):  # FunctionChunk
                        index = getattr(chunk, "index", fallback_index)
                        if index not in function_calls:
                            function_calls[index] = {"name": "", "args": "", "id": None}

                        if chunk.name:
                            function_calls[index]["name"] += chunk.name
                        if chunk.args:
                            function_calls[index]["args"] += chunk.args

                            # check if args is completed (workaround for improper chunk indexing)
                            try:
                                import json

                                json.loads(function_calls[index]["args"])
                                fallback_index += 1
                            except json.JSONDecodeError:
                                pass

                        function_calls[index]["id"] = (
                            chunk.id or function_calls[index]["id"] or str(index)
                        )
                    elif hasattr(chunk, "text"):  # TextChunk
                        text += chunk.text
                        yield _message_to_generate_content_response(
                            ChatCompletionAssistantMessage(
                                role="assistant",
                                content=chunk.text,
                            ),
                            is_partial=True,
                        )
                    elif hasattr(chunk, "prompt_tokens"):  # UsageMetadataChunk
                        usage_metadata = types.GenerateContentResponseUsageMetadata(
                            prompt_token_count=chunk.prompt_tokens,
                            candidates_token_count=chunk.completion_tokens,
                            total_token_count=chunk.total_tokens,
                        )

                    if (
                        finish_reason == "tool_calls" or finish_reason == "stop"
                    ) and function_calls:
                        tool_calls = []
                        for index, func_data in function_calls.items():
                            if func_data["id"]:
                                tool_calls.append(
                                    ChatCompletionMessageToolCall(
                                        type="function",
                                        id=func_data["id"],
                                        function=Function(
                                            name=func_data["name"],
                                            arguments=func_data["args"],
                                            index=index,
                                        ),
                                    )
                                )
                        aggregated_llm_response_with_tool_call = (
                            _message_to_generate_content_response(
                                ChatCompletionAssistantMessage(
                                    role="assistant",
                                    content="",
                                    tool_calls=tool_calls,
                                )
                            )
                        )
                        function_calls.clear()
                    elif finish_reason == "stop" and text:
                        aggregated_llm_response = _message_to_generate_content_response(
                            ChatCompletionAssistantMessage(
                                role="assistant", content=text
                            )
                        )
                        text = ""

            # waiting until streaming ends to yield the llm_response as litellm tends
            # to send chunk that contains usage_metadata after the chunk with
            # finish_reason set to tool_calls or stop.
            if aggregated_llm_response:
                if usage_metadata:
                    aggregated_llm_response.usage_metadata = usage_metadata
                    usage_metadata = None
                yield aggregated_llm_response

            if aggregated_llm_response_with_tool_call:
                if usage_metadata:
                    aggregated_llm_response_with_tool_call.usage_metadata = (
                        usage_metadata
                    )
                yield aggregated_llm_response_with_tool_call

        else:
            response = await self.llm_client.acompletion(**completion_args)
            yield _model_response_to_generate_content_response(response)
