# -*- coding: utf-8 -*-


import json
from typing import Optional, cast

import requests

from .logging import get_logger

_logger = get_logger(__name__)


def post_query_llm(
    prompt: str,
    api_key: str,
    url: str = "https://open.bigmodel.cn/api/paas/v4/chat/completions",
    model: str = "glm-4-flash",
    image_file: Optional[str] = None,
    max_tokens: Optional[int] = 10,
    temperature: Optional[float] = 0.1,
    top_p: Optional[float] = 1.0,
    timeout: Optional[int] = 120,
) -> str:
    """
    Sends a query to Zhipu AI API endpoint.

    Args:
        prompt: The prompt to generate completions for.
        api_key: The API key used for authentication.
        url: The chat completion API endpoint (default: https://open.bigmodel.cn/api/paas/v4/chat/completions).
        model: Model name used to generate the response (default: glm-4-flash).
        image_file: The image file to generation completions for (not currently supported).
        max_tokens: The maximum number of tokens that can be generated.
        temperature: The sampling temperature used for the generation.
        top_p: The parameter for nucleus sampling, where the model considers the
          results of the tokens with top_p probability mass.
        timeout: The timeout value for the LLM request.

    Returns:
        The response content from the API.

    """
    del image_file  # Not currently supported

    # Get API key from environment

    messages: list[dict[str, object]] = [{"role": "user", "content": prompt}]

    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "stream": False,
    }

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }

    try:
        response = requests.post(url, headers=headers, data=json.dumps(payload), timeout=timeout)
        response.raise_for_status()
        response_data = response.json()
    except requests.exceptions.RequestException as e:
        _logger.warning("HTTP request error in `post_query_llm`: %s", e)
        return ""
    except KeyError as e:
        _logger.warning("Response parsing error in `post_query_llm`: %s", e)
        return ""
    except Exception as e:
        _logger.warning("Unexpected error in `post_query_llm` due to exception: %s", repr(e))
        return ""
    else:
        # Extract content from Zhipu AI response format
        if "choices" in response_data and len(response_data["choices"]) > 0:
            content = response_data["choices"][0]["message"]["content"]
            return cast(str, content)
        _logger.error("Unexpected response format from Zhipu AI API: %s", response_data)
        return ""
