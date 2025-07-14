# -*- coding: utf-8 -*-


from collections.abc import Sequence
from typing import Optional, Union

import msgspec


class OCRVerifierConfig(msgspec.Struct, frozen=True, tag_field="verifier_type", tag="ocr"):
    strict_boxed_extraction: bool = True
    edit_distance_upper_bound: float = 1.0
    edit_distance_lower_bound: float = 0.0
    ignore_case: bool = False
    enable_llm_judge_fallback: bool = True
    llm_api_key: Optional[Union[Sequence[str], str]] = None
    llm_judge_url: Optional[Union[Sequence[str], str]] = None
    llm_judge_prompt_template: Optional[str] = None
    llm_model: Optional[Union[Sequence[str], str]] = None
    llm_max_tokens: int = 10
    llm_temperature: float = 0.1
    llm_top_p: float = 1.0
