# -*- coding: utf-8 -*-


from collections.abc import Sequence
from typing import Optional, Union

import msgspec


class ChemistryVerifierConfig(msgspec.Struct, frozen=True, tag_field="verifier_type", tag="chemistry"):
    sympy_tolerance: float = 1e-5
    strict_boxed_extraction: bool = True
    enable_llm_judge_fallback: bool = True
    llm_api_key: Optional[Union[Sequence[str], str]] = None
    llm_judge_url: Optional[Union[Sequence[str], str]] = None
    llm_judge_prompt_template: Optional[str] = None
    llm_model: Optional[Union[Sequence[str], str]] = None
    llm_max_tokens: int = 10
    llm_temperature: float = 0.1
    llm_top_p: float = 1.0
