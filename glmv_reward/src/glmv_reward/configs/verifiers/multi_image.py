# -*- coding: utf-8 -*-


from typing import Optional

import msgspec


class MultiImageVerifierConfig(msgspec.Struct, frozen=True, tag_field="verifier_type", tag="multi_image"):
    sympy_tolerance: float = 1e-5
    strict_boxed_extraction: bool = True
    enable_llm_judge_fallback: bool = True
    llm_api_key: Optional[str] = None
    llm_judge_url: Optional[str] = None
    llm_judge_prompt_template: Optional[str] = None
    llm_model: str = "glm-4-flash"
    llm_max_tokens: int = 10
    llm_temperature: float = 0.1
    llm_top_p: float = 1.0
