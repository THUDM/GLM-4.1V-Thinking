# -*- coding: utf-8 -*-


from typing import Optional

import msgspec


class GeneralVerifierConfig(msgspec.Struct, frozen=True, tag_field="verifier_type", tag="general"):
    llm_api_key: str
    llm_judge_url: str
    llm_judge_prompt_template: str
    llm_model: str = "glm-4-flash"
    llm_max_tokens: int = 10
    llm_temperature: float = 0.1
    llm_top_p: float = 1.0
    answer_extraction_regex: Optional[str] = None
