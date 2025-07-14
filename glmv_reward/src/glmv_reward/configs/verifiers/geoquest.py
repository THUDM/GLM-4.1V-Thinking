# -*- coding: utf-8 -*-


from collections.abc import Sequence
from typing import Optional, Union

import msgspec


class GeoQuestVerifierConfig(msgspec.Struct, frozen=True, tag_field="verifier_type", tag="geoquest"):
    llm_api_key: Union[Sequence[str], str]
    llm_judge_url: Union[Sequence[str], str]
    llm_judge_prompt_template: str
    llm_model: Optional[Union[Sequence[str], str]] = None
    llm_max_tokens: int = 4096
    llm_temperature: float = 0.8
    llm_top_p: float = 0.6
    strict_boxed_extraction: bool = True
