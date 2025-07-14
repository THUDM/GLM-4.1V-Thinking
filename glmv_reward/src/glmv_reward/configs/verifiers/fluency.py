# -*- coding: utf-8 -*-


import msgspec


class FluencyVerifierConfig(msgspec.Struct, frozen=True, tag_field="verifier_type", tag="fluency"):
    llm_judge_url: str
    llm_model: str
