# -*- coding: utf-8 -*-


import msgspec


class InstructVerifierConfig(msgspec.Struct, frozen=True, tag_field="verifier_type", tag="instruct"):
    llm_judge_url: str
    llm_model: str
