# -*- coding: utf-8 -*-


from typing import Optional

import msgspec


class PostRLFormatVerifierConfig(msgspec.Struct, frozen=True, tag_field="verifier_type", tag="post_rl_format"):
    forbid_long_paragraph_mixing: bool = False
    forbid_box: bool = False
    forbid_repeat: bool = False
    use_instruct_reward_model: bool = False
    instruct_llm_judge_url: Optional[str] = None
    instruct_llm_model: Optional[str] = None
    use_fluency_reward_model: bool = False
    fluency_llm_judge_url: Optional[str] = None
    fluency_llm_model: Optional[str] = None
