# -*- coding: utf-8 -*-


from collections.abc import Mapping

import msgspec

from .verifiers import VerifierConfig


class RewardSystemConfig(msgspec.Struct, frozen=True):
    datasource_reward_config_mapping: Mapping[str, str]
    reward_configs: Mapping[str, VerifierConfig]
    enable_mix_verifier: bool = True
    reward_log_dir: str = "logs"
