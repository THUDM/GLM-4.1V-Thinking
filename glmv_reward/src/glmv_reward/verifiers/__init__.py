# -*- coding: utf-8 -*-


import inspect
from typing import Any

import msgspec

from glmv_reward.configs.verifiers import VerifierConfig
from glmv_reward.utils.logging import get_logger
from glmv_reward.utils.misc import ensure_text
from glmv_reward.utils.msgspec import get_struct_attr, get_struct_tag

from ._base_verifier import Verifier
from .biology_verifier import BiologyVerifier
from .chart_verifier import ChartVerifier
from .chemistry_verifier import ChemistryVerifier
from .counting_verifier import CountingVerifier
from .general_verifier import GeneralVerifier
from .geography_verifier import GeographyVerifier
from .geoquest_verifier import GeoQuestVerifier
from .language_mix_verifier import LanguageMixVerifier
from .liberal_arts_verifier import LiberalArtsVerifier
from .math_verifier import MathVerifier
from .mmsi_verifier import MmsiVerifier
from .multi_image_verifier import MultiImageVerifier
from .ocr_verifier import OCRVerifier
from .physics_verifier import PhysicsVerifier
from .vqa_verifier import VQAVerifier

_VERIFIER_REGISTRY: dict[str, type[Verifier]] = {
    "biology": BiologyVerifier,
    "chart": ChartVerifier,
    "chemistry": ChemistryVerifier,
    "counting": CountingVerifier,
    "general": GeneralVerifier,
    "geography": GeographyVerifier,
    "geoquest": GeoQuestVerifier,
    "language_mix": LanguageMixVerifier,
    "liberal_arts": LiberalArtsVerifier,
    "math": MathVerifier,
    "mmsi": MmsiVerifier,
    "multi_image": MultiImageVerifier,
    "ocr": OCRVerifier,
    "physics": PhysicsVerifier,
    "vqa": VQAVerifier,
}
_VERIFIER_INSTANCE_REGISTRY: dict[str, Verifier] = {}


_logger = get_logger(__name__)


def get_verifier_from_config(config: VerifierConfig, datasource: str) -> Verifier:
    """
    Factory function to get an instance of a verifier.
    """
    verifier_type = get_struct_tag(config)
    if verifier_type is None:
        err_msg = f"Failed to guess the verifier type from the config: {ensure_text(msgspec.json.encode(config))}."
        raise ValueError(err_msg)

    verifier_type = verifier_type.lower()
    if verifier_type not in _VERIFIER_REGISTRY:
        err_msg = f"Verifier '{verifier_type}' is not supported."
        raise ValueError(err_msg)

    verifier_cls = _VERIFIER_REGISTRY[verifier_type]
    verifier_instance_key = f"{datasource}@{verifier_type}"

    if verifier_instance_key not in _VERIFIER_INSTANCE_REGISTRY:
        kwargs: dict[str, Any] = {}
        for key in inspect.signature(verifier_cls.__init__).parameters:
            if key == "self":
                continue
            try:
                value = get_struct_attr(config, key)
            except Exception:
                _logger.debug("Configuration field `%s` is missing, will use the default value.", key)
            else:
                kwargs[key] = value
        _VERIFIER_INSTANCE_REGISTRY[verifier_instance_key] = verifier_cls(**kwargs)
    return _VERIFIER_INSTANCE_REGISTRY[verifier_instance_key]
