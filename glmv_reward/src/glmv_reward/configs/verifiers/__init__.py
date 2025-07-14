# -*- coding: utf-8 -*-


from typing import Union

from .biology import BiologyVerifierConfig
from .chart import ChartVerifierConfig
from .chemistry import ChemistryVerifierConfig
from .counting import CountingVerifierConfig
from .fluency import FluencyVerifierConfig
from .general import GeneralVerifierConfig
from .geography import GeographyVerifierConfig
from .geoquest import GeoQuestVerifierConfig
from .instruct import InstructVerifierConfig
from .language_mix import LanguageMixVerifierConfig
from .liberal_arts import LiberalArtsVerifierConfig
from .math import MathVerifierConfig
from .mmsi import MmsiVerifierConfig
from .multi_image import MultiImageVerifierConfig
from .ocr import OCRVerifierConfig
from .physics import PhysicsVerifierConfig
from .post_rl_format import PostRLFormatVerifierConfig
from .vqa import VQAVerifierConfig

VerifierConfig = Union[
    BiologyVerifierConfig,
    ChartVerifierConfig,
    ChemistryVerifierConfig,
    CountingVerifierConfig,
    FluencyVerifierConfig,
    GeneralVerifierConfig,
    GeographyVerifierConfig,
    GeoQuestVerifierConfig,
    InstructVerifierConfig,
    LanguageMixVerifierConfig,
    LiberalArtsVerifierConfig,
    MathVerifierConfig,
    MmsiVerifierConfig,
    MultiImageVerifierConfig,
    OCRVerifierConfig,
    PhysicsVerifierConfig,
    PostRLFormatVerifierConfig,
    VQAVerifierConfig,
]
