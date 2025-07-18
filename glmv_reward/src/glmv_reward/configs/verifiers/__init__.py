# -*- coding: utf-8 -*-


from typing import Union

from .biology import BiologyVerifierConfig
from .chart import ChartVerifierConfig
from .chemistry import ChemistryVerifierConfig
from .counting import CountingVerifierConfig
from .general import GeneralVerifierConfig
from .geography import GeographyVerifierConfig
from .geoquest import GeoQuestVerifierConfig
from .language_mix import LanguageMixVerifierConfig
from .liberal_arts import LiberalArtsVerifierConfig
from .long_doc import LongDocVerifierConfig
from .math import MathVerifierConfig
from .mmsi import MmsiVerifierConfig
from .multi_image import MultiImageVerifierConfig
from .ocr import OCRVerifierConfig
from .physics import PhysicsVerifierConfig
from .vqa import VQAVerifierConfig
from .androidworld import AndroidworldVerifierConfig
from .webvoyager import WebvoyagerVerifierConfig
from .osworld import OsworldVerifierConfig
from .file_based import FileBasedVerifierConfig

VerifierConfig = Union[
    BiologyVerifierConfig,
    ChartVerifierConfig,
    ChemistryVerifierConfig,
    CountingVerifierConfig,
    GeneralVerifierConfig,
    GeographyVerifierConfig,
    GeoQuestVerifierConfig,
    LanguageMixVerifierConfig,
    LiberalArtsVerifierConfig,
    LongDocVerifierConfig,
    MathVerifierConfig,
    MmsiVerifierConfig,
    MultiImageVerifierConfig,
    OCRVerifierConfig,
    PhysicsVerifierConfig,
    VQAVerifierConfig,
    AndroidworldVerifierConfig,
    WebvoyagerVerifierConfig,
    OsworldVerifierConfig,
    FileBasedVerifierConfig,
]
