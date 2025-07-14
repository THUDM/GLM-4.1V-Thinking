# -*- coding: utf-8 -*-


import msgspec


class LanguageMixVerifierConfig(msgspec.Struct, frozen=True, tag_field="verifier_type", tag="language_mix"):
    pass
