"""
Tests for translator.py's language-pack handling, against a fake argostranslate.

Argos ships a pack between English and each other language, plus a few direct pairs such as
es <-> pt. Any other pair, for example fr -> de, is translated by chaining two packs through
English, so setting such a pair up means installing both of them.
"""

import pytest


class TestIsPackageInstalled:
    def test_nothing_installed(self, fake_argos, translator_module):
        assert not translator_module.is_package_installed("fr", "de")

    def test_direct_pack(self, fake_argos, translator_module):
        fake_argos.add_installed("es", "pt")
        assert translator_module.is_package_installed("es", "pt")

    def test_both_legs_of_a_pivot_through_english(self, fake_argos, translator_module):
        fake_argos.add_installed("fr", "en")
        fake_argos.add_installed("en", "de")
        assert translator_module.is_package_installed("fr", "de")

    @pytest.mark.parametrize("leg", [("fr", "en"), ("en", "de")])
    def test_one_leg_alone_is_not_enough(self, fake_argos, translator_module, leg):
        fake_argos.add_installed(*leg)
        assert not translator_module.is_package_installed("fr", "de")

    def test_legs_must_run_in_the_right_direction(self, fake_argos, translator_module):
        # en -> fr and de -> en would translate de -> fr, not fr -> de.
        fake_argos.add_installed("en", "fr")
        fake_argos.add_installed("de", "en")
        assert not translator_module.is_package_installed("fr", "de")

    def test_english_pairs_never_pivot(self, fake_argos, translator_module):
        fake_argos.add_installed("fr", "en")
        assert translator_module.is_package_installed("fr", "en")
        assert not translator_module.is_package_installed("en", "fr")


class TestSetupTranslationModels:
    def test_same_language_needs_nothing(self, fake_argos, translator_module):
        assert translator_module.setup_translation_models("en", "en") is True
        assert fake_argos.index_updates == 0

    def test_installed_pair_makes_no_network_calls(self, fake_argos, translator_module):
        fake_argos.add_installed("fr", "en")
        fake_argos.add_installed("en", "de")
        assert translator_module.setup_translation_models("fr", "de") is True
        assert fake_argos.index_updates == 0
        assert fake_argos.installs == []

    def test_direct_pack_is_used_when_the_index_has_one(self, fake_argos, translator_module):
        fake_argos.add_available("es", "pt")
        fake_argos.add_available("es", "en")
        fake_argos.add_available("en", "pt")
        assert translator_module.setup_translation_models("es", "pt") is True
        assert fake_argos.installs == [("es", "pt")]

    def test_installs_both_legs_when_there_is_no_direct_pack(self, fake_argos, translator_module):
        # The reported bug: fr -> de has no direct pack, so setup used to fail.
        fake_argos.add_available("fr", "en")
        fake_argos.add_available("en", "de")
        assert translator_module.setup_translation_models("fr", "de") is True
        assert fake_argos.installs == [("fr", "en"), ("en", "de")]
        assert translator_module.is_package_installed("fr", "de")

    def test_installs_only_the_missing_leg(self, fake_argos, translator_module):
        fake_argos.add_installed("fr", "en")
        fake_argos.add_available("fr", "en")
        fake_argos.add_available("en", "de")
        assert translator_module.setup_translation_models("fr", "de") is True
        assert fake_argos.installs == [("en", "de")]

    def test_fails_without_downloading_if_a_leg_is_not_in_the_index(
        self, fake_argos, translator_module
    ):
        fake_argos.add_available("fr", "en")  # en -> de is missing
        assert translator_module.setup_translation_models("fr", "de") is False
        assert fake_argos.downloaded_pairs == []

    def test_fails_when_a_download_fails(self, fake_argos, translator_module):
        fake_argos.add_available("fr", "en")
        broken = fake_argos.add_available("en", "de")
        broken.download_error = OSError("network down")
        assert translator_module.setup_translation_models("fr", "de") is False
        assert not translator_module.is_package_installed("fr", "de")

    def test_unknown_pair_fails(self, fake_argos, translator_module):
        assert translator_module.setup_translation_models("fr", "de") is False
        assert fake_argos.installs == []

    def test_a_stale_index_is_used_when_the_update_fails(self, fake_argos, translator_module):
        fake_argos.index_error = OSError("offline")
        fake_argos.add_available("fr", "en")
        fake_argos.add_available("en", "de")
        assert translator_module.setup_translation_models("fr", "de") is True
        assert fake_argos.index_updates == 1
