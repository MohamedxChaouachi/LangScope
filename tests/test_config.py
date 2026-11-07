"""
config: the log level comes from LANGSCOPE_LOGLEVEL and falls back to INFO.
"""

import logging

import pytest

from config import LOG_FORMAT, Config, configure_logging, load_config


class TestLoadConfig:
    def test_the_default_is_info(self):
        assert load_config({}) == Config(log_level=logging.INFO)

    @pytest.mark.parametrize(
        "name, level",
        [
            ("DEBUG", logging.DEBUG),
            ("INFO", logging.INFO),
            ("WARNING", logging.WARNING),
            ("ERROR", logging.ERROR),
            ("CRITICAL", logging.CRITICAL),
        ],
    )
    def test_each_level_name_is_understood(self, name, level):
        assert load_config({"LANGSCOPE_LOGLEVEL": name}).log_level == level

    @pytest.mark.parametrize("name", ["debug", "Debug", " debug ", "dEbUg"])
    def test_case_and_surrounding_spaces_do_not_matter(self, name):
        assert load_config({"LANGSCOPE_LOGLEVEL": name}).log_level == logging.DEBUG

    @pytest.mark.parametrize("name", ["", "LOUD", "10", "root", "Level 10", "getLogger"])
    def test_anything_else_means_info(self, name):
        assert load_config({"LANGSCOPE_LOGLEVEL": name}).log_level == logging.INFO

    @pytest.mark.parametrize("name", ["BASIC_FORMAT", "basic_format"])
    def test_a_name_that_only_exists_as_a_logging_attribute_means_info(self, name):
        # logging.BASIC_FORMAT is a format string. It used to be passed on as the level, and
        # logging.basicConfig then failed.
        assert load_config({"LANGSCOPE_LOGLEVEL": name}).log_level == logging.INFO

    def test_other_variables_are_ignored(self):
        assert load_config({"PATH": "x", "LOGLEVEL": "DEBUG"}).log_level == logging.INFO

    def test_the_process_environment_is_used_by_default(self, monkeypatch):
        monkeypatch.setenv("LANGSCOPE_LOGLEVEL", "WARNING")
        assert load_config().log_level == logging.WARNING

    def test_the_config_cannot_be_changed(self):
        with pytest.raises(AttributeError):
            Config().log_level = logging.DEBUG


class TestConfigureLogging:
    def test_it_sets_the_level_and_the_format(self, monkeypatch):
        calls = []
        monkeypatch.setattr(logging, "basicConfig", lambda **kw: calls.append(kw))
        configure_logging(Config(log_level=logging.WARNING))
        assert calls == [{"level": logging.WARNING, "format": LOG_FORMAT}]

    def test_the_format_names_time_level_logger_and_message(self):
        assert LOG_FORMAT == "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
