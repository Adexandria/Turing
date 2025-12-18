import pandas as pd
import pytest

from turing.features import (
    FeatureEngineer,
    FeaturePipelineConfig,
    TextProcessor,
)

# --- Fixtures ---


@pytest.fixture(scope="module")
def full_config():
    """Returns a config with stopwords and lemmatization enabled."""
    return FeaturePipelineConfig(
        use_stopwords=True,
        use_lemmatization=True,
        use_combo_feature=False,
        max_features=5000,
        min_comment_length=10,
        max_comment_length=500,
        enable_augmentation=False,
        custom_tags="test",
    )


@pytest.fixture(scope="module")
def basic_config():
    """Returns a config with all extra steps disabled."""
    return FeaturePipelineConfig(
        use_stopwords=False,
        use_lemmatization=False,
        use_combo_feature=False,
        max_features=100,
        min_comment_length=5,
        max_comment_length=200,
        enable_augmentation=False,
    )


@pytest.fixture(scope="module")
def full_processor(full_config):
    """A TextProcessor with all steps enabled."""
    return TextProcessor(config=full_config, language="english")


@pytest.fixture(scope="module")
def basic_processor(basic_config):
    """A TextProcessor with only basic cleaning (lowercase, punctuation)."""
    return TextProcessor(config=basic_config, language="english")


# --- Tests ---


class TestFeaturePipelineConfig:
    def test_config_id_generation(self, full_config, basic_config):
        """Tests that the readable ID is generated correctly."""
        assert full_config.hash_id == "clean-k5000-test"
        assert basic_config.hash_id == "clean-k100"

    def test_config_attributes(self, full_config):
        """Tests that attributes are set correctly."""
        assert full_config.use_stopwords is True
        assert full_config.use_lemmatization is True
        assert full_config.max_features == 5000


class TestTextProcessor:
    def test_clean_text_basic(self, basic_processor):
        """Tests lowercase and punctuation removal."""
        text = "This is a TEST... with punctuation!!"
        expected = "this is a test with punctuation"
        assert basic_processor.clean_text(text) == expected

    def test_clean_text_stopwords(self, full_processor, basic_processor):
        """Tests stopword removal logic."""
        text = "this is a test with a stopword"

        # With stopwords enabled
        expected_full = "test stopword"
        assert full_processor.clean_text(text) == expected_full

        # With stopwords disabled
        expected_basic = "this is a test with a stopword"
        assert basic_processor.clean_text(text) == expected_basic

    def test_clean_text_lemmatization(self, full_processor, basic_processor):
        """Tests lemmatization logic."""
        text = "running tests while dogs are barking"

        # With lemmatization enabled
        expected_full = "running test dog barking"  # 'are' and 'while' are stopwords
        assert full_processor.clean_text(text) == expected_full

        # With lemmatization disabled
        expected_basic = "running tests while dogs are barking"
        assert basic_processor.clean_text(text) == expected_basic

    def test_clean_text_handles_none(self, basic_processor):
        """Tests that it doesn't crash on None or pd.NA."""
        assert basic_processor.clean_text(None) == ""
        assert basic_processor.clean_text(pd.NA) == ""


class TestFeatureEngineer:
    def test_extract_numeric_features(self, basic_config):
        """Tests that extract_features_for_check adds metadata features."""
        fe = FeatureEngineer(config=basic_config)
        data = {"comment_sentence": ["This is short.", "This one is a bit longer.", ""]}
        df = pd.DataFrame(data)
        df_out = fe.extract_features_for_check(df)

        assert "f_length" in df_out.columns
        assert "f_word_count" in df_out.columns
        assert "f_starts_verb" in df_out.columns
        assert "text_hash" in df_out.columns

        assert df_out["f_length"].tolist() == [14, 25, 0]
        assert df_out["f_word_count"].tolist() == [3, 6, 0]
