import ast
import hashlib
from pathlib import Path
import random
import re
from typing import List, Tuple

import nltk
from nltk.corpus import stopwords, wordnet
from nltk.stem import PorterStemmer, WordNetLemmatizer
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest, chi2
import typer

from turing.config import (
    INTERIM_DATA_DIR,
    LABEL_COLUMN,
    LANGS,
)
from turing.dataset import DatasetManager

# --- NLTK Resource Check ---
REQUIRED_NLTK_PACKAGES = [
    "stopwords",
    "wordnet",
    "omw-1.4",
    "averaged_perceptron_tagger",
    "punkt",
]
for package in REQUIRED_NLTK_PACKAGES:
    try:
        nltk.data.find(f"corpora/{package}")
    except LookupError:
        try:
            nltk.download(package, quiet=True)
        except Exception:
            pass

app = typer.Typer()


# --- CONFIGURATION CLASS ---
class FeaturePipelineConfig:
    """
    Configuration holder for the pipeline. Generates a unique ID based on parameters
    to version the output directories.
    """

    def __init__(
        self,
        use_stopwords: bool,
        use_lemmatization: bool,
        use_combo_feature: bool,
        max_features: int,
        min_comment_length: int,
        max_comment_length: int,
        enable_augmentation: bool,
        custom_tags: str = "base",
    ):
        self.use_stopwords = use_stopwords
        self.use_lemmatization = use_lemmatization
        self.use_combo_feature = use_combo_feature
        self.max_features = max_features
        self.min_comment_length = min_comment_length
        self.max_comment_length = max_comment_length
        self.enable_augmentation = enable_augmentation
        self.custom_tags = custom_tags
        self.hash_id = self._generate_readable_id()

    def _generate_readable_id(self) -> str:
        tags = ["clean"]
        if self.enable_augmentation:
            tags.append("aug-soft")
        tags.append(f"k{self.max_features}")
        if self.custom_tags != "base":
            tags.append(self.custom_tags)
        return "-".join(tags)


# --- TEXT UTILITIES ---
class TextCanonicalizer:
    """
    Reduces text to a 'canonical' form (stemmed, lowercase)
    to detect semantic duplicates.
    preserves javadoc tags to distinguish usage (@return) from summary (Returns).
    """

    def __init__(self):
        self.stemmer = PorterStemmer()
        self.stop_words = set(stopwords.words("english"))
        # Code keywords are preserved as they carry semantic weight
        self.code_keywords = {
            "return",
            "true",
            "false",
            "null",
            "if",
            "else",
            "void",
            "int",
            "boolean",
            "param", 
            "throws",
            "exception",
        }

    def to_canonical(self, text: str) -> str:
        if pd.isna(text):
            return ""
        text = str(text).lower()
        text = re.sub(r"[^a-z0-9\s@]", " ", text)
        
        words = text.split()
        canonical_words = []
        
        for w in words:
            # If the word starts with @ (e.g., @return), keep it as is
            if w.startswith("@"):
                canonical_words.append(w)
                continue

            if w in self.stop_words and w not in self.code_keywords:
                continue
            
            stemmed = self.stemmer.stem(w)
            canonical_words.append(stemmed)
            
        return " ".join(canonical_words).strip()


class TextProcessor:
    """
    Standard text cleaning logic for final feature extraction (TF-IDF).
    """

    def __init__(self, config: FeaturePipelineConfig, language: str = "english"):
        self.config = config
        self.stop_words = set(stopwords.words(language))
        self.lemmatizer = WordNetLemmatizer()

    def clean_text(self, text: str) -> str:
        if pd.isna(text):
            return ""
        text = str(text).lower()
        # Remove heavy code markers but keep text structure
        text = re.sub(r"(^\s*//+|^\s*/\*+|\*/$)", "", text)
        # Keep only alpha characters for NLP model (plus pipe for combo)
        text = re.sub(r"[^a-z\s|]", " ", text)
        tokens = text.split()
        if self.config.use_stopwords:
            tokens = [w for w in tokens if w not in self.stop_words]
        if self.config.use_lemmatization:
            tokens = [self.lemmatizer.lemmatize(w) for w in tokens]
        return " ".join(tokens)


# --- AUGMENTATION ---
class SafeAugmenter:
    """
    protects reserved keywords from synonym replacement.
    """

    def __init__(self, aug_prob=0.3):
        self.aug_prob = aug_prob
        self.protected_words = {
            "return",
            "public",
            "private",
            "void",
            "class",
            "static",
            "final",
            "if",
            "else",
            "for",
            "while",
            "try",
            "catch",
            "import",
            "package",
            "null",
            "true",
            "false",
            "self",
            "def",
            "todo",
            "fixme",
            "param",
            "throw",
        }

    def get_synonyms(self, word):
        synonyms = set()
        for syn in wordnet.synsets(word):
            for lemma in syn.lemmas():
                name = lemma.name().replace("_", " ")
                if name.isalpha() and name.lower() != word.lower():
                    synonyms.add(name)
        return list(synonyms)

    def augment(self, text: str) -> str:
        if pd.isna(text) or not text:
            return ""
        words = text.split()
        if len(words) < 2:
            return text
        new_words = []
        for word in words:
            word_lower = word.lower()

            if word_lower in self.protected_words:
                new_words.append(word)
                continue

            # Random Case Injection (Noise)
            if random.random() < 0.1:
                if word[0].isupper():
                    new_words.append(word.lower())
                else:
                    new_words.append(word.capitalize())
                continue

            # Synonym Replacement
            if random.random() < self.aug_prob and len(word) > 3:
                syns = self.get_synonyms(word_lower)
                if syns:
                    replacement = random.choice(syns)
                    if word[0].isupper():
                        replacement = replacement.capitalize()
                    new_words.append(replacement)
                else:
                    new_words.append(word)
            else:
                new_words.append(word)
        return " ".join(new_words)

    def apply_balancing(
        self, df: pd.DataFrame, min_samples: int = 100
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Generates synthetic data for minority classes.
        Returns: (Balanced DataFrame, Report DataFrame)
        """
        df["temp_label_str"] = df[LABEL_COLUMN].astype(str)
        counts = df["temp_label_str"].value_counts()
        print(
            f"\n   [Balance Check - PRE] Min class size: {counts.min()} | Max: {counts.max()}"
        )

        existing_sentences = set(df["comment_sentence"].str.strip())
        new_rows = []
        report_rows = []

        for label_str, count in counts.items():
            if count < min_samples:
                needed = min_samples - count
                class_subset = df[df["temp_label_str"] == label_str]
                if class_subset.empty:
                    continue

                samples = class_subset["comment_sentence"].tolist()
                orig_label = class_subset[LABEL_COLUMN].iloc[0]

                # Propagate 'combo' if present
                orig_combo = None
                if "combo" in class_subset.columns:
                    orig_combo = class_subset["combo"].iloc[0]

                generated = 0
                attempts = 0
                # Cap attempts to avoid infinite loops if vocabulary is too small
                while generated < needed and attempts < needed * 5:
                    attempts += 1
                    src = random.choice(samples)
                    aug_txt = self.augment(src).strip()

                    # Ensure Global Uniqueness
                    if aug_txt and aug_txt not in existing_sentences:
                        row = {
                            "comment_sentence": aug_txt,
                            LABEL_COLUMN: orig_label,
                            "partition": "train_aug",
                            "index": -1,  # Placeholder
                        }
                        if orig_combo:
                            row["combo"] = orig_combo

                        new_rows.append(row)
                        report_rows.append(
                            {
                                "original_text": src,
                                "augmented_text": aug_txt,
                                "label": label_str,
                                "reason": f"Class has {count} samples (Target {min_samples})",
                            }
                        )
                        existing_sentences.add(aug_txt)
                        generated += 1

        df = df.drop(columns=["temp_label_str"])
        df_report = pd.DataFrame(report_rows)

        if new_rows:
            augmented_df = pd.concat([df, pd.DataFrame(new_rows)], ignore_index=True)
            augmented_df["index"] = range(len(augmented_df))

            temp_counts = augmented_df[LABEL_COLUMN].astype(str).value_counts()
            print(
                f"   [Balance Check - POST] Min class size: {temp_counts.min()} | Max: {temp_counts.max()}"
            )
            return augmented_df, df_report

        return df, df_report


# --- CLEANING LOGIC ---
def clean_training_data_smart(
    df: pd.DataFrame, min_len: int, max_len: int, language: str = "english"
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Performs 'Smart Cleaning' on the Training Set with language-specific heuristics.
    """
    canon = TextCanonicalizer()
    dropped_rows = []

    print(f"   [Clean] Computing heuristics (Language: {language})...")
    df["canon_key"] = df["comment_sentence"].apply(canon.to_canonical)

    # 1. Token Length Filter
    def count_code_tokens(text):
        return len([t for t in re.split(r"[^a-zA-Z0-9]+", str(text)) if t])

    df["temp_token_len"] = df["comment_sentence"].apply(count_code_tokens)

   
    MIN_ALPHA_CHARS = 6
    MAX_SYMBOL_RATIO = 0.50

    # 2. Heuristic Filters (Tiny/Huge/Code)
    def get_heuristics(text):
        s = str(text).strip()
        char_len = len(s)
        if char_len == 0:
            return False, False, 1.0
        
        alpha_len = sum(1 for c in s if c.isalpha())
        
        non_alnum_chars = sum(1 for c in s if not c.isalnum() and not c.isspace())
        symbol_ratio = non_alnum_chars / char_len if char_len > 0 else 0

        is_tiny = alpha_len < MIN_ALPHA_CHARS
        is_huge = char_len > 800
        is_code = symbol_ratio > MAX_SYMBOL_RATIO
        
        return is_tiny, is_huge, is_code

    heuristics = df["comment_sentence"].apply(get_heuristics)
    df["is_tiny"] = [x[0] for x in heuristics]
    df["is_huge"] = [x[1] for x in heuristics]
    df["symbol_ratio"] = [x[2] for x in heuristics] 
    
    
    df["is_code"] = df["symbol_ratio"] > 0.50

    mask_keep = (
        (df["temp_token_len"] >= min_len)
        & (df["temp_token_len"] <= max_len)
        & (~df["is_tiny"])
        & (~df["is_huge"])
        & (~df["is_code"])
    )

    df_dropped_qual = df[~mask_keep].copy()
    if not df_dropped_qual.empty:
        def reason(row):
            if row["is_tiny"]:
                return f"Too Tiny (<{MIN_ALPHA_CHARS} alpha)"
            if row["is_huge"]:
                return "Too Huge (>800 chars)"
            if row["is_code"]:
                return f"Pure Code (>{int(MAX_SYMBOL_RATIO*100)}% symbols)"
            return f"Token Count ({row['temp_token_len']})"

        df_dropped_qual["drop_reason"] = df_dropped_qual.apply(reason, axis=1)
        dropped_rows.append(df_dropped_qual)

    df = df[mask_keep].copy()

    # 3. Semantic Conflicts (Ambiguity)
    df["label_s"] = df[LABEL_COLUMN].astype(str)
    conflict_counts = df.groupby("canon_key")["label_s"].nunique()
    conflicting_keys = conflict_counts[conflict_counts > 1].index

    mask_conflicts = df["canon_key"].isin(conflicting_keys)
    df_dropped_conflicts = df[mask_conflicts].copy()
    if not df_dropped_conflicts.empty:
        df_dropped_conflicts["drop_reason"] = "Semantic Conflict"
        dropped_rows.append(df_dropped_conflicts)

    df = df[~mask_conflicts].copy()

    # 4. Exact Duplicates
    mask_dupes = df.duplicated(subset=["comment_sentence"], keep="first")
    df_dropped_dupes = df[mask_dupes].copy()
    if not df_dropped_dupes.empty:
        df_dropped_dupes["drop_reason"] = "Exact Duplicate"
        dropped_rows.append(df_dropped_dupes)

    df = df[~mask_dupes].copy()

    # Cleanup columns
    cols_to_drop = [
        "canon_key",
        "label_s",
        "temp_token_len",
        "is_tiny",
        "is_huge",
        "is_code",
        "symbol_ratio"
    ]
    df = df.drop(columns=cols_to_drop, errors="ignore")

    if dropped_rows:
        df_report = pd.concat(dropped_rows, ignore_index=True)
        cols_rep = ["index", "comment_sentence", LABEL_COLUMN, "drop_reason"]
        final_cols = [c for c in cols_rep if c in df_report.columns]
        df_report = df_report[final_cols]
    else:
        df_report = pd.DataFrame(columns=["index", "comment_sentence", "drop_reason"])

    print(f"   [Clean] Removed {len(df_report)} rows. Final: {len(df)}.")
    return df, df_report

# --- FEATURE ENGINEERING ---
class FeatureEngineer:
    def __init__(self, config: FeaturePipelineConfig):
        self.config = config
        self.processor = TextProcessor(config=config)
        self.tfidf_vectorizer = TfidfVectorizer(max_features=config.max_features)

    def extract_features_for_check(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extracts metadata features for analysis."""

        def analyze(text):
            s = str(text)
            words = s.split()
            n_words = len(words)
            if n_words == 0:
                return 0, 0, 0
            first_word = words[0].lower()
            starts_verb = (
                1
                if first_word.endswith("s")
                or first_word.startswith("get")
                or first_word.startswith("set")
                else 0
            )
            return (len(s), n_words, starts_verb)

        metrics = df["comment_sentence"].apply(analyze)
        df["f_length"] = [x[0] for x in metrics]
        df["f_word_count"] = [x[1] for x in metrics]
        df["f_starts_verb"] = [x[2] for x in metrics]
        # Calculate MD5 hash for efficient exact duplicate detection in Deepchecks
        df["text_hash"] = df["comment_sentence"].apply(
            lambda x: hashlib.md5(str(x).encode()).hexdigest()
        )
        return df

    def vectorize_and_select(self, df_train, df_test):
        def clean_fn(x):
            return re.sub(r"[^a-zA-Z\s]", "", str(x).lower())

        X_train = self.tfidf_vectorizer.fit_transform(
            df_train["comment_sentence"].apply(clean_fn)
        )
        y_train = np.stack(df_train[LABEL_COLUMN].values)

        # Handling multi-label for Chi2 (using sum or max)
        y_train_sum = (
            y_train.sum(axis=1) if len(y_train.shape) > 1 else y_train
        )
        selector = SelectKBest(
            chi2, k=min(self.config.max_features, X_train.shape[1])
        )
        X_train = selector.fit_transform(X_train, y_train_sum)

        X_test = self.tfidf_vectorizer.transform(
            df_test["comment_sentence"].apply(clean_fn)
        )
        X_test = selector.transform(X_test)

        vocab = [
            self.tfidf_vectorizer.get_feature_names_out()[i]
            for i in selector.get_support(indices=True)
        ]
        return X_train, X_test, vocab


# --- MAIN EXECUTION ---
def main(
    feature_dir: Path = typer.Option(
        INTERIM_DATA_DIR / "features", help="Output dir."
    ),
    reports_root: Path = typer.Option(
        Path("reports/data"), help="Reports root."
    ),
    max_features: int = typer.Option(5000),
    min_comment_length: int = typer.Option(
        2, help="Remove comments shorter than chars."
    ),
    max_comment_length: int = typer.Option(300),
    augment: bool = typer.Option(False, "--augment", help="Enable augmentation."),
    balance_threshold: int = typer.Option(100, help="Min samples per class."),
    run_vectorization: bool = typer.Option(False, "--run-vectorization"),
    run_nlp_check: bool = typer.Option(
        True, "--run-nlp", help="Run Deepchecks NLP suite."
    ),
    custom_tags: str = typer.Option("base", help="Custom tags."),
    save_full_csv: bool = typer.Option(False, "--save-full-csv"),
    languages: List[str] = typer.Option(LANGS, show_default=False),
):

    config = FeaturePipelineConfig(
        True,
        True,
        True,
        max_features,
        min_comment_length,
        max_comment_length,
        augment,
        custom_tags,
    )
    print(f"=== Pipeline ID: {config.hash_id} ===")

    dm = DatasetManager()
    full_dataset = dm.get_dataset()
    fe = FeatureEngineer(config)
    augmenter = SafeAugmenter()

    feat_output_dir = feature_dir / config.hash_id
    feat_output_dir.mkdir(parents=True, exist_ok=True)
    report_output_dir = reports_root / config.hash_id

    for lang in languages:
        print(f"\n{'='*30}\nPROCESSING LANGUAGE: {lang.upper()}\n{'='*30}")
        df_train = full_dataset[f"{lang}_train"].to_pandas()
        df_test = full_dataset[f"{lang}_test"].to_pandas()

        # Standardize Label Format
        for df in [df_train, df_test]:
            if isinstance(df[LABEL_COLUMN].iloc[0], str):
                df[LABEL_COLUMN] = (
                    df[LABEL_COLUMN]
                    .str.replace(r"\s+", ", ", regex=True)
                    .apply(ast.literal_eval)
                )

        lang_report_dir = report_output_dir / lang

        # CLEANING & AUGMENTATION
        print("\n   >>> Phase 2: Smart Cleaning & Augmentation")
        df_train, df_dropped = clean_training_data_smart(
            df_train, min_comment_length, max_comment_length, language=lang
        )

        if augment:
            print("   [Augment] Applying Soft Balancing...")
            df_train, df_aug_report = augmenter.apply_balancing(
                df_train, min_samples=balance_threshold
            )

        # FINAL PROCESSING & SAVING
        print("\n   >>> Phase 4: Final Processing & Save")
        df_train["comment_clean"] = df_train["comment_sentence"].apply(
            fe.processor.clean_text
        )
        df_test["comment_clean"] = df_test["comment_sentence"].apply(
            fe.processor.clean_text
        )

        if config.use_combo_feature:
            if "combo" in df_train.columns:
                df_train["combo_clean"] = df_train["combo"].apply(
                    fe.processor.clean_text
                )
            if "combo" in df_test.columns:
                df_test["combo_clean"] = df_test["combo"].apply(
                    fe.processor.clean_text
                )

        X_train, X_test, vocab = None, None, []
        if run_vectorization:
            print("   [Vectorization] TF-IDF & Chi2...")
            X_train, X_test, vocab = fe.vectorize_and_select(df_train, df_test)
        def format_label_robust(lbl):
            if hasattr(lbl, "tolist"): # Check if numpy array
                lbl = lbl.tolist()
            return str(lbl)

        df_train[LABEL_COLUMN] = df_train[LABEL_COLUMN].apply(format_label_robust)
        df_test[LABEL_COLUMN] = df_test[LABEL_COLUMN].apply(format_label_robust)

        cols_to_save = [
            "index",
            LABEL_COLUMN,
            "comment_sentence",
            "comment_clean",
        ]
        if "combo" in df_train.columns:
            cols_to_save.append("combo")
        if "combo_clean" in df_train.columns:
            cols_to_save.append("combo_clean")
        meta_cols = [c for c in df_train.columns if c.startswith("f_")]
        cols_to_save.extend(meta_cols)

        print(f"   [Save] Columns: {cols_to_save}")
        df_train[cols_to_save].to_csv(
            feat_output_dir / f"{lang}_train.csv", index=False
        )
        df_test[cols_to_save].to_csv(
            feat_output_dir / f"{lang}_test.csv", index=False
        )

        if run_vectorization and X_train is not None:
            from scipy.sparse import save_npz

            save_npz(feat_output_dir / f"{lang}_train_tfidf.npz", X_train)
            save_npz(feat_output_dir / f"{lang}_test_tfidf.npz", X_test)
            with open(
                feat_output_dir / f"{lang}_vocab.txt", "w", encoding="utf-8"
            ) as f:
                f.write("\n".join(vocab))

    print(f"\nAll Done. Reports in: {report_output_dir}")


if __name__ == "__main__":
    typer.run(main)