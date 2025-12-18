from pathlib import Path
import traceback
from typing import List

from deepchecks.tabular import Dataset, Suite
from deepchecks.tabular.checks import (
    ConflictingLabels,
    DataDuplicates,
    LabelDrift,
    OutlierSampleDetection,
    TrainTestSamplesMix,
)
import numpy as np
import pandas as pd

from turing.config import LABEL_COLUMN, LABELS_MAP

try:
    from deepchecks.nlp import TextData
    from deepchecks.nlp.checks import (
        PropertyDrift,
        TextEmbeddingsDrift,
    )

    NLP_AVAILABLE = True
except ImportError:
    NLP_AVAILABLE = False


def _encode_labels_for_validation(
    series: pd.Series, class_names: List[str]
) -> pd.Series:
    def encode(lbl):
        active_labels = []
        for idx, is_active in enumerate(lbl):
            if is_active:
                if idx < len(class_names):
                    active_labels.append(class_names[idx])
                else:
                    active_labels.append(f"Class_{idx}")
        if not active_labels:
            return "No_Label"
        return " & ".join(active_labels)

    return series.apply(encode)


def _calculate_code_specific_properties(text_series: List[str]) -> pd.DataFrame:
    props = []
    for text in text_series:
        s = str(text)
        length = len(s)
        non_alnum = sum(1 for c in s if not c.isalnum() and not c.isspace())
        props.append(
            {
                "Text_Length": length,
                "Symbol_Ratio": non_alnum / length if length > 0 else 0.0,
            }
        )
    return pd.DataFrame(props)


def _nuke_rogue_files():
    """
    delete .npy files
    """
    rogue_filenames = [
        "embeddings.npy"
     
    ]
    for fname in rogue_filenames:
        p = Path(fname) 
        if p.exists():
            try:
                p.unlink()
            except Exception:
                pass


def run_custom_deepchecks(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    output_dir: Path,
    stage: str,
    language: str,
):
    print(f"   [Deepchecks] Running Integrity Suite ({stage})...")
    output_dir.mkdir(parents=True, exist_ok=True)

    class_names = LABELS_MAP.get(language, [])
    cols = ["f_length", "f_word_count", "f_starts_verb", "text_hash"]

    for c in cols:
        if c not in df_train.columns:
            df_train[c] = 0
        if c not in df_test.columns:
            df_test[c] = 0

    train_ds_df = df_train[cols].copy()
    train_ds_df["target"] = _encode_labels_for_validation(
        df_train[LABEL_COLUMN], class_names
    )
    test_ds_df = df_test[cols].copy()
    test_ds_df["target"] = _encode_labels_for_validation(
        df_test[LABEL_COLUMN], class_names
    )

    cat_features = ["text_hash", "f_starts_verb"]
    train_ds = Dataset(train_ds_df, label="target", cat_features=cat_features)
    test_ds = Dataset(test_ds_df, label="target", cat_features=cat_features)

    check_conflicts = ConflictingLabels(columns=["text_hash"])
    if hasattr(check_conflicts, "add_condition_ratio_of_conflicting_labels_not_greater_than"):
        check_conflicts.add_condition_ratio_of_conflicting_labels_not_greater_than(0)
    else:
        check_conflicts.add_condition_ratio_of_conflicting_labels_less_or_equal(0)

    check_duplicates = DataDuplicates()
    if hasattr(check_duplicates, "add_condition_ratio_not_greater_than"):
        check_duplicates.add_condition_ratio_not_greater_than(0.05)
    else:
        check_duplicates.add_condition_ratio_less_or_equal(0.05)

    check_leakage = TrainTestSamplesMix(columns=["text_hash"])
    try:
        if hasattr(check_leakage, "add_condition_ratio_not_greater_than"):
            check_leakage.add_condition_ratio_not_greater_than(0)
    except Exception:
        pass

    check_outliers = OutlierSampleDetection()
    try:
        if hasattr(check_outliers, "add_condition_outlier_ratio_less_or_equal"):
            check_outliers.add_condition_outlier_ratio_less_or_equal(0.05)
    except Exception:
        pass

    custom_suite = Suite(
        "Code Quality & Integrity",
        check_conflicts,
        check_duplicates,
        check_leakage,
        LabelDrift(),
        check_outliers,
    )

    try:
        result = custom_suite.run(train_dataset=train_ds, test_dataset=test_ds)
        report_path = output_dir / f"1_Integrity_{stage}.html"
        result.save_as_html(str(report_path), as_widget=False)
        print(f"   [Deepchecks] Report Saved: {report_path}")
    except Exception as e:
        print(f"   [Deepchecks] Error: {e}")
        traceback.print_exc()


def run_targeted_nlp_checks(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    output_dir: Path,
    stage: str,
    language: str = "english",
):
    if not NLP_AVAILABLE:
        print("   [Skip] NLP Suite skipped (libs not installed).")
        return

    from deepchecks.nlp import Suite as NLPSuite

    print(f"   [NLP Check] Running Semantic Analysis ({stage})...")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Clean up any existing garbage before starting
    _nuke_rogue_files()

    DRIFT_THRESHOLD = 0.20
    PROP_THRESHOLD = 0.35
    SAMPLE_SIZE = 2000
    df_tr = (
        df_train.sample(n=SAMPLE_SIZE, random_state=42)
        if len(df_train) > SAMPLE_SIZE
        else df_train
    )
    df_te = (
        df_test.sample(n=SAMPLE_SIZE, random_state=42)
        if len(df_test) > SAMPLE_SIZE
        else df_test
    )

    try: # START MAIN TRY BLOCK
        y_tr = np.vstack(df_tr[LABEL_COLUMN].tolist())
        y_te = np.vstack(df_te[LABEL_COLUMN].tolist())

        train_ds = TextData(
            df_tr["comment_sentence"].tolist(),
            label=y_tr,
            task_type="text_classification",
        )
        test_ds = TextData(
            df_te["comment_sentence"].tolist(),
            label=y_te,
            task_type="text_classification",
        )

        print("   [NLP Check] Calculating custom code properties...")
        train_props = _calculate_code_specific_properties(
            df_tr["comment_sentence"].tolist()
        )
        test_props = _calculate_code_specific_properties(
            df_te["comment_sentence"].tolist()
        )

        train_ds.set_properties(train_props)
        test_ds.set_properties(test_props)

        # In-memory calculation only. 
        train_ds.calculate_builtin_embeddings()
        test_ds.calculate_builtin_embeddings()

        check_embeddings = TextEmbeddingsDrift()
        if hasattr(check_embeddings, "add_condition_drift_score_not_greater_than"):
            check_embeddings.add_condition_drift_score_not_greater_than(DRIFT_THRESHOLD)
        elif hasattr(check_embeddings, "add_condition_drift_score_less_than"):
            check_embeddings.add_condition_drift_score_less_than(DRIFT_THRESHOLD)

        check_len = PropertyDrift(custom_property_name="Text_Length")
        if hasattr(check_len, "add_condition_drift_score_not_greater_than"):
            check_len.add_condition_drift_score_not_greater_than(PROP_THRESHOLD)
        elif hasattr(check_len, "add_condition_drift_score_less_than"):
            check_len.add_condition_drift_score_less_than(PROP_THRESHOLD)

        check_sym = PropertyDrift(custom_property_name="Symbol_Ratio")
        if hasattr(check_sym, "add_condition_drift_score_not_greater_than"):
            check_sym.add_condition_drift_score_not_greater_than(PROP_THRESHOLD)
        elif hasattr(check_sym, "add_condition_drift_score_less_than"):
            check_sym.add_condition_drift_score_less_than(PROP_THRESHOLD)

        suite = NLPSuite(
            "Code Comment Semantic Analysis", 
            check_embeddings, 
            check_len, 
            check_sym
        )

        res = suite.run(train_ds, test_ds)
        
        report_path = output_dir / f"2_Semantic_{stage}.html"
        res.save_as_html(str(report_path), as_widget=False)
        print(f"   [NLP Check] Report saved: {report_path}")

        try:
            passed = res.get_passed_checks()
            n_passed = len(passed)
            n_total = len(res.results)
            print(f"   [NLP Result] {n_passed}/{n_total} checks passed.")
            
            if n_passed < n_total:
                print("   [NLP Warning] Failed Checks details:")
                for result in res.results:
                    if not result.passed_conditions():
                        print(f"     - {result.check.name}: {result.conditions_results[0].details}")
        except Exception:
            pass

    except Exception as e:
        print(f"   [NLP Check] Failed: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        _nuke_rogue_files()