import pytest

# These tests check that "noise" (like capitalization or punctuation) does not change the prediction


@pytest.mark.parametrize(
    "comment",
    [
        ":param user_id: The ID of the user.",  # Base
        ":PARAM USER_ID: THE ID OF THE USER.",  # Uppercase
        "  :param user_id: The ID of the  user .  ",  # Whitespace
        ":param user_id: The ID of the user!!!",  # Punctuation
    ],
)
def test_python_invariance_parameters(python_model, comment, get_predicted_labels):
    """Tests that noise  doesn't break ':param' detection."""
    expected = {"Parameters"}
    preds = get_predicted_labels(python_model, comment, "python")
    assert preds == expected


def test_java_invariance_deprecation(java_model, get_predicted_labels):
    """Tests that noise doesn't break '@deprecated' detection"""
    base_comment = "/** @deprecated Use newUserMethod() */"
    pert_comment = "/** @DEPRECATED... Use newUserMethod()!!! */"

    preds_base = get_predicted_labels(java_model, base_comment, "java")
    preds_pert = get_predicted_labels(java_model, pert_comment, "java")

    assert {"deprecation"} <= preds_base
    assert preds_base == preds_pert


def test_python_invariance_summary(python_model, get_predicted_labels):
    """Tests that noise doesn't break a simple 'Summary' detection"""

    base_comment = "a service specific account of type bar."
    expected = {"Summary"}

    # Perturbations
    variants = [
        base_comment,
        "A SERVICE SPECIFIC ACCOUNT OF TYPE BAR.",
        "  a service specific account of type bar.  ",
        "a service specific account of type bar!!!",
    ]

    for comment in variants:
        preds = get_predicted_labels(python_model, comment, "python")
        assert preds == expected


def test_pharo_invariance_intent(pharo_model, get_predicted_labels):
    """Tests that noise doesn't break Pharo's 'Intent' detection"""

    base_comment = '"i am a simple arrow like arrowhead."'
    expected = {"Intent"}

    # Perturbations
    variants = [
        base_comment,
        '"I AM A SIMPLE ARROW LIKE ARROWHEAD."',
        '  "i am a simple arrow like arrowhead."  ',
        '"i am a simple arrow like arrowhead !!"',  #
    ]

    for comment in variants:
        preds = get_predicted_labels(pharo_model, comment, "pharo")
        assert preds == expected


def test_python_invariance_typos_parameters(python_model, get_predicted_labels):
    """
    Tests typo tolerance

    """

    # Define the single expected outcome
    expected_labels = {"Parameters"}

    # Define the base case and all its variants (with typos)
    variants = [
        ":param user_id: The ID of the user.",
        ":paramater user_id: The ID of the user.",
        ":pram user_id: The ID of teh user.",
    ]

    # Loop through all variants and assert they all produce the *exact* expected outcome
    for comment in variants:
        preds = get_predicted_labels(python_model, comment, "python")
        assert preds == expected_labels


def test_java_invariance_semantic_summary(java_model, get_predicted_labels):
    """
    Tests semantic invariance

    """

    # Get the prediction for the base comment
    base_comment = "/** Returns the user ID. */"
    base_preds = get_predicted_labels(java_model, base_comment, "java")

    # Define semantic paraphrases of the base comment
    variants = [
        base_comment,
        "/** Gets the user ID. */",
        "/** Fetches the ID for the user. */",
        "/** A method to return the user's ID. */",
    ]

    # Check that the base prediction is valid (summary)
    assert "summary" in base_preds

    for comment in variants:
        preds = get_predicted_labels(java_model, comment, "java")
        assert preds == base_preds
