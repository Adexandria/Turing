# These tests check that adding or removing keywords logically changes the prediction


def test_java_directional_add_deprecation(java_model, get_predicted_labels):
    """Tests that adding '@deprecated' ADDs the 'deprecation' label"""
    # Base comment should be a 'Pointer' due to the link
    base_comment = "/** Use {@link #newUserMethod()} instead. */"
    # Perturbed comment adds a keyword
    pert_comment = "/** @deprecated Use {@link #newUserMethod()} instead. */"

    preds_base = get_predicted_labels(java_model, base_comment, "java")
    preds_pert = get_predicted_labels(java_model, pert_comment, "java")

    # The base comment should not have 'deprecation'
    assert "deprecation" not in preds_base
    # The perturbed comment must have 'deprecation'
    assert "deprecation" in preds_pert
    # The original 'Pointer' label should still be there
    assert "Pointer" in preds_base
    assert "Pointer" in preds_pert


def test_python_directional_remove_todo(python_model, get_predicted_labels):
    """Tests that removing 'TODO' REMOVES the 'DevelopmentNotes' labe."""
    base_comment = "# TODO: Refactor this entire block."
    pert_comment = "# Refactor this entire block."

    preds_base = get_predicted_labels(python_model, base_comment, "python")
    preds_pert = get_predicted_labels(python_model, pert_comment, "python")

    # The base comment must have 'DevelopmentNotes'
    assert "DevelopmentNotes" in preds_base
    # The perturbed comment must not have 'DevelopmentNotes'
    assert "DevelopmentNotes" not in preds_pert


def test_pharo_directional_add_responsibility(pharo_model, get_predicted_labels):
    """Tests that adding 'i am responsible for' adds the 'Responsibilities' label"""
    base_comment = '"i am a simple arrow"'
    pert_comment = '"i am a simple arrow. i am responsible for drawing."'

    preds_base = get_predicted_labels(pharo_model, base_comment, "pharo")
    preds_pert = get_predicted_labels(pharo_model, pert_comment, "pharo")

    # base comment should have 'Intent'
    assert "Intent" in preds_base
    # base comment should not have 'Responsibilities'
    assert "Responsibilities" not in preds_base
    # perturbed comment must have 'Responsibilities'
    assert "Responsibilities" in preds_pert
    # original 'Intent' label should still be there
    assert "Intent" in preds_pert


def test_java_directional_contrast_rational(java_model, get_predicted_labels):
    """
    Tests that adding a design rationale adds the 'rational' label
    """
    # Base comment is a simple summary
    base_comment = "/** Returns the user ID. */"
    # Perturbed comment adds a design rationale
    pert_comment = "/** Returns the user ID. This is cached for performance. */"

    preds_base = get_predicted_labels(java_model, base_comment, "java")
    preds_pert = get_predicted_labels(java_model, pert_comment, "java")

    # Base comment should be a 'summary'
    assert "summary" in preds_base
    # Base comment should not have 'rational'
    assert "rational" not in preds_base
    # Perturbed comment must now have 'rational'
    assert "rational" in preds_pert
    # Perturbed comment should ideally still be a 'summary'
    assert "summary" in preds_pert


def test_python_directional_contrast_todo(python_model, get_predicted_labels):
    """
    Tests that adding a "TODO" clause adds the 'DevelopmentNotes' label
    """
    # Base comment is a simple summary
    base_comment = "Fetches the user profile."
    # Perturbed comment adds a development note
    pert_comment = "Fetches the user profile. TODO: This is deprecated."

    preds_base = get_predicted_labels(python_model, base_comment, "python")
    preds_pert = get_predicted_labels(python_model, pert_comment, "python")

    # Base comment should be a 'Summary'
    assert "Summary" in preds_base
    # Base comment should not have 'DevelopmentNotes'
    assert "DevelopmentNotes" not in preds_base
    # Perturbed comment must now have 'DevelopmentNotes'
    assert "DevelopmentNotes" in preds_pert
    # Perturbed comment should ideally still be a 'Summary'
    assert "Summary" in preds_pert


def test_pharo_directional_contrast_collaborators(pharo_model, get_predicted_labels):
    """
    Tests that adding a 'but i work with' clause adds the 'Collaborators' label
    """
    # Base comment is a simple intent
    base_comment = '"i am a simple arrow like arrowhead."'
    pert_comment = '"i am a simple arrow, but i work with BlSpace to position."'

    preds_base = get_predicted_labels(pharo_model, base_comment, "pharo")
    preds_pert = get_predicted_labels(pharo_model, pert_comment, "pharo")

    # Base comment should be 'Intent'
    assert "Intent" in preds_base
    # Base comment should not  have 'Collaborators'
    assert "Collaborators" not in preds_base
    # Perturbed comment must now have 'Collaborators'
    assert "Collaborators" in preds_pert
    # Perturbed comment should ideally still have 'Intent'
    assert "Intent" in preds_pert


def test_java_directional_shift_summary_to_expand(java_model, get_predicted_labels):
    """
    Tests that replacing a simple 'summary' with an 'Expand' implementation note
    shifts the primary classification from 'summary' to 'Expand'
    """
    # Base comment is a simple summary
    base_comment = "/** Returns the user ID. */"
    # Perturbed comment shifts the focus entirely to implementation details
    pert_comment = "/** Implementation Note: This delegates to the old system. */"

    preds_base = get_predicted_labels(java_model, base_comment, "java")
    preds_pert = get_predicted_labels(java_model, pert_comment, "java")

    # Base comment must have 'summary'
    assert "summary" in preds_base
    # Perturbed comment must not have 'summary'
    assert "summary" not in preds_pert
    #  Perturbed comment must now have 'Expand'
    assert "Expand" in preds_pert


def test_python_directional_shift_summary_to_devnotes(python_model, get_predicted_labels):
    """
    Tests that replacing a 'Summary' with a critical development note (deprecated)
    shifts the classification from 'Summary' to 'DevelopmentNotes'
    """
    print(f"\n[DEBUG] Oggetto modello Python: {python_model}, Lingua: {python_model.language}")
    # Base comment is a clear Summary
    base_comment = "Fetches the user profile."
    # Perturbed comment shifts the focus entirely to a note about future work
    pert_comment = "DEPRECATED: This function is scheduled for removal in v2.0."

    preds_base = get_predicted_labels(python_model, base_comment, "python")
    preds_pert = get_predicted_labels(python_model, pert_comment, "python")

    # Base comment must have 'Summary'
    assert "Summary" in preds_base
    # Perturbed comment must not have 'Summary'
    assert "Summary" not in preds_pert
    # Perturbed comment must now have 'DevelopmentNotes'
    assert "DevelopmentNotes" in preds_pert


def test_pharo_directional_shift_to_example(pharo_model, get_predicted_labels):
    """
    Tests that changing a comment from a 'Responsibility' statement to an
    explicit 'Example' statement shifts the primary classification
    """
    # Base comment is a clear 'Responsibilities'
    base_comment = '"i provide a data structure independent api"'
    # Perturbed comment replaces the responsibility claim with an explicit example pattern
    pert_comment = '"[Example] run the data structure independent api."'

    preds_base = get_predicted_labels(pharo_model, base_comment, "pharo")
    preds_pert = get_predicted_labels(pharo_model, pert_comment, "pharo")

    # Base comment msut have Responsibilities
    assert "Responsibilities" in preds_base
    # Base comment should not have Example
    assert "Example" not in preds_base
    # Perturbed comment must now have Example
    assert "Example" in preds_pert
    # Perturbed comment should not have Responsibilities
    assert "Responsibilities" not in preds_pert
