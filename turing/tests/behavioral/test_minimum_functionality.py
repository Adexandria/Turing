import pytest

# These tests check for basic, obvious classifications


@pytest.mark.parametrize(
    "comment, expected_labels",
    [
        ("test getfilestatus and related listing operations.", {"summary"}),
        ("/* @deprecated Use something else. */", {"deprecation"}),
        ("code source of this file http grepcode.com", {"Pointer"}),
        ("this is balanced if each pool is balanced.", {"rational"}),
        ("// For internal use only.", {"Ownership"}),
        ("this impl delegates to the old filesystem", {"Expand"}),
        ("/** Usage: new MyClass(arg1). */", {"usage"}),
    ],
)
def test_java_mft(java_model, comment, expected_labels, get_predicted_labels):
    preds = get_predicted_labels(java_model, comment, "java")
    assert preds == expected_labels


@pytest.mark.parametrize(
    "comment, expected_labels",
    [
        ("a service specific account of type bar.", {"Summary"}),
        (":param user_id: The ID of the user.", {"Parameters"}),
        ("# TODO: Refactor this entire block.", {"DevelopmentNotes"}),
        ("use this class if you want access to all of the mechanisms", {"Usage"}),
        ("# create a new list by filtering duplicates from the input", {"Expand"}),
    ],
)
def test_python_mft(python_model, comment, expected_labels, get_predicted_labels):
    preds = get_predicted_labels(python_model, comment, "python")
    assert preds == expected_labels


@pytest.mark.parametrize(
    "comment, expected_labels",
    [
        ("i am a simple arrow like arrowhead.", {"Intent"}),
        ("the example below shows how to create a simple element", {"Example"}),
        ("i provide a data structure independent api", {"Responsibilities"}),
        ("the cache is cleared after each test to ensure isolation.", {"Keyimplementationpoints"}),
        ("it is possible hovewer to customize a length fraction", {"Keymessages"}),
        ("collaborators: BlElement, BlSpace", {"Collaborators"}),
    ],
)
def test_pharo_mft(pharo_model, comment, expected_labels, get_predicted_labels):
    """Tests basic keyword-to-label mapping for Pharo (e.g., 'I am...')."""
    preds = get_predicted_labels(pharo_model, comment, "pharo")
    assert preds == expected_labels
