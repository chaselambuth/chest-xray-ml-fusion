import pandas as pd

from xray_fusion.data import assign_clean_label, clean_text, encode_text, split_semicolon_terms


# Nullable report text should normalize into safe lowercase strings.
def test_clean_text_handles_nulls_and_spacing():
    assert clean_text(None) == ""
    assert clean_text("  Mild Opacity  ") == "mild opacity"


# NIH metadata fields use semicolons and inconsistent whitespace.
def test_split_semicolon_terms_returns_clean_terms():
    assert split_semicolon_terms("Cardiomegaly ; Pleural Effusion; ") == [
        "cardiomegaly",
        "pleural effusion",
    ]


# Label assignment should map report metadata into the compact project labels.
def test_assign_clean_label_uses_report_terms():
    row = pd.Series(
        {
            "MeSH": "Cardiomegaly",
            "Problems": "",
            "impression": "enlarged cardiac silhouette",
        }
    )

    assert assign_clean_label(row) == "cardiomegaly"


# The simple text baseline should pad or truncate token ids to a fixed length.
def test_encode_text_returns_fixed_length_ids():
    vocab = {"<pad>": 0, "<unk>": 1, "clear": 2, "lungs": 3}

    assert encode_text("clear lungs unknown", vocab, max_len=5) == [2, 3, 1, 0, 0]
    assert encode_text("clear lungs unknown", vocab, max_len=2) == [2, 3]
