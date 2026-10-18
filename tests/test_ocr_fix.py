from mmr.index import fix_ocr_spacing


def test_concatenated_ocr_words_are_split():
    out = fix_ocr_spacing("Teletexthasbecomeoutdated,duetothe popularity")
    assert "has become outdated" in out
    assert "due to the" in out


def test_known_words_untouched_and_short_unknown_words_kept():
    assert fix_ocr_spacing("Remote control popularity") == "Remote control popularity"
    out = fix_ocr_spacing("Teletext")
    assert out.startswith("Teletext")  # original kept so a "teletext" query still matches
