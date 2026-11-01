"""Arabic packaging and branding checks; no GUI objects are constructed."""

from pathlib import Path
import importlib.util
import polib


ROOT = Path(__file__).resolve().parents[1]


def test_arabic_keeps_the_product_name_in_english():
    catalog = polib.pofile(str(ROOT / 'translations/ar/LC_MESSAGES/winzapp.po'))
    for entry in catalog:
        if not entry.obsolete:
            assert 'وينزاب' not in entry.msgstr, entry.msgctxt
            if 'WinZapp' in entry.msgid:
                assert 'WinZapp' in entry.msgstr, entry.msgctxt


def test_arabic_help_is_generated_and_links_to_the_shipped_english_guide(tmp_path):
    source = ROOT / 'docs/user-guide/arabic_help.py'
    spec = importlib.util.spec_from_file_location('arabic_help_for_test', source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    output = module.generate(tmp_path)
    text = output.read_text(encoding='utf-8')
    assert text == module.SOURCE.read_text(encoding='utf-8')
    assert '<html lang="ar" dir="rtl">' in text
    assert 'href="en-US.html"' in text
    assert (ROOT / 'client/data/help_guide/en-US.html').is_file()
