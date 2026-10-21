"""The usage guide must keep up with the changelog.

Every commit on main becomes an alpha, and the alpha workflow runs the whole
suite first, so this test is what makes updating the guide a prerequisite of
a complete alpha. How to fix a failure: docs/user-guide/PROCESSO.md.
"""

import sys
from pathlib import Path

import pytest

GUIDE_DIR = Path(__file__).resolve().parents[1] / "docs" / "user-guide"
REPO = GUIDE_DIR.parents[1]

sys.path.insert(0, str(GUIDE_DIR))
import verificar_guia  # noqa: E402


@pytest.fixture(autouse=True)
def _quiet(capsys):
    yield
    capsys.readouterr()


def test_every_changelog_item_of_the_current_version_is_in_the_guide():
    missing = verificar_guia.itens_sem_registro(REPO)
    assert not missing, (
        "The changelog has items the usage guide does not cover yet. Describe "
        "each one in docs/user-guide/secoes/<every language>/, then run "
        "`python docs/user-guide/verificar_guia.py . --registrar` and "
        "`python docs/user-guide/gerar_guia.py .`:\n  - " + "\n  - ".join(missing)
    )


def test_the_shipped_guides_are_what_the_generator_produces():
    stale = verificar_guia.guias_desatualizados(REPO)
    assert not stale, (
        f"client/data/help_guide is out of date for {stale}. "
        "Run `python docs/user-guide/gerar_guia.py .` and commit the result."
    )


def test_all_languages_have_the_same_sections():
    assert verificar_guia.divergencias_entre_idiomas() == []
