"""Ship an Arabic introduction while the complete guide remains in English."""

from pathlib import Path


SOURCE = Path(__file__).parent / "arabic-introduction.html"


def generate(output_folder):
    output = Path(output_folder) / "ar.html"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(SOURCE.read_text(encoding="utf-8"), encoding="utf-8")
    return output
