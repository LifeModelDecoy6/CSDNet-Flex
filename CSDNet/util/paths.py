from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = PACKAGE_ROOT / "config"
PMO_VOCAB_DIR = PACKAGE_ROOT / "exp" / "pmo" / "vocab"
DEFAULT_VALENCE_DICT = CONFIG_DIR / "valence_dict.json"
