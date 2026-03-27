"""
Tests automatisés pour le pipeline Voxtral TTS + STT.

Ces tests vérifient que les modules se chargent correctement
et que les fonctions utilitaires fonctionnent.

Pour les tests end-to-end (nécessitant les modèles MLX),
utiliser directement les scripts src/tts_test.py, src/stt_test.py
et src/roundtrip_test.py.
"""

import pytest
from pathlib import Path


def test_output_dir_exists():
    """Le dossier output/ doit exister."""
    output = Path(__file__).resolve().parent.parent / "output"
    output.mkdir(parents=True, exist_ok=True)
    assert output.is_dir()


def test_levenshtein_similarity():
    """Test de la fonction de similarité utilisée dans le round-trip."""
    from src.roundtrip_test import levenshtein_similarity

    assert levenshtein_similarity("hello", "hello") == 1.0
    assert levenshtein_similarity("", "") == 1.0
    assert levenshtein_similarity("abc", "") == 0.0
    assert levenshtein_similarity("kitten", "sitting") == pytest.approx(0.571, abs=0.01)
    # Case insensitive
    assert levenshtein_similarity("Bonjour", "bonjour") == 1.0


def test_tts_module_importable():
    """Le module TTS doit être importable."""
    import src.tts_test as tts
    assert hasattr(tts, "load_tts_model")
    assert hasattr(tts, "generate_and_save")
    assert hasattr(tts, "MODEL_ID")


def test_stt_module_importable():
    """Le module STT doit être importable."""
    import src.stt_test as stt
    assert hasattr(stt, "load_stt_model")
    assert hasattr(stt, "transcribe")
    assert hasattr(stt, "MODEL_ID")


def test_roundtrip_module_importable():
    """Le module round-trip doit être importable."""
    import src.roundtrip_test as rt
    assert hasattr(rt, "roundtrip_test")
    assert hasattr(rt, "levenshtein_similarity")
