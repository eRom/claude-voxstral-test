"""
Test TTS (Text-to-Speech) avec Voxtral sur Apple Silicon.

Modèle : mlx-community/Voxtral-4B-TTS-2603-mlx-6bit (~3 GB RAM)
Lib    : mlx-audio

Tests couverts :
  1. Génération simple (FR)
  2. Génération multi-langues (FR, EN, ES)
  3. Clonage vocal depuis un fichier de référence
  4. Mesure de latence (time-to-first-audio)
"""

import argparse
import time
from pathlib import Path

import numpy as np
import soundfile as sf

MODEL_ID = "mlx-community/Voxtral-4B-TTS-2603-mlx-6bit"
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "output"


def load_tts_model(model_id: str = MODEL_ID):
    """Charge le modèle TTS MLX."""
    from mlx_audio.tts.utils import load_model

    print(f"[TTS] Chargement du modèle : {model_id}")
    t0 = time.perf_counter()
    model = load_model(model_id)
    dt = time.perf_counter() - t0
    print(f"[TTS] Modèle chargé en {dt:.1f}s")
    return model


def generate_and_save(model, text: str, language: str, output_path: Path,
                      voice_ref: str | None = None):
    """Génère l'audio et sauvegarde en WAV. Retourne la durée de génération."""
    print(f"[TTS] Génération : '{text[:60]}...' ({language})")

    kwargs = {"text": text, "language": language}
    if voice_ref:
        kwargs["voice_ref"] = voice_ref

    t0 = time.perf_counter()
    results = list(model.generate(**kwargs))
    dt = time.perf_counter() - t0

    audio = np.array(results[0].audio)
    sample_rate = getattr(results[0], "sample_rate", 24_000)
    sf.write(str(output_path), audio, sample_rate)

    duration_audio = len(audio) / sample_rate
    print(f"[TTS] -> {output_path.name} ({duration_audio:.1f}s audio, généré en {dt:.2f}s)")
    return dt


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_simple_fr(model):
    """Test 1 : Génération simple en français."""
    text = "Bonjour, je suis un test de synthèse vocale avec Voxtral sur Apple Silicon."
    output = OUTPUT_DIR / "tts_simple_fr.wav"
    dt = generate_and_save(model, text, "French", output)
    print(f"  -> Latence totale : {dt:.2f}s\n")
    return dt


def test_multi_langues(model):
    """Test 2 : Génération dans plusieurs langues."""
    phrases = [
        ("French", "La technologie vocale avance à grands pas."),
        ("English", "Voice technology is advancing rapidly."),
        ("Spanish", "La tecnología de voz avanza rápidamente."),
    ]
    results = {}
    for lang, text in phrases:
        output = OUTPUT_DIR / f"tts_{lang.lower()}.wav"
        dt = generate_and_save(model, text, lang, output)
        results[lang] = dt
    print(f"  -> Latences : {results}\n")
    return results


def test_clonage_vocal(model, voice_ref_path: str):
    """Test 3 : Clonage vocal depuis un fichier de référence (min 3s)."""
    text = "Ce texte est lu avec une voix clonée depuis un échantillon de référence."
    output = OUTPUT_DIR / "tts_clone.wav"
    dt = generate_and_save(model, text, "French", output, voice_ref=voice_ref_path)
    print(f"  -> Latence clonage : {dt:.2f}s\n")
    return dt


def test_latence(model, n_runs: int = 3):
    """Test 4 : Mesure de latence moyenne sur N runs."""
    text = "Test de latence pour la génération vocale."
    times = []
    for i in range(n_runs):
        t0 = time.perf_counter()
        list(model.generate(text=text, language="French"))
        dt = time.perf_counter() - t0
        times.append(dt)
        print(f"  Run {i + 1}/{n_runs} : {dt:.3f}s")
    avg = sum(times) / len(times)
    print(f"  -> Latence moyenne : {avg:.3f}s (sur {n_runs} runs)\n")
    return avg


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Test TTS Voxtral")
    parser.add_argument("--model", default=MODEL_ID, help="ID du modèle TTS")
    parser.add_argument("--voice-ref", help="Fichier audio de référence pour le clonage vocal")
    parser.add_argument("--latency-runs", type=int, default=3, help="Nombre de runs pour le test de latence")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    model = load_tts_model(args.model)

    print("\n=== Test 1 : Génération simple FR ===")
    test_simple_fr(model)

    print("=== Test 2 : Multi-langues ===")
    test_multi_langues(model)

    if args.voice_ref:
        print("=== Test 3 : Clonage vocal ===")
        test_clonage_vocal(model, args.voice_ref)
    else:
        print("=== Test 3 : Clonage vocal [SKIP — pas de --voice-ref] ===\n")

    print("=== Test 4 : Latence ===")
    test_latence(model, n_runs=args.latency_runs)

    print("=== Tous les tests TTS terminés ===")


if __name__ == "__main__":
    main()
