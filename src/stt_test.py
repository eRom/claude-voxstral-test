"""
Test STT (Speech-to-Text) avec Voxtral sur Apple Silicon.

Modèle : mistralai/Voxtral-Mini-3B-2507 (~2.5 GB RAM en 4-bit)
Lib    : mlx-voxtral

Tests couverts :
  1. Transcription d'un fichier audio local
  2. Transcription avec modèle quantifié 4-bit
  3. Mesure de latence (temps de transcription vs durée audio = RTF)
"""

import argparse
import time
from pathlib import Path

import soundfile as sf

MODEL_ID = "mistralai/Voxtral-Mini-3B-2507"
MODEL_4BIT = "mlx-community/Voxtral-Mini-4B-Realtime-2602-4bit"
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "output"


def load_stt_model(model_id: str = MODEL_ID):
    """Charge le modèle STT et le processeur."""
    from mlx_voxtral import VoxtralForConditionalGeneration, VoxtralProcessor

    print(f"[STT] Chargement du modèle : {model_id}")
    t0 = time.perf_counter()
    model = VoxtralForConditionalGeneration.from_pretrained(model_id)
    processor = VoxtralProcessor.from_pretrained(model_id)
    dt = time.perf_counter() - t0
    print(f"[STT] Modèle chargé en {dt:.1f}s")
    return model, processor


def transcribe(model, processor, audio_path: str, language: str = "fr",
               max_tokens: int = 1024) -> tuple[str, float]:
    """Transcrit un fichier audio. Retourne (texte, durée)."""
    print(f"[STT] Transcription : {audio_path} (langue={language})")

    inputs = processor.apply_transcrition_request(
        language=language,
        audio=audio_path,
    )

    t0 = time.perf_counter()
    outputs = model.generate(**inputs, max_new_tokens=max_tokens, temperature=0.0)
    dt = time.perf_counter() - t0

    transcription = processor.decode(
        outputs[0][inputs.input_ids.shape[1]:],
        skip_special_tokens=True,
    )

    print(f"[STT] -> Transcription ({dt:.2f}s) : {transcription[:100]}...")
    return transcription, dt


def get_audio_duration(audio_path: str) -> float:
    """Retourne la durée d'un fichier audio en secondes."""
    info = sf.info(audio_path)
    return info.duration


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_transcription_fichier(model, processor, audio_path: str, language: str = "fr"):
    """Test 1 : Transcription d'un fichier audio local."""
    text, dt = transcribe(model, processor, audio_path, language)
    duration = get_audio_duration(audio_path)
    rtf = dt / duration if duration > 0 else float("inf")
    print(f"  -> Durée audio : {duration:.1f}s | Temps transcription : {dt:.2f}s | RTF : {rtf:.2f}x\n")
    return {"transcription": text, "time": dt, "audio_duration": duration, "rtf": rtf}


def test_transcription_multi_fichiers(model, processor, audio_dir: str, language: str = "fr"):
    """Test 2 : Transcription de tous les .wav dans un dossier."""
    audio_dir = Path(audio_dir)
    wav_files = sorted(audio_dir.glob("*.wav"))

    if not wav_files:
        print(f"  -> Aucun fichier .wav trouvé dans {audio_dir}\n")
        return []

    results = []
    for wav in wav_files:
        result = test_transcription_fichier(model, processor, str(wav), language)
        results.append({"file": wav.name, **result})
    return results


def test_latence_stt(model, processor, audio_path: str, n_runs: int = 3):
    """Test 3 : Mesure de latence moyenne sur N runs."""
    times = []
    for i in range(n_runs):
        _, dt = transcribe(model, processor, audio_path)
        times.append(dt)
        print(f"  Run {i + 1}/{n_runs} : {dt:.3f}s")
    avg = sum(times) / len(times)
    duration = get_audio_duration(audio_path)
    avg_rtf = avg / duration if duration > 0 else float("inf")
    print(f"  -> Latence moyenne : {avg:.3f}s | RTF moyen : {avg_rtf:.2f}x\n")
    return avg


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Test STT Voxtral")
    parser.add_argument("--model", default=MODEL_ID, help="ID du modèle STT")
    parser.add_argument("--audio", required=True, help="Fichier audio ou dossier à transcrire")
    parser.add_argument("--language", default="fr", help="Langue de transcription (fr, en, es...)")
    parser.add_argument("--latency-runs", type=int, default=3, help="Nombre de runs pour le test de latence")
    args = parser.parse_args()

    model, processor = load_stt_model(args.model)

    audio_path = Path(args.audio)

    if audio_path.is_dir():
        print("\n=== Test : Transcription multi-fichiers ===")
        test_transcription_multi_fichiers(model, processor, str(audio_path), args.language)
    elif audio_path.is_file():
        print("\n=== Test 1 : Transcription fichier ===")
        test_transcription_fichier(model, processor, str(audio_path), args.language)

        print("=== Test 2 : Latence STT ===")
        test_latence_stt(model, processor, str(audio_path), n_runs=args.latency_runs)
    else:
        print(f"[ERREUR] Fichier ou dossier introuvable : {args.audio}")
        return

    print("=== Tous les tests STT terminés ===")


if __name__ == "__main__":
    main()
