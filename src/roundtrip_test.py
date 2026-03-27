"""
Test Round-Trip : Texte → TTS → Audio → STT → Texte

Vérifie la cohérence du pipeline complet Voxtral :
  1. Génère de l'audio depuis un texte (TTS)
  2. Transcrit l'audio généré (STT)
  3. Compare le texte original avec la transcription
  4. Calcule un score de similarité (Levenshtein normalisé)

Ce test valide que les deux modèles fonctionnent ensemble
et produisent des résultats cohérents.
"""

import argparse
import time
from pathlib import Path

import numpy as np
import soundfile as sf

TTS_MODEL = "mlx-community/Voxtral-4B-TTS-2603-mlx-6bit"
STT_MODEL = "mistralai/Voxtral-Mini-3B-2507"
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "output"


def levenshtein_similarity(s1: str, s2: str) -> float:
    """Calcule la similarité normalisée entre deux chaînes (0.0 à 1.0)."""
    s1, s2 = s1.lower().strip(), s2.lower().strip()
    if s1 == s2:
        return 1.0
    len1, len2 = len(s1), len(s2)
    if len1 == 0 or len2 == 0:
        return 0.0

    matrix = [[0] * (len2 + 1) for _ in range(len1 + 1)]
    for i in range(len1 + 1):
        matrix[i][0] = i
    for j in range(len2 + 1):
        matrix[0][j] = j

    for i in range(1, len1 + 1):
        for j in range(1, len2 + 1):
            cost = 0 if s1[i - 1] == s2[j - 1] else 1
            matrix[i][j] = min(
                matrix[i - 1][j] + 1,
                matrix[i][j - 1] + 1,
                matrix[i - 1][j - 1] + cost,
            )

    distance = matrix[len1][len2]
    return 1.0 - distance / max(len1, len2)


def roundtrip_test(text: str, language: str, tts_lang: str,
                   tts_model_id: str = TTS_MODEL,
                   stt_model_id: str = STT_MODEL) -> dict:
    """Exécute un test round-trip complet."""
    from mlx_audio.tts.utils import load_model as load_tts
    from mlx_voxtral import VoxtralForConditionalGeneration, VoxtralProcessor

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # --- Étape 1 : TTS ---
    print(f"\n[ROUNDTRIP] Texte original : '{text}'")
    print(f"[ROUNDTRIP] Étape 1/3 : Génération TTS ({tts_lang})...")

    tts = load_tts(tts_model_id)
    t0 = time.perf_counter()
    results = list(tts.generate(text=text, language=tts_lang))
    tts_time = time.perf_counter() - t0

    audio = np.array(results[0].audio)
    sample_rate = getattr(results[0], "sample_rate", 24_000)
    wav_path = OUTPUT_DIR / "roundtrip.wav"
    sf.write(str(wav_path), audio, sample_rate)
    audio_duration = len(audio) / sample_rate
    print(f"  -> Audio généré : {wav_path.name} ({audio_duration:.1f}s, en {tts_time:.2f}s)")

    # --- Étape 2 : STT ---
    print(f"[ROUNDTRIP] Étape 2/3 : Transcription STT ({language})...")

    stt = VoxtralForConditionalGeneration.from_pretrained(stt_model_id)
    processor = VoxtralProcessor.from_pretrained(stt_model_id)

    inputs = processor.apply_transcrition_request(
        language=language,
        audio=str(wav_path),
    )

    t0 = time.perf_counter()
    outputs = stt.generate(**inputs, max_new_tokens=1024, temperature=0.0)
    stt_time = time.perf_counter() - t0

    transcription = processor.decode(
        outputs[0][inputs.input_ids.shape[1]:],
        skip_special_tokens=True,
    )
    print(f"  -> Transcription : '{transcription}'")
    print(f"  -> Temps STT : {stt_time:.2f}s")

    # --- Étape 3 : Comparaison ---
    similarity = levenshtein_similarity(text, transcription)
    print(f"\n[ROUNDTRIP] Étape 3/3 : Comparaison")
    print(f"  Original     : {text}")
    print(f"  Transcription: {transcription}")
    print(f"  Similarité   : {similarity:.1%}")
    print(f"  TTS time     : {tts_time:.2f}s")
    print(f"  STT time     : {stt_time:.2f}s")
    print(f"  Total        : {tts_time + stt_time:.2f}s")

    status = "PASS" if similarity > 0.7 else "FAIL"
    print(f"  Résultat     : {status} (seuil: 70%)\n")

    return {
        "original": text,
        "transcription": transcription,
        "similarity": similarity,
        "tts_time": tts_time,
        "stt_time": stt_time,
        "audio_duration": audio_duration,
        "status": status,
    }


def main():
    parser = argparse.ArgumentParser(description="Test Round-Trip TTS → STT")
    parser.add_argument("--tts-model", default=TTS_MODEL)
    parser.add_argument("--stt-model", default=STT_MODEL)
    parser.add_argument("--language", default="fr", help="Langue STT (fr, en, es...)")
    parser.add_argument("--tts-lang", default="French", help="Langue TTS (French, English...)")
    args = parser.parse_args()

    test_phrases = [
        "Bonjour, ceci est un test du pipeline vocal complet.",
        "La synthèse et la reconnaissance vocale fonctionnent ensemble.",
        "Voxtral est un modèle open source de Mistral AI.",
    ]

    print("=" * 60)
    print("  TEST ROUND-TRIP : Texte → TTS → Audio → STT → Texte")
    print("=" * 60)

    all_results = []
    for phrase in test_phrases:
        result = roundtrip_test(
            text=phrase,
            language=args.language,
            tts_lang=args.tts_lang,
            tts_model_id=args.tts_model,
            stt_model_id=args.stt_model,
        )
        all_results.append(result)

    # --- Résumé ---
    print("=" * 60)
    print("  RÉSUMÉ")
    print("=" * 60)
    passed = sum(1 for r in all_results if r["status"] == "PASS")
    total = len(all_results)
    avg_sim = sum(r["similarity"] for r in all_results) / total
    print(f"  Tests réussis : {passed}/{total}")
    print(f"  Similarité moy: {avg_sim:.1%}")
    for r in all_results:
        print(f"  [{r['status']}] {r['similarity']:.0%} | {r['original'][:50]}...")
    print()


if __name__ == "__main__":
    main()
