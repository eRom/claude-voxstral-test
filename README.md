# Voxtral Test — TTS & STT sur Apple Silicon (M1 Pro)

Test du pipeline vocal complet avec **Voxtral** (Mistral, mars 2026) sur Apple Silicon.

## Modèles utilisés

| Modèle | Type | Usage |
|--------|------|-------|
| `mlx-community/Voxtral-4B-TTS-2603-mlx-6bit` | TTS (6-bit) | Génération vocale (~3 GB RAM) |
| `mistralai/Voxtral-Mini-3B-2507` | STT | Transcription audio (~2.5 GB RAM) |

## Installation

```bash
# Créer un environnement virtuel
python3 -m venv .venv
source .venv/bin/activate

# Installer les dépendances
pip install -r requirements.txt
```

## Usage

### Test TTS (Text-to-Speech)

```bash
# Génération simple + multi-langues + latence
python -m src.tts_test

# Avec clonage vocal (fichier de référence min 3s)
python -m src.tts_test --voice-ref samples/ma_voix.wav
```

### Test STT (Speech-to-Text)

```bash
# Transcription d'un fichier
python -m src.stt_test --audio samples/speech.wav

# Transcription d'un dossier entier
python -m src.stt_test --audio samples/ --language fr
```

### Test Round-Trip (TTS → STT)

```bash
# Pipeline complet : texte → audio → texte (vérification cohérence)
python -m src.roundtrip_test
```

### Tests unitaires

```bash
pytest
```

## Structure

```
├── src/
│   ├── tts_test.py          # Test TTS
│   ├── stt_test.py          # Test STT
│   └── roundtrip_test.py    # Pipeline combiné
├── samples/                 # Fichiers audio de référence
├── output/                  # Sorties générées
└── tests/                   # Tests automatisés
```

## Pipeline complet

```
Microphone → [STT Voxtral] → texte → [LLM] → réponse → [TTS Voxtral] → audio → Speakers
```

Tout tourne **on-device**, sans cloud, ~6 GB RAM unifiée.
