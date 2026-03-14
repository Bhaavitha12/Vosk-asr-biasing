# Vosk ASR Biasing

This project explores **improving recognition of domain-specific words in Vosk Automatic Speech Recognition (ASR)** using a **bias word list**.

The system compares:

* Baseline Vosk transcription
* Bias-assisted transcription
* Error analysis of predicted vs ground truth text
* Recall comparison for bias words

The goal is to evaluate whether **biasing improves recognition accuracy for important vocabulary**.

---

# Project Structure

```
VOSK_2.0
│
├── scripts/
│   ├── transcribe.py                # baseline transcription
│   ├── transcribe_with_bias.py     # transcription with bias list
│   ├── bias_word_analysis.py       # analysis of bias word predictions
│   ├── error_analysis.py           # compare predictions with ground truth
│   └── recall_comparision.py       # recall comparison for bias words
│
├── audio/
│   └── ground_truth/               # ground truth transcripts
│
├── bias_words.txt                  # list of bias words
├── .gitignore
└── README.md
```

Large files like **audio datasets and models are intentionally not stored in this repository**.

---

# Setup

## 1. Clone the repository

```
git clone https://github.com/Bhaavitha12/Vosk-asr-biasing.git
cd Vosk-asr-biasing
```

---

## 2. Create a Python environment

```
python -m venv venv
venv\Scripts\activate
```

Install dependencies:

```
pip install vosk
pip install soundfile
```

---

## 3. Download Vosk Model

Download the small English model:

https://alphacephei.com/vosk/models

Recommended model:

```
vosk-model-small-en-us-0.15
```

After downloading, place it inside:

```
model/vosk-model-small-en-us-0.15
```

---

# Running the System

## Baseline transcription

```
python scripts/transcribe.py
```

---

## Transcription with bias words

```
python scripts/transcribe_with_bias.py
```

---

## Error analysis

```
python scripts/error_analysis.py
```

---

## Bias word recall comparison

```
python scripts/recall_comparision.py
```

---

# Bias Word List

The file `bias_words.txt` contains **domain-specific vocabulary** that the ASR system should prioritize during decoding.

The bias list is passed to the recognizer to improve recognition of these words.

---

# Future Improvements

* Larger evaluation datasets
* Domain-specific language models
* Quantitative WER comparison
* Real-time ASR biasing

---

# Author

Bhaavitha
