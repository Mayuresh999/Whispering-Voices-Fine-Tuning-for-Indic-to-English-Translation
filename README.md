# Whispering Voices: Fine-Tuning for Indic-to-English Translation

#### Model Name: Ablecredit-Whisper-Small

- Model Overview:

    This is a fine-tuned version of OpenAI's Whisper model, trained to perform speech-to-text translation from Indic languages to English. The model has been optimized using open-source datasets and proprietary dataset created by AbleCredit.com to enhance transcription quality for Indic languages.

- Languages Supported

    The model is trained to handle speech-to-text translation for the following Indic languages:

    - Hindi (hi)
    - Marathi (mr)
    - Gujarati (gu)
    - Tamil (ta)
    - Telugu (te)
    - Kannada (kn)

## Fine-Tuning Process

The base Whisper model was fine-tuned using a carefully curated dataset consisting of open-source corpus and a well curated internal dataset from AbleCredit. The dataset underwent preprocessing, including text normalization, score based filtering and feature engineering.

- Key Improvements After Fine-Tuning

    📉 Word Error Rate (WER) Reduction: 54% lower WER compared to the base model.
    
    📈 BLEU Score Increase: 5.5× higher BLEU score, indicating improved translation accuracy.

## Evaluation Metrics

- Word Error Rate (WER)
WER is used to measure the error rate in transcriptions. A lower WER means better transcription accuracy.

    WER Reduction: 54% improvement over the base Whisper model.

- BLEU Score
The BLEU (Bilingual Evaluation Understudy) Score is a metric used to evaluate machine translation quality. Higher BLEU scores indicate better translation quality.

    BLEU Score Improvement: 5.5× higher BLEU compared to the base model.

Use Cases
Transcribing Indic-language speech into English text
