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

## Run with your own audio

- Step 1 :- Clone the repository
```
git clone https://github.com/Mayuresh999/Whispering-Voices-Fine-Tuning-for-Indic-to-English-Translation.git
```
- Step 2 :- Install packages in environment of your choise 
```
pip install -r requirements
```
- Step 3 :- Run the Streamlit webapp
```
streamlit run app.py
```

### Read more in the detailed blog [HERE](https://www.linkedin.com/pulse/whispering-voices-fine-tuning-indic-to-english-mayuresh-madiwale-dl6if?lipi=urn%3Ali%3Apage%3Ad_flagship3_series_entity%3BXlAvtQNlTsaFP1j28ivl%2BQ%3D%3D)
