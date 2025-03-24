import gc
import numpy as np
import torch
import librosa
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import streamlit as st
from streamlit_mic_recorder import mic_recorder
import os
import io
import warnings

warnings.filterwarnings("ignore")

# Language mapping
language_map = {"hindi": "hi", "tamil": "ta", "english": "en", "marathi": "mr", "gujarati": "gu", "kannada": "kn", "telugu": "te"}

@st.cache_resource
def load_model_and_processor():
    model_path = "AbleCredit/Ablecredit-Whisper-Small" 
    model = WhisperForConditionalGeneration.from_pretrained(model_path, device_map="auto")
    model.eval()
    processor = WhisperProcessor.from_pretrained(model_path)
    return model, processor

model, processor = load_model_and_processor()

SUPPORTED_EXTENSIONS = {'.mp3', '.wav', '.m4a', '.flac', '.ogg', '.aac'}

def preprocess_audio_chunk(audio, sampling_rate=16000, chunk_length_s=30):
    samples_per_chunk = sampling_rate * chunk_length_s
    total_samples = len(audio)
    chunks = []
    for start in range(0, total_samples, samples_per_chunk):
        end = min(start + samples_per_chunk, total_samples)
        chunk = audio[start:end]
        inputs = processor(chunk, sampling_rate=sampling_rate, return_tensors="pt")
        input_features = inputs.input_features
        attention_mask = torch.ones(input_features.shape[:2], dtype=torch.long)
        if len(chunk) < samples_per_chunk:
            pad_length = samples_per_chunk - len(chunk)
            pad_frames = int(pad_length / 16000 * 100)
            attention_mask[:, -pad_frames:] = 0
        chunks.append({"input_features": input_features, "attention_mask": attention_mask})
    return chunks

def translate_audio(audio, sr, model, processor, chunk_length_s=30):
    st.write(f"Audio duration: {len(audio) / sr:.2f} seconds")
    audio_chunks = preprocess_audio_chunk(audio, sampling_rate=sr, chunk_length_s=chunk_length_s)
    st.write(f"Number of chunks: {len(audio_chunks)}")
    
    full_translation = []
    for i, chunk in enumerate(audio_chunks):
        input_features = chunk["input_features"].to("cuda" if torch.cuda.is_available() else "cpu")
        attention_mask = chunk["attention_mask"].to("cuda" if torch.cuda.is_available() else "cpu")
        
        with torch.no_grad():
            with torch.amp.autocast(device_type="cuda" if torch.cuda.is_available() else "cpu"):
                generated_tokens = model.generate(
                    input_features=input_features,
                    attention_mask=attention_mask,
                    max_new_tokens=128,
                    max_length=60,
                    num_beams=5,
                    task="translate",
                    forced_decoder_ids=None
                ).cpu().numpy()
        
        decoded_preds = processor.tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
        chunk_translation = decoded_preds[0]
        st.write(f"Chunk {i + 1} Translation: {chunk_translation}")
        full_translation.append(chunk_translation)
        
        del input_features, attention_mask, generated_tokens
        gc.collect()
    
    final_translation = " ".join(full_translation).strip()
    return final_translation

st.title("Whispering Voices: Fine-Tuning for Indic-to-English Translation")
st.write("Translate audio from Indian languages to English using finetuned Whisper-small!")

sample_audios = {
    "Hindi Sample": "samples/hindi.mp3",
    "Tamil Sample": "samples/tamil.mp3",
    "Kannada Sample": "samples/kannada.mp3",
    "Marathi Sample": "samples/marathi.mp3",
    "Gujarati Sample": "samples/gujarati.mp3",
    "Telugu Sample": "samples/telugu.mp3"
}

input_method = st.radio("Choose input method:", ("Select Sample Audio", "Upload Audio", "Record Audio"))

audio_data = None
sr = 16000

if input_method == "Select Sample Audio":
    sample_choice = st.selectbox("Select a sample audio:", list(sample_audios.keys()))
    if sample_choice:
        audio_path = sample_audios[sample_choice]
        if os.path.exists(audio_path):
            audio_data, sr = librosa.load(audio_path, sr=16000)
            st.audio(audio_path)
        else:
            st.error(f"Sample file {audio_path} not found in the repository!")

elif input_method == "Upload Audio":
    uploaded_file = st.file_uploader("Upload an audio file", type=list(SUPPORTED_EXTENSIONS))
    if uploaded_file:
        audio_bytes = uploaded_file.read()
        audio_data, sr = librosa.load(io.BytesIO(audio_bytes), sr=16000)
        st.audio(audio_bytes, format=uploaded_file.type)

elif input_method == "Record Audio":
    st.write("Click the button below to start recording:")
    recorded_audio = mic_recorder(start_prompt="Start Recording", stop_prompt="Stop Recording", format="wav")
    if recorded_audio:
        audio_data, sr = librosa.load(io.BytesIO(recorded_audio["bytes"]), sr=16000)
        st.audio(recorded_audio["bytes"], format="audio/wav")

if audio_data is not None:
    if st.button("Translate"):
        with st.spinner("Translating..."):
            try:
                translation = translate_audio(audio_data, sr, model, processor)
                st.success("Translation completed!")
                st.write("**Final Translation:**", translation)
            except Exception as e:
                st.error(f"Error during translation: {str(e)}")
else:
    st.write("Please select, upload, or record an audio file to translate.")