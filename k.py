import streamlit as st
import torch
from parler_tts import ParlerTTSForConditionalGeneration
from transformers import AutoTokenizer
import soundfile as sf
import io

# Title
st.title("🗣️ Kannada Text-to-Speech (Parler-TTS)")

# Input
user_text = st.text_area("Enter Kannada text", "ನಮಸ್ಕಾರ! ಇಂದು ನಿಮಗೆ ಹೇಗಿದೆ?", height=100)

# Button
if st.button("🔊 Generate and Play"):
    with st.spinner("Generating audio... please wait"):
        device = "cpu"

        # Load model
        model = ParlerTTSForConditionalGeneration.from_pretrained("ai4bharat/indic-parler-tts-pretrained").to(device)
        tokenizer = AutoTokenizer.from_pretrained("ai4bharat/indic-parler-tts-pretrained")
        description_tokenizer = AutoTokenizer.from_pretrained(model.config.text_encoder._name_or_path)

        # Customize voice
        description = "Anu speaks in a clear and natural voice."

        # Tokenize
        description_input_ids = description_tokenizer(description, return_tensors="pt").to(device)
        prompt_input_ids = tokenizer(user_text, return_tensors="pt").to(device)

        # Generate audio
        generation = model.generate(
            input_ids=description_input_ids.input_ids,
            attention_mask=description_input_ids.attention_mask,
            prompt_input_ids=prompt_input_ids.input_ids,
            prompt_attention_mask=prompt_input_ids.attention_mask
        )

        # Convert and save to in-memory file
        audio_arr = generation.cpu().numpy().squeeze()
        wav_io = io.BytesIO()
        sf.write(wav_io, audio_arr, model.config.sampling_rate, format="WAV")
        wav_io.seek(0)

        # Play audio
        st.audio(wav_io, format='audio/wav')
        st.success("✅ Done!")
