import torch
from parler_tts import ParlerTTSForConditionalGeneration
from transformers import AutoTokenizer
import soundfile as sf

# Set device to CPU
device = "cpu"

# Load model and tokenizers
model = ParlerTTSForConditionalGeneration.from_pretrained("ai4bharat/indic-parler-tts-pretrained").to(device)
tokenizer = AutoTokenizer.from_pretrained("ai4bharat/indic-parler-tts-pretrained")
description_tokenizer = AutoTokenizer.from_pretrained(model.config.text_encoder._name_or_path)

# Kannada text input
prompt =  "ನನಗೆ ನೀನು ಹೇಗೆ ಮಾತನಾಡಬಹುದು!" 
description = "Anu speaks in an angry tone, with sharp emphasis and quick pacing, conveying frustration and irritation."

# Tokenize
description_input_ids = description_tokenizer(description, return_tensors="pt").to(device)
prompt_input_ids = tokenizer(prompt, return_tensors="pt").to(device)

# Generate audio
generation = model.generate(
    input_ids=description_input_ids.input_ids,
    attention_mask=description_input_ids.attention_mask,
    prompt_input_ids=prompt_input_ids.input_ids,
    prompt_attention_mask=prompt_input_ids.attention_mask
)

# Convert to numpy array and save as .wav
audio_arr = generation.cpu().numpy().squeeze()
sf.write("kannada_tts_output.wav", audio_arr, model.config.sampling_rate)

print("✅ Audio saved to kannada_tts_output.wav")
