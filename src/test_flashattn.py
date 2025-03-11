from transformers import MarianMTModel, MarianTokenizer, AutoConfig
import torch

# Load Model Configuration with Flash Attention (If Supported)
model_name = "Helsinki-NLP/opus-mt-en-roa"
config = AutoConfig.from_pretrained(model_name)

# Check if Flash Attention 2 is available
config.use_flash_attention_2 = True

# Load Tokenizer
tokenizer = MarianTokenizer.from_pretrained(model_name)
print(tokenizer.supported_language_codes)

# Load Model with Updated Config
model = MarianMTModel.from_pretrained(model_name, config=config, attn_implementation="flash_attention_2").cuda()

# Enable Efficient Computation with torch.compile (If Flash Attention isn't supported natively)
model = torch.compile(model)

# Prepare Inputs
src_text = [
    ">>fra<< this is a sentence in english that we want to translate to french",
    ">>por<< This should go to portuguese",
    ">>esp<< And this to Spanish",
]

inputs = tokenizer(src_text, return_tensors="pt", padding=True)

# Move to CUDA for Faster Inference (If Available)
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
inputs = {key: val.to(device) for key, val in inputs.items()}

# Generate Translations with Flash Attention Optimization
with torch.inference_mode():
    translated = model.generate(**inputs)

# Decode and Print Translations
print([tokenizer.decode(t, skip_special_tokens=True) for t in translated])
