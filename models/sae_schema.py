import torch
import sae_lens
import transformers
from transformers import AutoTokenizer, AutoModelForCausualLM



model_id = "google/gemma-3-4b-it"

print("Loading Gemma model")

tokenizer = AutoTokenizer.from_pretrained(model_id)


gemma = AutoModelForCausualLM.from_pretrained(model_id, device_map = "auto", torch_dtype = torch.bfloat16)


