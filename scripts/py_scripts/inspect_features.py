import torch
from pathlib import Path
import numpy as np
from scripts.py_scripts.data import prompts_at_type_and_category, df_name_config, ROOT_PATH
from scripts.py_scripts.gemma import load_gemma_resources
import sae_lens

tokenizer, gemma, GEMMA_SPARSE_AUTO_ENCODERS = load_gemma_resources()


def extract_gemma_sae_features(text, layer_num=0):
    activations = {}

    def hook_fn(module, input, output):
        if isinstance(output, tuple):
            hidden_activation = output[0]
        else:
            hidden_activation = output

        activations["hidden"] = hidden_activation.detach().cpu()

    gemma_decoder_layer = gemma.model.language_model.layers[layer_num]
    
    handle = gemma_decoder_layer.register_forward_hook(hook_fn)
    
    inputs = tokenizer(text, return_tensors="pt").to("cpu")

    sae_features: torch.Tensor = None
    
    with torch.no_grad():
        
        gemma(**inputs)
        
        sae = GEMMA_SPARSE_AUTO_ENCODERS[layer_num]
        
        sae_features = sae.encode(activations["hidden"])
   
    handle.remove()
    
    return sae_features.detach().cpu()


def get_top_features(sae_features, k):
    
    return torch.topk(sae_features, k=k, dim=-1)


def topk_per_token(text, sae_features, k):
    
    top_sae_features, indices= get_top_features(sae_features, k)
    
    inputs = tokenizer(text, return_tensors="pt")

    tokens = tokenizer.convert_ids_to_tokens(
        inputs["input_ids"][0].tolist()
    )

    res_list = []

    for index, token in enumerate(tokens):
        
        res_list.append([token, top_sae_features[:,index,:]])

    return res_list, top_sae_features, indices


def topk_agg_prompts_at_type(df_name, type, k, layer_num = 5):


    text = prompts_at_type_and_category(df_name, type )

    sae_features = extract_gemma_sae_features(text, layer_num)

    activations = [topk_per_token(prompt, sae_features, k) for prompt in text]

    file_dir: Path = ROOT_PATH / "activations" / "gemma" / f"layer_{layer_num}" / type

    file_dir.mkdir(parents = True, exist_ok = True)

    for i, (_, top_sae_features ,  indices) in enumerate(activations):

        #Tesor shape is 

        res_tensor = torch.zeros_like(sae_features).unsqueeze(0)

        res_tensor = res_tensor.scatter_add(dim = 1, index = indices, src = top_sae_features)

        nmpy_clone = res_tensor.numpy()

        ind_nmpy_clone = indices.numpy()

        arr_file = file_dir / f"arr_prompt_{i}.npy"

        ind_file = file_dir/ f"ind_prompt_{i}.npy"

        arr_file.touch(exist_ok = True)

        np.save(
            file = arr_file,
            arr = nmpy_clone
        )

        np.save(
            file = ind_file,
            arr = ind_nmpy_clone
        )


def feature_intervention(sae_features: torch.Tensor, sae_indices: torch.Tensor, layer_num):


    activation = {}


    def hook_fn(module, input, output):

        sae_encoder: sae_lens.SAE  = GEMMA_SPARSE_AUTO_ENCODERS[layer_num]

        if isinstance(output,  tuple):
            output = output[0]

        active_features = torch.gather(sae_features, dim = -1, index = sae_indices)

        active_features = 3*active_features+5

        sae_features.scatter_(dim = -1, index = sae_indices, src = active_features)

        return sae_encoder.decode(sae_features)







        









print(topk_per_token(prompts_at_type_and_category("homonyms",1), 100, 5 ))


