import logging
from typing import Optional

import torch
import torch.nn.functional as F
import numpy as np

from vllm import LLM, SamplingParams
from vllm.attention.layer import Attention
from vllm.forward_context import get_forward_context

from transformers import AutoTokenizer

MODEL = "facebook/opt-125m"
# Sample prompts.
prompts = [
    "Once upon a time, a princess was living in a castle",
]
logging.basicConfig(level=logging.INFO)

tokenizer = AutoTokenizer.from_pretrained(MODEL)

tokens = tokenizer.tokenize(prompts[0])
token_ids = tokenizer.encode(prompts[0])
logging.info(f"Original sentence: {prompts[0]}")

# Assume that slot_mapping doesnt change (1D dimension)
slot_mapping = dict()
curr = 0

# Monkey Patch Attention.forward
# def attention_map(cache: torch.Tensor, slot_mapping: torch.Tensor, query: torch.Tensor, block_size=16):
#     # logging.info(slot_mapping)
#     slot_mapping = torch.tensor(slot_mapping, device=cache.device)
    
#     # Extract values of K Cache into shape ([num_tokens=n, num_kv_heads * head_size])
#     num_kv_heads = cache.shape[3] 
#     head_size = cache.shape[4]
#     scaling = (num_kv_heads * head_size) ** 0.5
    
#     block_idx = slot_mapping // block_size
#     pos_idx = slot_mapping % block_size
    
    
#     key = cache[0, block_idx, pos_idx, :, :]  # shape: [num_slots, num_kv_heads, head_size]
#     key = key.reshape(len(slot_mapping), -1).float()  # flatten and convert to float32
        
#     # for idx, slot in enumerate(slot_mapping):
#     #     key[idx, :] = cache[0, slot // block_size, slot % block_size, :, :].view(-1).to(cache.dtype)
    
#     # Attention should follow the equation softmax(Q @ K.T)
#     query = query.float()
#     attention = F.softmax(query @ key.T / query.shape[1]**0.5, dim=-1)
    
#     # Q should have a dimension of [num_tokens=1, num_kv_heads * head_size]
#     # logging.info(f"[Attention Map]  {attention}")
    
#     # logging.info(f"[Attention Sum] {attention.sum()}")
#     # Output should have a tensor of shape (1, n)    
    
def attention_map(cache: torch.Tensor, slot_mapping: list, output: torch.Tensor, block_size=16):
    # num_kv_heads = cache.shape[3] 
    # head_size = cache.shape[4]
    
    slot_mapping = torch.tensor(slot_mapping, device=cache.device)

    block_idx = slot_mapping // block_size
    pos_idx = slot_mapping % block_size
    
    value = cache[1, block_idx, pos_idx, :, :]
    value = value.reshape(len(slot_mapping), -1).float()
    
    
    output = output.float()
    # Solving for attention map equation output = attention_map @ v
    attention = torch.linalg.lstsq(value.T, output.T).solution.T
    
    # Apply softmax 
    attention = torch.clamp(attention, min=0)
    attention = attention / attention.sum(dim=-1, keepdim=True)
    
    logging.info(f"[Attention Map] {attention}")

_original_forward = Attention.forward

def patched_forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        output_shape: Optional[torch.Size] = None,
    ) -> torch.Tensor:

    forward_context = get_forward_context()
    attn_metadata = forward_context.attn_metadata
    layer_id = getattr(self, "layer_name", "unknown")

    result = _original_forward(self, query, key, value, output_shape)

    self_kv_cache = self.kv_cache[forward_context.virtual_engine]
    
    global slot_mapping

    # Take note that for decode dimension will be (1, *)
    if layer_id != "unknown" and attn_metadata and self_kv_cache.numel() > 0:
        try:
            # logging.info(f"[Slot Mapping] {attn_metadata[layer_id]}")
            if layer_id not in slot_mapping.keys():
                # Will be initialised during prefill phase
                slot_mapping[layer_id] = attn_metadata[layer_id].slot_mapping.clone().cpu().tolist()
            else:
                # This should run during decoding phase
                # attention_map(self_kv_cache, slot_mapping[layer_id], query)
                attention_map(self_kv_cache, slot_mapping[layer_id], result)
                slot_mapping[layer_id].extend(attn_metadata[layer_id].slot_mapping.clone().cpu().tolist())
            # logging.info(f"[Output Hidden Layer] {result.shape}")
        except Exception as e:
            logging.error(e)

    return result

Attention.forward = patched_forward


# Create a sampling params object.
sampling_params = SamplingParams(temperature=0.6, top_p=0.95, top_k=20)


def main():
    # Create an LLM.
    llm = LLM(model=MODEL, tensor_parallel_size=1, gpu_memory_utilization=0.1, max_model_len=64, enforce_eager=True)
    # Generate texts from the prompts.
    # The output is a list of RequestOutput objects
    # that contain the prompt, generated text, and other information.
    outputs = llm.generate(prompts, sampling_params)
    # Print the outputs.
    print("\nGenerated Outputs:\n" + "-" * 60)
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"Prompt:    {prompt!r}")
        print(f"Output:    {generated_text!r}")
        print("-" * 60)



if __name__ == "__main__":
    main()