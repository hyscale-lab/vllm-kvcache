import logging
from typing import Optional

import torch

from vllm import LLM, SamplingParams
from vllm.attention.layer import Attention
from vllm.forward_context import get_forward_context

logging.basicConfig(level=logging.INFO, filename="./run.log")

START_ELE = 3
END_ELE = 6

# Monkey Patch Attention.forward

def zero_func(cache: torch.Tensor, slot_mapping: torch.Tensor, block_size=16):
    for idx in slot_mapping[START_ELE, max(END_ELE-START_ELE, len(slot_mapping))]:
        # logging.info(f"[KV Cache] before index: {slot_mapping}, {cache[:, idx // block_size, idx % block_size, :, :]}")
        cache[:, idx // block_size, idx % block_size, :, :].zero_()
        # logging.info(f"[KV Cache] after index: {slot_mapping}, {cache[:, idx // block_size, idx % block_size, :, :]}")

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
    
    # Take note that for decode dimension will be (1, *)
    if key.shape[0] != 1 and self_kv_cache.numel() > 0:
        try:
            
            # logging.info(f"[KV LOG] Layer {layer_id} key shape: {tuple(key.shape)} value_shape: {tuple(value.shape)} query_shape: {tuple(query.shape)} virtual_engine: {forward_context.virtual_engine}")

            # logging.info(f"[Slot Mapping] {attn_metadata[layer_id].slot_mapping}")
            
            zero_func(self_kv_cache, attn_metadata[layer_id].slot_mapping)
        except Exception as e:
            logging.error(e)
    
    return result
Attention.forward = patched_forward

sampling_params = SamplingParams(
    max_tokens=50,
)

llm = LLM(
    model="facebook/opt-125m",
    gpu_memory_utilization=0.3,
    enforce_eager=True,
)

prompt = "Once upon a time, a princess lived in a castle"

outputs = llm.generate(prompt, sampling_params=sampling_params)

logging.info(f"Generated Text: {outputs[0].outputs[0].text}")