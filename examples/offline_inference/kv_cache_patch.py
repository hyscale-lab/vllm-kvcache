import logging
from typing import Optional

import torch

from vllm import LLM, SamplingParams
from vllm.attention.layer import Attention
from vllm.forward_context import get_forward_context

from transformers import AutoTokenizer

MODEL = "/home/users/ntu/wpang010/scratch/models/QwQ-32B"
# Sample prompts.
prompts = [
    r"Once upon a time, a princess was living in a castle",
]
logging.basicConfig(level=logging.INFO)

tokenizer = AutoTokenizer.from_pretrained(MODEL)

tokens = tokenizer.tokenize(prompts[0])
token_ids = tokenizer.encode(prompts[0])
logging.info(f"Original sentence: {prompts[0]}")
logging.info(f"Tokens: {tokens}")

PATCHED = True 
START_ELE = 483 
END_ELE = 533 

if PATCHED:
    logging.info(f"[TOKEN ID LENGTH] {len(token_ids)}")
    logging.info(f"[Deleted KV Cache Tokens] {tokenizer.decode(token_ids[START_ELE:END_ELE])}")

# Monkey Patch Attention.forward

def zero_func(cache: torch.Tensor, slot_mapping: torch.Tensor, block_size=16):
    for idx in slot_mapping[START_ELE: min(END_ELE, len(slot_mapping))]:
        #logging.info(f"[KV Cache] before index: {idx}, {cache[:, idx // block_size, idx % block_size, :, :]}")
        cache[:, idx // block_size, idx % block_size, :, :].zero_()
        #logging.info(f"[KV Cache] after index: {idx}, {cache[:, idx // block_size, idx % block_size, :, :]}")

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
    if key.shape[0] != 1 and layer_id != "unknown" and attn_metadata and PATCHED:
        try:

            #logging.info(f"[KV LOG] Layer {layer_id} key shape: {tuple(key.shape)} value_shape: {tuple(value.shape)} query_shape: {tuple(query.shape)} virtual_engine: {forward_context.virtual_engine} kv_cache_shape: {self_kv_cache.shape}")
            #logging.info(f"[Slot Mapping] {attn_metadata[layer_id].slot_mapping}")
            #logging.info(f"[attn_metadata] {attn_metadata}")
            zero_func(self_kv_cache, attn_metadata[layer_id].slot_mapping)
        except Exception as e:
            logging.error(e)

    return result
Attention.forward = patched_forward


# Create a sampling params object.
sampling_params = SamplingParams(temperature=0.8, top_p=0.95, max_tokens=8192)


def main():
    # Create an LLM.
    llm = LLM(model="/home/users/ntu/wpang010/scratch/models/QwQ-32B", tensor_parallel_size=4, gpu_memory_utilization=0.7, enforce_eager=True)
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