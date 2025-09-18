import logging
from typing import Optional

import torch
import random

from vllm import LLM, SamplingParams
from vllm.attention.layer import Attention
from vllm.forward_context import get_forward_context

from transformers import AutoTokenizer

MODEL = "/home/users/ntu/wpang010/scratch/models/QwQ-32B"
# Sample prompts.
prompts = [
    #"Once upon a time, a princess was living in a castle",
    #r"Answer the following math problem.\nThe last line of your response should be your integer answer within \\boxed{{}}.\n\nLet $b \\geq 2$ be an integer. Call a positive integer $n$ $b$\\textit{-eautiful} if it has exactly two digits when expressed in base $b$, and these two digits sum to $\\sqrt{n}$. For example, $81$ is $13$-eautiful because $81=\\underline{6}\\underline{3}_{13}$ and $6+3=\\sqrt{81}$. Find the least integer $b \\geq 2$ for which there are more than ten $b$-eautiful integers.\n\nPut your final answer within \\boxed{{}}\nThink step by step before answering.",
]
logging.basicConfig(level=logging.INFO)

tokenizer = AutoTokenizer.from_pretrained(MODEL)

tokens = tokenizer.tokenize(prompts[0])
token_ids = tokenizer.encode(prompts[0])
logging.info(f"Original sentence: {prompts[0]}")

EPSILON = 0.1

# Monkey Patch Attention.forward

def zero_func(cache: torch.Tensor, slot_mapping: torch.Tensor, block_size=16):
    for idx in slot_mapping:
        if random.random() < EPSILON:        
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
    if key.shape[0] == 1 and layer_id != "unknown" and attn_metadata:
        # Drop token with epsilon chance during decoding stage
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
sampling_params = SamplingParams(temperature=0.6, top_p=0.95, top_k=20, max_tokens=8192)


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