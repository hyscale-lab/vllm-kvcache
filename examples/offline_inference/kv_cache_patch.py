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
    "Once upon a time, a princess was living in a castle",
    #r"Answer the following math problem.\nThe last line of your response should be your integer answer within \\boxed{{}}.\n\nLet $b \\geq 2$ be an integer. Call a positive integer $n$ $b$\\textit{-eautiful} if it has exactly two digits when expressed in base $b$, and these two digits sum to $\\sqrt{n}$. For example, $81$ is $13$-eautiful because $81=\\underline{6}\\underline{3}_{13}$ and $6+3=\\sqrt{81}$. Find the least integer $b \\geq 2$ for which there are more than ten $b$-eautiful integers.\n\nPut your final answer within \\boxed{{}}\nThink step by step before answering.",
    #r"Answer the following math problem.\nThe last line of your response should be your integer answer within \\boxed{{}}.\n\nWhat is the sum of all positive integers n less than 1000 for which n^2 + 3n + 2 is divisible by n + 1?\n\nPut your final answer within \\boxed{{}}\nThink step by step before answering.",
    #r'Answer the following math problem.\nThe last line of your response should be your integer answer within \\boxed{{}}.\n\nEvery morning Aya goes for a $9$-kilometer-long walk and stops at a coffee shop afterwards. When she walks at a constant speed of $s$ kilometers per hour, the walk takes her 4 hours, including $t$ minutes spent in the coffee shop. When she walks $s+2$ kilometers per hour, the walk takes her 2 hours and 24 minutes, including $t$ minutes spent in the coffee shop. Suppose Aya walks at $s+\frac{1}{2}$ kilometers per hour. Find the number of minutes the walk takes her, including the $t$ minutes spent in the coffee shop.\n\nPut your final answer within \\boxed{{}}\nThink step by step before answering.'
    #r"Answer the following math problem.\nThe last line of your response should be your integer answer within \\boxed{{}}.\n\nEvery morning Aya goes for a $9$-kilometer-long walk and stops at a coffee shop afterwards. When she walks at a constant speed of $s$ kilometers per hour, the walk takes her 4 hours, including $t$ minutes spent in the coffee shop. When she walks $s+2$ kilometers per hour, the walk takes her 2 hours and 24 minutes, including $t$ minutes spent in the coffee shop. Suppose Aya walks at $s+\frac{1}{2}$ kilometers per hour. Find the number of minutes the walk takes her, including the $t$ minutes spent in the coffee shop.\n\nPut your final answer within \\boxed{{}}\nThink step by step before answering. Okay, let's see. So Aya walks 9 kilometers every morning and then stops at a coffee shop. The problem gives two different walking speeds and the total time including the coffee shop time t. I need to find the time it takes when she walks at s + 1/2 km/h, including the same t minutes.\n\nFirst, let's parse the problem again. When she walks at speed s, the total time (walking plus coffee) is 4 hours, which is 240 minutes. When she walks at s + 2 km/h, the total time is 2 hours and 24 minutes, which is 144 minutes. The coffee shop time t is the same in both cases. So, the difference in total time comes from the difference in walking time.\n\nLet me denote the walking time as T1 and T2 for the two scenarios. So:\n\nFirst case:\nWalking speed = s km/h\nWalking time = 9 / s hours\nTotal time (walking + coffee) = 4 hours = 240 minutes\nBut the coffee time is t minutes. Wait, need to be careful with units here. Since the walking time is in hours, maybe I should convert everything to hours or everything to minutes. Let me choose hours to keep the units consistent.\n\nSo, 4 hours total is 4 hours, which includes the walking time and t minutes. Wait, but t is in minutes. Hmm, maybe I should convert t to hours or convert the total time to minutes. Let me see. Alternatively, perhaps set up equations with variables in hours. Let me try that.\n\nLet me denote the coffee shop time as t hours. Wait, but the problem says t minutes. Hmm, so maybe better to convert all time to minutes. Let me try that approach.\n\nFirst scenario:\nWalking speed = s km/h\nTime taken to walk 9 km: (9 / s) hours. To convert that to minutes, multiply by 60: (9/s)*60 = 540/s minutes.\n\nThen, total time is walking time + t minutes = 4 hours = 240 minutes.\n\nSo equation 1: (540 / s) + t = 240.\n\nSecond scenario:\nWalking speed is s + 2 km/h\nTime taken to walk 9 km: 9/(s + 2) hours, which is (9/(s + 2))*60 = 540/(s + 2) minutes.\n\nTotal time here is 2 hours 24 minutes, which is 144 minutes. So equation 2: (540 / (s + 2)) + t = 144.\n\nNow we have two equations:\n\n1) 540/s + t = 240\n\n2) 540/(s + 2) + t = 144\n\nWe can subtract equation 2 from equation 1 to eliminate t:\n\n[540/s + t] - [540/(s + 2) + t] = 240 - 144\n\nSimplify left side: 540/s - 540/(s + 2) = 96\n\nSo 540 [1/s - 1/(s + 2)] = 96\n\nLet me compute 1/s - 1/(s + 2) = [ (s + 2) - s ] / [s(s + 2) ] = 2 / [s(s + 2)]\n\nSo substituting back:\n\n540 * [2 / (s(s + 2))] = 96\n\nSo 1080 / [s(s + 2)] = 96\n\nMultiply both sides by s(s + 2):\n\n1080 = 96 s(s + 2)\n\nDivide both sides by 96:\n\n1080 / 96 = s(s + 2)\n\nSimplify 1080 divided by 96. Let me compute:\n\nDivide numerator and denominator by 24: 1080 ÷24=45; 96 ÷24=4. So 45/4 = 11.25.",
]
logging.basicConfig(level=logging.INFO)

tokenizer = AutoTokenizer.from_pretrained(MODEL)

tokens = tokenizer.tokenize(prompts[0])
token_ids = tokenizer.encode(prompts[0])
logging.info(f"Original sentence: {prompts[0]}")
#logging.info(f"Tokens: {tokens}")

PATCHED = True
#START_ELE = 483 
START_ELE = 500
END_ELE = 533 
#END_ELE = 500

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