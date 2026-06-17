from typing import cast

from transformers import LlamaForCausalLM, LlamaTokenizer


def load_llama_3_2_1b() -> tuple[LlamaTokenizer, LlamaForCausalLM]:
    model_id = "meta-llama/Llama-3.2-1B"
    tokenizer = LlamaTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token
    model = LlamaForCausalLM.from_pretrained(model_id)
    model.config.pad_token_id = tokenizer.pad_token_id
    model.generation_config.pad_token_id = tokenizer.pad_token_id
    return cast(LlamaTokenizer, tokenizer), cast(LlamaForCausalLM, model)
