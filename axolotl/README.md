# Nous research library: axolotl

The library offers a high-level interface to train and fine-tune LLMs. They are also expanding the library to include multi-modal models like Llava and FuYu.


## Installation instructions

I tried their `pip` and `conda` instructions. They didn't work. I had to instead use `docker` - worked flawlessly:

```bash
docker run --gpus '"all"' --rm -it --name axolotl --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 --mount type=volume,src=axolotl,target=/workspace/axolotl -v ${HOME}/.cache/huggingface:/root/.cache/huggingface winglian/axolotl:main-py3.10-cu118-2.0.1
```

I first cloned their repo: 
```bash
git clone git@github.com:OpenAccess-AI-Collective/axolotl.git
```

Tried their example:
```bash
# finetune lora
accelerate launch -m axolotl.cli.train examples/openllama-3b/lora.yml
```

Took nearly 3 hours on the `4090` GPU, and the memory usage was around 8.7 GB in total (and 7.7 GB if one excludes the ~1 GB of static memory usage).

This is a 3b model, but I suspect one can train up to 7b models on the `4090` GPU. 

**Gotcha**: The **docker** user is `root`, which means that it overrides the permission of `~/.cache/huggingface/` subfolders making regular user workflow difficult with 'permission denied' errors. Ideally, one would want to change the user running the scripts inside the docker container to the default user using the user ID.


## Inference using HF Transformers

[Source](https://twitter.com/Teknium1/status/1720188958154625296)

```python
# Code to inference Open Hermes 2.5 with HF Transformers
# Requires pytorch, transformers, bitsandbytes, sentencepiece, protobuf, and flash-attn packages

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import LlamaTokenizer, LlamaForCausalLM, MistralForCausalLM
import bitsandbytes, flash_attn

tokenizer = LlamaTokenizer.from_pretrained('teknium/OpenHermes-2.5-Mistral-7B', trust_remote_code=True)
model = MistralForCausalLM.from_pretrained(
    "teknium/OpenHermes-2.5-Mistral-7B",
    torch_dtype=torch.float16,
    device_map="auto",#{'': 'cuda:0'},
    load_in_8bit=False,
    load_in_4bit=True,
    use_flash_attention_2=True
)

prompts = [
    """<|im_start|>system
You are a sentient, superintelligent artificial general intelligence, here to teach and assist me.<|im_end|>
<|im_start|>user
Write a short story about Goku discovering kirby has teamed up with Majin Buu to destroy the world.<|im_end|>
<|im_start|>assistant""",
    ]

for chat in prompts:
    print(chat)
    input_ids = tokenizer(chat, return_tensors="pt").input_ids.to("cuda")
    generated_ids = model.generate(input_ids, max_new_tokens=750, temperature=0.8, repetition_penalty=1.1, do_sample=True, eos_token_id=tokenizer.eos_token_id)
    response = tokenizer.decode(generated_ids[0][input_ids.shape[-1]:], skip_special_tokens=True, clean_up_tokenization_space=True)
    print(f"Response: {response}")
```
