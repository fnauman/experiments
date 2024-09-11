# Hugginface Transformers

Experiments using the `huggingface` set of libraries along with other libraries such as `peft` to train and infer LLMs. 

## Blog: Personal Copilot

[Source](https://huggingface.co/blog/personal-copilot)

Installation instructions:

```bash
conda create -n huggingface_llms
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
git clone git@github.com:pacman100/DHS-LLM-Workshop.git
cd DHS-LLM-Workshop
pip install -r requirements.txt # Might need to do it again inside the personal_copilot/training folder
MAX_JOBS=8 pip install flash-attn --no-build-isolation
# sudo apt install g++ # By default, Ubuntu doesn't have g++ installed
conda install -c conda-forge jupyterlab pandas numpy matplotlib scipy pip plotly
huggingface-cli login # Enter API token for huggingface
wandb login --relogin # Enter API token for wandb
cd personal_copilot/training
bash run_deepspeed.sh # By default, runs `starcoderbase-1b`
```

`starcoderbase-1b` surprisingly takes 15.5 GBs of RAM - very surprising. I would've thought `1 (model) + 1 (gradients) + 2 (optimizer) + 2 (activations/batch)` = 6 GBs of RAM. 

Runtime: ~2.5 hours on the 4090 GPU.
