# `diffusers`: Inpainting and image editing

Two key challenges based on 30-min of experimentation:

- Mask has to be precise. A mask that is larger than the region of interest will lead the model to change the image in undesired/unexpected ways.
- Image editing, as with all kinds of generative AI models, is extremely sensitive to the prompt. 


## Installation

By default, the `diffusers` library seems to prefer the CPU version of `pytorch`. To fix this, I first installed the GPU version of `pytorch` and then install the diffusers library.

```bash
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
conda install -c conda-forge accelerate transformers diffusers
conda install -c conda-forge ipywidgets jupyterlab pandas numpy matplotlib scipy pip plotly 
```

That seems to be it. The memory consumption using the `diffedit_example` that also includes the inpainting example using Kardinsky is ~15 GBs, but then goes down after the download stage is over.
