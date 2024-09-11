# Supervision library from Roboflow


`supervision` library from roboflow offers a number of convenience functions out of the box for models like YOLOv5, EfficientDet, SegmentAnythingModel (SAM), and more. Great for rapid testing.

**BUT** it seems that the library is not really suited for training and annotating. It can annotate using existing models, but does not offer a nice interface for annotations.

## Environment setup

```bash
conda create -n supervision python=3.10
conda activate supervision
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch-nightly -c nvidia
conda install -c conda-forge jupyterlab pandas numpy matplotlib scipy pip plotly scikit-learn seaborn streamlit gradio
pip install supervision[desktop]
pip install ultralytics
```
