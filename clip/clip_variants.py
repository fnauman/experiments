# 1. General update:
# conda update --all
# 2. Update pytorch and torchvision:
# conda install pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia
# 3. Install open_clip:
# pip install open_clip_torch
# 4. Install transformers:
# pip install git+https://github.com/huggingface/transformers

# Model size: 898 MB

import torch
from PIL import Image


IMG_FILE = "./sample/test1.jpg"
TXT_LIST = ["green", "blue", "gray", "red", "pink", "yellow", "black", "multicolor", "white"]

MODEL = 'hf-hub:Marqo/marqo-fashionSigLIP' # 'jinaai/jina-clip-v2'

device = "cuda" if torch.cuda.is_available() else "cpu"

if MODEL == 'hf-hub:Marqo/marqo-fashionSigLIP':
    import open_clip
    model, preprocess_train, preprocess_val = open_clip.create_model_and_transforms(MODEL)
    tokenizer = open_clip.get_tokenizer(MODEL)

    # Process image and text
    image = preprocess_val(Image.open(IMG_FILE)).unsqueeze(0)
    text = tokenizer(TXT_LIST)
    with torch.inference_mode(), torch.autocast(device, dtype=torch.bfloat16):
        image_features = model.encode_image(image)
        text_features = model.encode_text(text)

elif MODEL == 'jinaai/jina-clip-v2':
    from transformers import AutoModel
    model = AutoModel.from_pretrained(MODEL, trust_remote_code=True).to(device)

    image = Image.open(IMG_FILE)
    text = TXT_LIST
    with torch.inference_mode(), torch.autocast(device, dtype=torch.bfloat16):
        image_features = torch.tensor(model.encode_image(image))
        text_features = torch.tensor(model.encode_text(text))

print(f"{MODEL}")
print(f"Image features: {image_features.shape}, Text features:, {text_features.shape}")
image_features = image_features / image_features.norm(dim=-1, keepdim=True)
text_features = text_features / text_features.norm(dim=-1, keepdim=True)

text_probs = (100.0 * image_features @ text_features.T).softmax(dim=-1)

print("Label probs:", text_probs)
# [0.9860219105287394, 0.00777916527489097, 0.006198924196369721]
