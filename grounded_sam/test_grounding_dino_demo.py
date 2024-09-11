from groundingdino.util.inference import load_model, load_image, predict, annotate, Model
import cv2


CONFIG_PATH = "GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
CHECKPOINT_PATH = "./groundingdino_swint_ogc.pth"
DEVICE = "cuda"
# TEXT_PROMPT = "Horse. Clouds. Grasses. Sky. Hill."
BOX_TRESHOLD = 0.35
TEXT_TRESHOLD = 0.25


IMG_NAME = 'test1.jpg'
IMAGE_PATH = "wargon/" + IMG_NAME
OUTPUT_PATH = "wargon/outputs/" 

TEXT_PROMPT = "Cloth. Damage. Stains. Textiles."

image_source, image = load_image(IMAGE_PATH)
model = load_model(CONFIG_PATH, CHECKPOINT_PATH)

boxes, logits, phrases = predict(
    model=model,
    image=image,
    caption=TEXT_PROMPT,
    box_threshold=BOX_TRESHOLD,
    text_threshold=TEXT_TRESHOLD,
    device=DEVICE,
)

annotated_frame = annotate(image_source=image_source, boxes=boxes, logits=logits, phrases=phrases)
cv2.imwrite(OUTPUT_PATH + IMG_NAME + "_GroundedDINO", annotated_frame)
with open(OUTPUT_PATH + IMG_NAME.split('.')[0] + '.txt', 'w') as f:
    f.write(TEXT_PROMPT)
    f.write('\n')
    f.write(str(boxes.tolist()))
    f.write('\n')
    f.write(str(logits.tolist()))
    f.write('\n')
    f.write(str(phrases))

print(f"Image saved to {OUTPUT_PATH}")
print(f"Boxes: {boxes.tolist()}")
print(f"Phrases: {phrases}")
print(f"Logits: {logits.tolist()}")
