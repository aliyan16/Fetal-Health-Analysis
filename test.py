import json, torch, open_clip
from PIL import Image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
with open("FetalCLIP_config.json","r") as f:
    cfg = json.load(f)
open_clip.factory._MODEL_CONFIGS["FetalCLIP"] = cfg

model, _, preprocess = open_clip.create_model_and_transforms("FetalCLIP", pretrained="FetalCLIP_weights.pt")
tok = open_clip.get_tokenizer("FetalCLIP")
model.eval().to(device)

img = preprocess(Image.new("RGB",(256,256),color="black")).unsqueeze(0).to(device)
texts = ["Fetal brain ultrasound", "Fetal abdomen ultrasound"]
text_tokens = tok(texts).to(device)

with torch.no_grad(), torch.cuda.amp.autocast():
    ti = model.encode_text(text_tokens); ii = model.encode_image(img)
    ti = ti/ti.norm(dim=-1, keepdim=True); ii = ii/ii.norm(dim=-1, keepdim=True)
    probs = (100.0 * ii @ ti.T).softmax(dim=-1)
print("Label probs:", probs[0].tolist())
