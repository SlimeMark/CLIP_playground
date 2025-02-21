import torch
import clip
from PIL import Image
import os
from tqdm import tqdm
import numpy as np

device = "mps" if torch.backends.mps.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

images = []
images_files = []
for _, file in tqdm(enumerate(os.listdir("/Users/hengjie/Downloads/pic"))):
    if file.endswith((".jpg", ".jpeg", ".png")):
        images_files.append(file)
        images.append(preprocess(Image.open(f"/Users/hengjie/Downloads/pic/{file}")).unsqueeze(0).to(device))

text = clip.tokenize(["An anime girl opens her arms in the sky."]).to(device)

results = []
batch_size = 16
with torch.no_grad():
    for i in range(0, len(images), batch_size):
        batch_images = torch.cat(images[i:i+batch_size])
        image_features = model.encode_image(batch_images)
        text_features = model.encode_text(text)
        
        logits_per_image, logits_per_text = model(batch_images, text)
        for image_feature in image_features:
            similarities = torch.cosine_similarity(image_feature.unsqueeze(0), text_features)
            results.append(similarities.cpu().numpy())
        
result_dict = dict(zip(images_files, results))
sorted_result_dict = dict(sorted(result_dict.items(), key=lambda x: np.mean(x[1]), reverse=True))

for k, v in sorted_result_dict.items():
    print(f"{k}: {v}")