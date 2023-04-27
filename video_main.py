import os
from PIL import Image
import numpy as np

import torch
import torch.nn as nn
import torchvision
from torchvision import transforms

images = os.listdir("./images")

os.environ["TORCH_HOME"] = "E:\model_weights_edir"
model = torchvision.models.resnet18(weights = "DEFAULT")

all_names = []
all_vecs = None
model.eval()
root = "./images/"

transform = transforms.Compose([
    transforms.Resize((256 , 256)) ,
    transforms.ToTensor() ,
    transforms.Normalize(mean = [0.485 , 0.456 , 0.406] , std = [0.229 , 0.224 , 0.225])
])

activation = {}
def get_activation(name):
    def hook(model , input , output):
        activation[name] = output.detach()
    return hook

model.avgpool.register_forward_hook(get_activation("avgpool"))

with torch.no_grad():
    for i , file in enumerate(images):
        try:
            img = Image.open(root + file)
            img = transform(img)
            out = model(img[None , ...])
            vec = activation["avgpool"].numpy().squeeze()[None , ...]
            if all_vecs is None:
                all_vecs = vec
            else:
                all_vecs = np.vstack([all_vecs , vec])
            all_names.append(file)
        except:

            continue
        if i % 100 == 0 and i != 0:
            print(i , "done")

#np.save("all_vecs.npy" , all_vecs)
#np.save("all_names.npy" , all_names)











