import torch
import torchvision.transforms as transforms
import os
import net
from PIL import Image

def detect(img_path):
    transform = transforms.Compose([transforms.RandomResizedCrop(224),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    model_path = r"D:\pythonItem\catdog\weight.pth"
    Mymodel = net.AlexNet()
    if os.path.exists(model_path):
        print("开始加载模型")
        Mymodel.load_state_dict(torch.load(model_path))
    else:
        print("模型不存在，开始训练")
    img = Image.open(img_path).convert('RGB')
    img_tensor=transform(img)
    img_tensor=img_tensor.unsqueeze(dim=0)
    out=torch.nn.functional.softmax(Mymodel(img_tensor))
    idx=torch.argmax(out)
    target_list = ["cat", "dog"]
    print(target_list[idx])

detect(r"D:\pythonItem\catdog\data_set\catsdogs\val\Dog\331.jpg")