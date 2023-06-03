import numpy as np
import torch
from torch import nn
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import torchvision
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os
import cv2
import net
from torchvision import datasets
from torch import optim
from tqdm import tqdm
import sys

def main():
    data_transform = {
        "train": transforms.Compose([transforms.RandomResizedCrop(224),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]),
        "val": transforms.Compose([transforms.Resize((224, 224)),  # cannot 224, must (224, 224)
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])}

    data_root=os.path.join(os.getcwd(), "data_set")
    print("训练集根目录：",data_root)

    image_path = os.path.join(data_root, "catsdogs")  # flower data set path
    print("训练集图片文件目录：",image_path)

    train_dataset = datasets.ImageFolder(root=os.path.join(image_path, "train"),
                                         transform=data_transform["train"])

    train_num = len(train_dataset)
    print("训练集图片数量：",train_num)

    target_list=["cat","dog"]

    batch_size = 32
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    print('正在使用{}个线程加载数据集'.format(nw))

    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size, shuffle=True,
                                               num_workers=nw)
    print("训练集加载完毕\n开始加载验证集")
    validate_dataset = datasets.ImageFolder(root=os.path.join(image_path, "val"),
                                            transform=data_transform["val"])
    val_num = len(validate_dataset)
    print("验证集图片数量：",val_num)
    validate_loader = torch.utils.data.DataLoader(validate_dataset,
                                                  batch_size=4, shuffle=False,
                                                  num_workers=nw)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        device_count = torch.cuda.device_count()
        print(f"找到 {device_count} 一个CUDA 设备:",end='')
        for i in range(device_count):
            print(torch.cuda.get_device_name(i))
    else:
        print("没有找到CUDA设备")
    print("device:", device)

    MyNet=net.AlexNet()
    MyNet.to(device)
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(MyNet.parameters(), lr=0.0002)
    save_path = r'.\AlexNet.pth'
    print("权重文件保存路径：",os.path.join(os.getcwd()))

    epochs=100
    best_acc = 0.0
    train_steps = len(train_loader)
    MyNet.train()
    for epoch in range(epochs):
        running_loss=0
        train_bar = tqdm(train_loader, file=sys.stdout)
        #此行代码用于将可迭代对象的迭代过程转化为进度条并输出到控制台
        for step, data in enumerate(train_bar):
            images, labels = data
            optimizer.zero_grad()
            outputs = MyNet(images.to(device))
            loss = loss_function(outputs, labels.to(device))
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            train_bar.desc = "train epoch[{}/{}] loss:{:.3f}".format(epoch + 1,
                                                                     epochs,
                                                                     loss)

            if epoch%5==0 and epoch>1 and step%100==0:
                MyNet.eval()
                acc = 0.0  # accumulate accurate number / epoch
                with torch.no_grad():
                    val_bar = tqdm(validate_loader, file=sys.stdout)
                    for val_data in val_bar:
                        val_images, val_labels = val_data
                        outputs = MyNet(val_images.to(device))
                        predict_y = torch.max(outputs, dim=1)[1]
                        acc += torch.eq(predict_y, val_labels.to(device)).sum().item()

                val_accurate = acc / val_num
                print('[epoch %d] train_loss: %.3f  val_accuracy: %.3f' %
                      (epoch + 1, running_loss / train_steps, val_accurate))

                if val_accurate > best_acc:
                    best_acc = val_accurate
                    torch.save(MyNet.state_dict(), save_path)

                save_path_history = r'.\save_pt\weight'+str(i)+"_"+str(step)+'.pth'
                torch.save(MyNet.state_dict(), save_path_history)
        torch.save(MyNet.state_dict(), "end.pth")
    print('Finished Training')

if __name__ == '__main__':
    main()







