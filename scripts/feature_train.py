import os
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
 
if torch.cuda.is_available():
    os.environ['CUDA_VISIBLE_DEVICES'] = '3,4'
 
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
 
label_list = []
 
def make_dataset(root):
    images = []
    cnames = os.listdir(root)
    for cname in cnames:
        label_list.append(cname)
        c_path = os.path.join(root, cname)
        if os.path.isdir(c_path):
            fnames = os.listdir(c_path)
            for fname in fnames:
                path = os.path.join(c_path, fname)
                images.append(path)
    return images
 
class ImageDataLoader(object):
    def __init__(self, image_root, batch_size, image_size):
        self.tranform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        self.dataset = torchvision.datasets.ImageFolder(image_root, transform=self.tranform)
        self.dataloader = torch.utils.data.DataLoader(
            self.dataset,
            shuffle=True,
            batch_size=batch_size,
            num_workers=4,
            drop_last=True
        )
 
    def load_data(self):
        return self
 
    def __len__(self):
        return len(self.dataset)
 
    def __iter__(self):
        for i, data in enumerate(self.dataloader):
            yield data
 
class Feature(nn.Module):
    def __init__(self, label_size=125):
        super(Feature, self).__init__()
        self.resnet = torchvision.models.resnet50(pretrained=True)
        self.features = nn.Sequential(*list(self.resnet.children())[:-1])
        self.fc = nn.Linear(2048, label_size)
 
    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x
 
 
if __name__ == "__main__":

    image_root = ""
    batch_size = 256
    image_size = 256
    dataloader = ImageDataLoader(image_root, batch_size, image_size)
    dataset = dataloader.load_data()

    model = Feature()
    model = torch.nn.DataParallel(model)
    model = model.cuda()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    make_dataset(image_root)

    running_loss = 0.0
    num_step = 100

    for step in range(100):
        for i, datas, in enumerate(dataset):
            model.train()

            optimizer.zero_grad()

            data = datas[0].to(device)
            label = datas[1].to(device)
            outputs = model(data)
            loss = criterion(outputs, label)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % num_step == 0 and step != 0:  # print every num_step mini-batches
                print(f'[{step}] loss: {running_loss / num_step:.3f}')
                running_loss = 0.0
                model.eval()
                accuracy = 0.0
                total = 0.0
                with torch.no_grad():
                    outputs = model(data)
                    _, predicted = torch.max(outputs.data, 1)
                    total += label.size(0)
                    accuracy += (predicted == label).sum().item()
                    print(f"accuracy is {100 * accuracy / total}%")
