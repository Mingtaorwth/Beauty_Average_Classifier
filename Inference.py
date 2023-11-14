import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
import matplotlib.pyplot as plt
from torchvision import transforms

labels_mark = {0: "average lady", 1: "beautiful lady"}


class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 12, 3)
        self.conv3 = nn.Conv2d(12, 24, 3)
        self.fc1 = nn.Linear(24 * 26 * 26, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 2)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(-1, 24 * 26 * 26)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


PATH = 'beautiful.pth'
image_pred = plt.imread('a_test_image.jpg')


loaded_model = ConvNet()
loaded_model.load_state_dict(torch.load(PATH))
loaded_model.eval()

to_tensor = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

img = to_tensor(image_pred)
img = torch.unsqueeze(img, 0)

img1 = img
outputs = loaded_model(img1)
outputs = F.softmax(outputs, dim=1)
predicted = torch.max(outputs, dim=1)[1].item()
outputs = outputs.squeeze(0)


confidence = outputs[predicted].item()
confidence = round(confidence, 6)
plt.imshow(image_pred)
plt.title(f"This lady is {labels_mark[predicted]}\nConfidence by CNN is : {100*confidence} %.")
plt.show()
