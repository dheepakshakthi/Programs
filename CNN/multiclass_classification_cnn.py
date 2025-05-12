'''
the complexity of the model is low and due to this, the predicted class is inaccurate 
the model's convolution layers can be increased, alterning the kernel's size & stride and the number of nodes per convolution layer can be decreased 
this can help to increase the dept of the model
'''
import torch 
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
from torchvision import transforms
from PIL import Image
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dataset = load_dataset("MCIC") # its a dataset of 15 classes of images

transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])

class LoadData(Dataset):    
    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image = self.dataset[idx]['image']
        label = self.dataset[idx]['label']
        if image.mode != 'RGB':
            image = image.convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, torch.tensor(label, dtype=torch.long)

train_dataset = LoadData(dataset['train'], transform=transform)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

class mcccnn(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  
            #64

            nn.Conv2d(64, 128, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  
            #32
            nn.Conv2d(128, 256, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)   
            #16
        )

        dummy_input = torch.zeros(1, 3, 128, 128)
        with torch.no_grad():
            dummy_output = self.conv(dummy_input)
        flatten_size = dummy_output.view(1, -1).shape[1]

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flatten_size, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 15)
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.fc(x)
        return x

model = mcccnn().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

model.train()
for epoch in range(5):
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
    
    print(f"epoch {epoch+1} loss: {loss.item()}")

model.eval()
