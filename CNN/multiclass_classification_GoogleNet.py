import torch
import matplotlib.pyplot as plt
from torchvision.io import read_image
import torchvision
device = torch.device("cuda")

training_data = torch.load('tensor_data.pt')

model = torchvision.models.GoogLeNet()
model.to(device)

eval_criteria = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
transformation = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor()
])

model.train()
for i in range(25):
    for images, labels in training_data:
        images = images.unsqueeze(0).to(device)
        labels = labels.unsqueeze(0).to(device)
        out = model(images)
        logits = out.logits
        labels = labels.type(torch.int64)
        loss = eval_criteria(logits, labels)
        optimizer.zero_grad()   
        loss.backward()
        optimizer.step()
    print("epoch", i, "Loss", loss.item())
torch.save(model.state_dict(), 'model.pth')
model.load_state_dict(torch.load('model.pth'))
model.eval()

from PIL import Image

img = Image.open('download.jpg')

img = img.resize((128, 128))

img = transformation(img)

prediction = model(img.unsqueeze(0).to(device))
print(prediction)