import torch
import torchvision

device = torch.device("cuda")

model = torchvision.models.vgg16()
model.to(device)

loss_criteria = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

training_data = torch.load('tensor_data.pt')
model.train()

for i in range(20):
    for images, labels in training_data:
        images = images.unsqueeze(0).to(device)
        labels = labels.unsqueeze(0).to(device)
        out = model(images)
        #logits = out.logit
        labels = labels.type(torch.int64)
        loss = loss_criteria(out, labels)
        optimizer.zero_grad()   
        loss.backward()
        optimizer.step()
    print("epoch", i, "Loss", loss.item())
torch.save(model.state_dict(), 'VGG16_model.pth')

model.load_state_dict(torch.load('VGG16_model.pth'))

model.eval()

from PIL import Image

img = Image.open('download.jpg')
img = img.resize((128, 128))
img = torchvision.transforms.ToTensor()(img)
img = img.unsqueeze(0).to(device)

prediction = model(img)
print(prediction)