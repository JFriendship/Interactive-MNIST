import torch
import torch.optim as optim
from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
import ConvNetMNIST     # Custom pytorch model found in ConvNetMNIST.py
#import matplotlib.pyplot as plt     # Used for plotting but has ultimately been commented out

# Set device to the GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#device = 'cuda'

# Getting the training and test data using torchvision's MNIST dataset
training_data = datasets.MNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor()
)

test_data = datasets.MNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor()
)

# Setup pytorch data loaders
BATCH_SIZE = 64
train_dataloader = DataLoader(training_data, batch_size=BATCH_SIZE, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=True)

# ===== Viewing some of the data using matplotlib =====
# figure = plt.figure(figsize=(8, 8))
# cols, rows = 3, 3
# for i in range(1, cols * rows + 1):
#     sample_idx = torch.randint(len(training_data), size=(1,)).item()
#     img, label = training_data[sample_idx]
#     figure.add_subplot(rows, cols, i)
#     #plt.title(labels_map[label])
#     plt.axis("off")
#     plt.imshow(img.squeeze(), cmap="gray")
# plt.show()
# =====================================================

# Miscellaneous Setup
learning_rate = 1e-3
EPOCHS = 50

loss_fn = torch.nn.CrossEntropyLoss()
model = ConvNetMNIST.ConvNetMNIST()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
model.to(device=device)

totalCorrect = 0
total = 0

# Training loop
for i in range(EPOCHS):
    if i>25:    # after 25 epochs, lower the learning rate to prevent overshooting the minimum
        learning_rate = 1e-5
    for step, (x, y) in enumerate(train_dataloader): # x is an array of 2d arrays with pixel values and y is the number it portrays
        y = y.to(device)
        x = x.to(device)
        model.train(True)
        optimizer.zero_grad()
        y_pred = model(x)

        loss = loss_fn(y_pred, y)
        loss.backward()

        optimizer.step()

    print(i, ': ', loss.item())

# Test the model using the test dataset
for step, (x, y) in enumerate(test_dataloader):
    y = y.to(device)
    x = x.to(device)
    with torch.no_grad():
        test_pred = model(x)
    
    pred = torch.argmax(test_pred, 1)

    y = y.squeeze()

    # Count number of correct predictions
    for i in range(pred.size(dim=0)):
        if pred[i].item() == y[i].item():
            totalCorrect += 1
        total += 1

print('Accuracy = ', totalCorrect/total)

# Should the model be saved?
save = input("1:Save, 0:Scrap --> ")
if(save):
    torch.save(model.state_dict(), './MNIST_Model.pt')

print('Program Complete')