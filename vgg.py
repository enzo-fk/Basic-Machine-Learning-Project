import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from torchvision import models

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

if torch.cuda.is_available():
    device = torch.device("cuda")
    print("GPU is available")
else:
    device = torch.device("cpu")
    print("No GPU available, using CPU")

transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(30),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,)),
    transforms.Lambda(lambda x: x.repeat(3, 1, 1)),

])
transform2 = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,)),
    transforms.Lambda(lambda x: x.repeat(3, 1, 1)),

])

trainset = MNIST(root='./data', train=True, download=True, transform=transform)1
trainloader = DataLoader(trainset, batch_size=64, shuffle=True)

testset = MNIST(root='./data', train=False, download=True, transform=transform2)
testloader = DataLoader(testset, batch_size=64, shuffle=False)

model = models.vgg16_bn(num_classes=10)
model.to(device)
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

num_epochs = 30
best_val_acc = 0.0
avg_train_losses, avg_train_accs, avg_val_losses, avg_val_accs = [], [], [], []

for epoch in range(num_epochs):
    model.train()
    train_loss_history = []
    train_acc_history = []
    for x, y in trainloader:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        x, y = x.to(device), y.to(device)

        y_one_hot = nn.functional.one_hot(y, num_classes=10).float()
        y_pred = model(x)

        loss = loss_fn(y_pred, y_one_hot)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        acc = (y_pred.argmax(dim=1) == y).float().mean()
        train_loss_history.append(loss.item())
        train_acc_history.append(acc.item())

    avg_train_loss = sum(train_loss_history) / len(train_loss_history)
    avg_train_acc = sum(train_acc_history) / len(train_acc_history)
    avg_train_losses.append(avg_train_loss)
    avg_train_accs.append(avg_train_acc)

    model.eval()
    val_loss_history = []
    val_acc_history = []
    for x, y in testloader:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        x, y = x.to(device), y.to(device)
        y_one_hot = nn.functional.one_hot(y, num_classes=10).float()
        with torch.no_grad():
            y_pred = model(x)
            loss = loss_fn(y_pred, y_one_hot)

            acc = (y_pred.argmax(dim=1) == y).float().mean()
        val_loss_history.append(loss.item())
        val_acc_history.append(acc.item())

    avg_val_loss = sum(val_loss_history) / len(val_loss_history)
    avg_val_acc = sum(val_acc_history) / len(val_acc_history)
    avg_val_losses.append(avg_val_loss)
    avg_val_accs.append(avg_val_acc)

    print(
        f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {avg_train_loss:.4f}, Train Acc: {avg_train_acc:.2f}%, Test Loss: {avg_val_loss :.4f}, Test Acc: {avg_val_acc:.2f}%")

    if avg_val_acc >= best_val_acc:
        best_val_acc = avg_val_acc
        torch.save(model.state_dict(), "best_model_Vgg.pth")

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.plot(avg_train_losses, label="Train Loss")
plt.plot(avg_val_losses, label="Test Loss")
plt.legend()
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Loss Curve")

plt.subplot(1, 2, 2)
plt.plot(avg_train_accs, label="Train Accuracy")
plt.plot(avg_val_accs, label="Test Accuracy")
plt.legend()
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Accuracy Curve")

plt.savefig("training_curve_Vgg.png")
plt.show()