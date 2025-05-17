import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.models import resnet50
from torch.utils.data import Dataset
from torchvision import datasets, models, transforms
from torch.nn.modules.loss import BCEWithLogitsLoss
from tqdm import tqdm
import matplotlib.pyplot as plt

def make_train_step(model, optimizer, loss_fn):
    def train_step(x, y):
        # make prediction
        yhat = model(x)
        # enter train mode
        model.train()
        # compute loss
        loss = loss_fn(yhat, y)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        return loss
    return train_step

device = "cuda" if torch.cuda.is_available() else "cpu"
if torch.cuda.is_available():
    device = torch.device("cuda")
    print("GPU is available")
else:
    device = torch.device("cpu")
    print("No GPU available, using CPU")

model = models.resnet50(pretrained=True)


in_features = model.fc.in_features
model.fc = nn.Linear(in_features, 1)
model.fc = nn.Sequential(model.fc, nn.Sigmoid())


model = model.to(device)

loss_fn = BCEWithLogitsLoss()  # binary cross-entropy with sigmoid, so no need to use sigmoid in the model

# optimizer
optimizer = torch.optim.Adam(model.fc.parameters())

# train step
train_step = make_train_step(model, optimizer, loss_fn)

traindir = "C:\\Users\Enzo Fabien\AppData\Local\Programs\Python\Python312\Hw2_H24115328_ 朱俊曉_V1\Q2_Dataset\dataset\\training_dataset"
testdir = "C:\\Users\Enzo Fabien\AppData\Local\Programs\Python\Python312\Hw2_H24115328_ 朱俊曉_V1\Q2_Dataset\dataset\\validation_dataset"

# transformations
train_transforms = transforms.Compose([transforms.Resize((224, 224)),
                                       transforms.ToTensor(),
                                       torchvision.transforms.Normalize(
                                           mean=[0.485, 0.456, 0.406],
                                           std=[0.229, 0.224, 0.225],
                                       ), #transforms.RandomErasing()
                                       ])
test_transforms = transforms.Compose([transforms.Resize((224, 224)),
                                      transforms.ToTensor(),
                                      torchvision.transforms.Normalize(
                                          mean=[0.485, 0.456, 0.406],
                                          std=[0.229, 0.224, 0.225],
                                      ),
                                      ])

# datasets
train_data = datasets.ImageFolder(traindir, transform=train_transforms)
test_data = datasets.ImageFolder(testdir, transform=test_transforms)

# dataloader
trainloader = torch.utils.data.DataLoader(train_data, shuffle=True, batch_size=16)
testloader = torch.utils.data.DataLoader(test_data, shuffle=True, batch_size=16)

losses = []
val_losses = []
train_accuracies = []  # Track training accuracy
test_accuracies = []  # Track test accuracy

epoch_train_losses = []
epoch_test_losses = []

n_epochs = 3
early_stopping_tolerance = 3
early_stopping_threshold = 0.03

for epoch in range(n_epochs):
    epoch_loss = 0
    correct_train = 0  # Counter for correct predictions during training
    total_train = 0    # Counter for total examples during training

    for i, data in tqdm(enumerate(trainloader), total=len(trainloader)):
        x_batch, y_batch = data
        x_batch = x_batch.to(device)  # move to gpu
        y_batch = y_batch.unsqueeze(1).float()  # convert target to the same nn output shape
        y_batch = y_batch.to(device)  # move to gpu

        loss = train_step(x_batch, y_batch)
        epoch_loss += loss / len(trainloader)
        losses.append(loss)

        # Calculate training accuracy
        predictions = (torch.sigmoid(model(x_batch)) > 0.5).float()
        correct_train += (predictions == y_batch).sum().item()
        total_train += y_batch.size(0)

    epoch_train_losses.append(epoch_loss)
    train_accuracy = correct_train / total_train
    train_accuracies.append(train_accuracy)
    print('\nEpoch : {}, train loss : {}, train accuracy: {:.2%}'.format(epoch + 1, epoch_loss, train_accuracy))

    # validation doesn't require gradient
    with torch.no_grad():
        cum_loss = 0
        correct_test = 0  # Counter for correct predictions during testing
        total_test = 0    # Counter for total examples during testing

        for x_batch, y_batch in testloader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.unsqueeze(1).float()  # convert target to the same nn output shape
            y_batch = y_batch.to(device)

            # model to eval mode
            model.eval()

            yhat = model(x_batch)
            val_loss = loss_fn(yhat, y_batch)
            cum_loss += val_loss / len(testloader)
            val_losses.append(val_loss.item())

            # Calculate test accuracy
            predictions = (torch.sigmoid(model(x_batch)) > 0.5).float()
            correct_test += (predictions == y_batch).sum().item()
            total_test += y_batch.size(0)

        epoch_test_losses.append(cum_loss)
        test_accuracy = correct_test / total_test
        test_accuracies.append(test_accuracy)
        print('Epoch : {}, val loss : {}, test accuracy: {:.2%}'.format(epoch + 1, cum_loss, test_accuracy))

        best_loss = min(epoch_test_losses)

        # save best model
        if cum_loss <= best_loss:
            best_model_wts = model.state_dict()
            torch.save(model.state_dict(), "best_model_res.pth")


plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.plot(torch.tensor(epoch_train_losses).cpu(), label="Train Loss")
plt.plot(torch.tensor(epoch_test_losses).cpu(), label="Test Loss")
plt.legend()
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Loss Curve")

plt.subplot(1, 2, 2)
plt.plot(train_accuracies, label="Train Accuracy")
plt.plot(test_accuracies, label="Test Accuracy")
plt.legend()
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Accuracy Curve")

plt.savefig("training_curve_res.png")
plt.show()