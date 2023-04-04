import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
import torch.nn.functional as F
from torch.optim import Adam
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from torchvision.models import resnet18, ResNet18_Weights

batch_size = 100
num_epochs = 5

transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                         shuffle=False)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# Load the ResNet18 model with the ImageNet pre-trained weights
resnet18_model = resnet18(weights = ResNet18_Weights.DEFAULT)

#get number of features from model, and use in next line to ensure input feature # not changed
num_features = resnet18_model.fc.in_features #512
#Resnet18 fully-connected layer has output of 1000 classes, we want to change to 10 classes for our dataset
resnet18_model.fc = nn.Linear(num_features, 10)
resnet18_model = resnet18_model.to(device)

#optimizer
optimizer = Adam(resnet18_model.parameters(), lr=3e-4, weight_decay=0.0001)

#loss function
loss_fn = CrossEntropyLoss()

train_loss = []
test_loss = []
train_acc = []
test_acc = []

#train model
for epoch in range(num_epochs):
    running_loss = 0.0
    running_corrects = 0.0
    total = 0

    # train
    resnet18_model.train()

    for xtrain, ytrain in trainloader:
        # for training of every batch, optimizer set to zero gradient so that past evalutions dont affect current
        optimizer.zero_grad() 

        # move to gpu b/c input and model need to be on same device
        xtrain = xtrain.to(device)
        train_probability = resnet18_model(xtrain)

        loss = loss_fn(train_probability, ytrain)
        loss.backward()
        optimizer.step()
        # training ends

        # calculate training loss
        running_loss += loss.item() * xtrain.size(0)

        # calculating training accuracy
        # comparing predicted values with actual values, count how many are same
        train_prediction = torch.max(train_probability, 1).indices
        running_corrects += int(torch.sum(train_prediction == ytrain)) 

    epoch_train_loss = 100* (running_loss / len(trainset))
    epoch_train_acc = 100 * (running_corrects / len(trainset))
    train_acc.append(epoch_train_acc)
    train_loss.append(epoch_train_loss)

           
    # evaluate model
    resnet18_model.eval()
    running_loss = 0.0
    running_corrects = 0.0
    total = 0
    with torch.no_grad():
        for xtest, ytest in testloader:

            xtest = xtest.to(device)
            test_prob = resnet18_model(xtest)
            test_prob = test_prob.cpu()

            # calculate test loss
            running_loss += loss.item() * xtest.size(0)

            # calculate test accuracy
            test_pred = torch.max(test_prob,1).indices
            running_corrects += int(torch.sum(test_pred == ytest))

        epoch_test_loss = 100*(running_loss / len(testset))
        epoch_test_acc = 100*(running_corrects / len(testset))
        test_loss.append(epoch_test_loss)
        test_acc.append(epoch_test_acc)   

    print("Epoch: {}, Train Loss: {:.2f}% , Train accuracy: {:.2f}%, Test Loss: {:.2f}%, Test accuracy: {:.2f}%"
          .format(epoch+1, epoch_train_loss, epoch_train_acc, epoch_test_loss, epoch_test_acc))        


# plot loss and accuracy curves
plt.figure(figsize=(10, 5))
plt.plot(train_loss, label='Train')
plt.plot(test_loss, label='Test')
plt.title('Loss curve')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

plt.figure(figsize=(10, 5))
plt.plot(train_acc, label='Train')
plt.plot(test_acc, label='Test')
plt.title('Accuracy curve')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()