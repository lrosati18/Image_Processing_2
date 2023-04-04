import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
# from tqdm import tqdm

# Define hyperparameters
num_epochs = 5
batch_size = 100
learning_rate = 0.001

# Define the transforms to apply to the data
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

# Load the CIFAR10 dataset
train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform_train)

test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform_test)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,
                                          shuffle=True)

test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size,
                                         shuffle=False)

# Define the model
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv4 = nn.Conv2d(128, 256, 3, padding=1)
        self.conv5 = nn.Conv2d(256, 512, 3, padding=1)

        x = torch.randn(3,50,50).view(-1,3,50,50)
        self._to_linear = None
        self.convs(x)

        self.fc = nn.Linear(self._to_linear, 10) #flattening.

    def convs(self, x):
        # max pooling over 2x2
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv3(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv4(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv5(x)), (2, 2))


        if self._to_linear is None:
            self._to_linear = x[0].shape[0]*x[0].shape[1]*x[0].shape[2]
        return x
    
    def forward(self, x):
        x = self.convs(x)
        x = x.view(-1, self._to_linear)  # .view is reshape ... this flattens X before 
        x = self.fc(x) # bc this is our output layer. No activation here.
        return x

net = Net()
# print(net)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=learning_rate)

# Train the model
for epoch in range(num_epochs):
    running_loss = 0.0
    for i, data in enumerate(train_loader, 0):
        inputs, labels = data
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        if i % 100 == 99:
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 100))
            running_loss = 0.0

print('Finished Training')

# Set the model to evaluation mode
net.eval()

# Initialize counters for the number of correct and total predictions
total = 0
correct = 0

# Disable gradient computation during evaluation
with torch.no_grad():
    # Loop over the test set
    for data in test_loader:
        inputs, labels = data
        # Forward pass
        outputs = net(inputs)
        # Get the predicted class
        _, predicted = torch.max(outputs.data, 1)
        # Update the counters
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

# Compute the accuracy
accuracy = 100 * correct / total
print('Accuracy on the test set: {:.2f}%'.format(accuracy))

# Set the model to evaluation mode
net.eval()

# Initialize counters for the number of correct and total predictions
total = 0
correct = 0

# Disable gradient computation during evaluation
with torch.no_grad():
    # Loop over the test set
    for data in test_loader:
        inputs, labels = data
        # Forward pass
        outputs = net(inputs)
        # Get the predicted class
        _, predicted = torch.max(outputs.data, 1)
        # Update the counters
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

# Compute the accuracy
accuracy = 100 * correct / total
print('Accuracy on the test set: {:.2f}%'.format(accuracy))