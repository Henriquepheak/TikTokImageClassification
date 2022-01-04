# Source:
# https://towardsdatascience.com/how-to-train-an-image-classifier-in-pytorch-and-use-it-to-perform-basic-inference-on-single-images-99465a1e9bf5


import numpy as np
import torch
from torch import nn
from torchvision import datasets, transforms, models
from torch.utils.data.sampler import SubsetRandomSampler
from PIL import Image
from torch import optim


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


def load_split_train_test(datadir, valid_size=.2):
    # Create image transformer to resize each image to the same size
    train_transforms = transforms.Compose([transforms.Resize((512, 512)),
                                           transforms.ToTensor()])

    test_transforms = transforms.Compose([transforms.Resize((512, 512)),
                                          transforms.ToTensor()
                                          ])

    # Create dataloaders to load all the images
    train_data = datasets.ImageFolder(datadir,
                                      transform=train_transforms)
    test_data = datasets.ImageFolder(datadir,
                                     transform=test_transforms)

    # Randomize order of dataset and divide into training and testing dataset
    num_train = len(train_data)
    indices = list(range(num_train))
    split = int(np.floor(valid_size * num_train))
    np.random.shuffle(indices)

    train_idx, test_idx = indices[split:], indices[:split]

    # PyTorch package which samples elements randomly from a given list of indices, without replacement.
    train_sampler = SubsetRandomSampler(train_idx)
    test_sampler = SubsetRandomSampler(test_idx)

    trainloader = torch.utils.data.DataLoader(train_data,
                                              sampler=train_sampler, batch_size=16)

    testloader = torch.utils.data.DataLoader(test_data,
                                             sampler=test_sampler, batch_size=16)

    return trainloader, testloader


def image_loader(loader, image_name):
    image = Image.open(image_name)
    image = loader(image).float()
    image = torch.tensor(image, requires_grad=True)
    image = image.unsqueeze(0)
    return image


def train(trainloader, testloader):
    # Epoch means one pass over the full training set.
    # Batch means that you use all your data to compute the gradient during one iteration.
    # Mini-batch means you only take a subset of all your data during one iteration.

    epochs = 30
    steps = 0
    running_loss = 0
    print_every = 100
    train_losses, test_losses = [], []

    optimizer = optim.Adam(model.fc.parameters(), lr=0.003)

    for epoch in range(epochs):
        for inputs, labels in trainloader:

            steps += 1

            # inputs, labels = inputs.to(device), labels.to(device)

            # Sets the gradients of all optimized torch.Tensor s to zero.
            # Clear the gradient to remove weights calculated in the previous mini-batch
            optimizer.zero_grad()

            logps = model(inputs)

            loss = criterion(logps, labels)

            # When we call loss.backward(), the whole graph is differentiated w.r.t. the loss, and all Variables in the graph will have their .grad Variable accumulated with the gradient.
            # backpropogate the error
            # Calculate the negative gradient for the loss function
            loss.backward()

            # Loops through paraemters and updates tensor values based on grad function
            optimizer.step()
            running_loss += loss.item()

            if steps % print_every == 0:
                test_loss = 0
                accuracy = 0

                # Set model to evaluate mode to calculate testing loss and accuarcy
                model.eval()

                # Remove the gradients during testing as it is only required during training
                with torch.no_grad():

                    # Loop through all images
                    for inputs, labels in testloader:
                        inputs, labels = inputs.to(device), labels.to(device)

                        logps = model(inputs)

                        batch_loss = criterion(logps, labels)
                        test_loss += batch_loss.item()

                        ps = torch.exp(logps)
                        top_p, top_class = ps.topk(1, dim=1)
                        equals = top_class == labels.view(*top_class.shape)
                        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

                train_losses.append(running_loss / len(trainloader))
                test_losses.append(test_loss / len(testloader))

                print(f"Epoch {epoch + 1}/{epochs}.. "
                      f"Train loss: {running_loss / print_every:.3f}.. "
                      f"Test loss: {test_loss / len(testloader):.3f}.. "
                      f"Test accuracy: {accuracy / len(testloader):.3f}")
                running_loss = 0
                model.train()
    torch.save(model.state_dict(), 'aerialmodel.pth')


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # Check for CUDA



    device = torch.device("cuda" if torch.cuda.is_available()
                          else "cpu")

    # Load pretrained resnet50 model
    model = models.resnet50(pretrained=True)

    # Add layers
    model.fc = nn.Sequential(nn.Linear(2048, 512),
                             nn.ReLU(),
                             nn.Dropout(0.2),
                             nn.Linear(512, 10),
                             nn.LogSoftmax(dim=1))

    model.to(device)

    # What is an activation function?
    # Used to convert the neural network outputs into a label

    imsize = 256
    loader = transforms.Compose([transforms.Scale((imsize, imsize)), transforms.ToTensor()])

    data_transforms = transforms.Compose([transforms.Resize((512, 512)),
                                          transforms.ToTensor()])


    data_dir = 'traindata/'

    trainloader, testloader = load_split_train_test(data_dir, .2)

    # Set model to evaluate mode to calculate testing loss and accuarcy
    model.eval()

    criterion = nn.NLLLoss()

    train(trainloader,testloader)

    # Remove the gradients during testing as it is only required during training
    with torch.no_grad():
        test_loss = 0
        accuracy = 0
        # Loop through all images
        for inputs, labels in testloader:
            inputs, labels = inputs.to(device), labels.to(device)

            logps = model(inputs)

            batch_loss = criterion(logps, labels)
            test_loss += batch_loss.item()

            ps = torch.exp(logps)
            top_p, top_class = ps.topk(1, dim=1)
            equals = top_class == labels.view(*top_class.shape)
            accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

        print(f"Test loss: {test_loss / len(testloader):.3f}.. "
              f"Test accuracy: {accuracy / len(testloader):.3f}")
