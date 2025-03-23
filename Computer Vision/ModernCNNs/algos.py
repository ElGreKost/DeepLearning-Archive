import torch.cuda
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import FashionMNIST
from torch import nn
import torch.optim as optim
from tqdm.notebook import tqdm


def load_mnist(batch_size=128, resize=(96, 96)):
    # Load the FashionMNIST dataset
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,)), transforms.Resize(resize)])
    train_dataset = FashionMNIST(root='./data', train=True, transform=transform, download=True)
    test_dataset = FashionMNIST(root='./data', train=False, transform=transform)
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader


def init_cnn(module):
    if type(module) == nn.Linear or type(module) == nn.Conv2d:
        nn.init.xavier_uniform_(module.weight)


def fit(model: nn.Module, train_loader: DataLoader):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    input_data = next(iter(train_loader))[0].to(device)
    model.apply_init([input_data], init_cnn)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), 0.001)

    total_step = len(train_loader)
    for epoch in range(10):
        epoch_loss = 0.0
        for i, (images, labels) in tqdm(enumerate(train_loader), total=total_step):
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        # Print average epoch loss
        average_loss = epoch_loss / total_step
        print(f"Epoch [{epoch + 1}/10], Average Loss: {average_loss:.4f}")

    return model
