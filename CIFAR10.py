import torchvision
from torchvision import transforms

def getData():
    trans = transforms.ToTensor()
    train_dataset = torchvision.datasets.CIFAR10(root='../../CIFAR10', train=True, transform=trans, download=False)
    test_dataset = torchvision.datasets.CIFAR10(root='../../CIFAR10', train=False, transform=trans, download=False)
    return train_dataset,test_dataset


if __name__ == '__main__':
    train_dataset, test_dataset=getData()
    print(len(train_dataset))