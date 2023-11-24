import torch
import torchvision
from torch.utils import data
from torchvision import transforms
# 通过ToTensor实例将图像数据从PIL类型变换成32位浮点数格式，
# 并除以255使得所有像素的数值均在0到1之间

def getData():
    trans = transforms.ToTensor()
    mnist_train = torchvision.datasets.FashionMNIST(root="../../fashion-mnist-master", train=True,
                                                    transform=trans, download=False)
    mnist_test = torchvision.datasets.FashionMNIST(root="../../fashion-mnist-master", train=False,
                                                   transform=trans, download=False)
    # print(len(mnist_train), len(mnist_test))
    # print(len(mnist_test[0]))
    return mnist_train,mnist_test

if __name__ == '__main__':
    getData()