
from torchvision import transforms

data_transforms = {
    'train':transforms.Compose([
        transforms.Resize(256),
        transforms.ToTensor(),
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.ToTensor(),
    ]),
}
