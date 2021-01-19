import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset
from torchvision import datasets, models, transforms

model_ft = models.resnet18(pretained=False)

DEVICE = 'cpu'
TARGET_NUM = 10


def get_model(target_num, isPretrained=False):

    model_ft = models.resnet18(pretained=isPretrained)
    model_ft.fc = nn.Linear(512, target_num)
    model_ft = model_ft.to(device)
    return model_ft


model_ft = get_model(TARGET_NUM, isPretrained=False)
optimizer = optim.SGD(model_ft.parameters(),lr=0.001, momentum=0.9)

criterion = nn.CrossEntropyLoss()

def train_model(model, criterion, optimizer, num_epochs=5, is_saved=False):
    best_acc = 0.0

    for epoch in range(num_epochs):
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            for i, (inputs, labels) in enumerate(image_dataloadeers[phase]):
                inputs = inputs.to(DEVICE)
                labels = labels.to(DEVICE)

                optimizer.zero_grad()

                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)

                loss = criterion(outputs, labels)
                print('  loaders:{}kaime'.format(i+1), '  loss{}'.format(loss))

                if phase == 'train':
                    loss.backward()
                    optimizer.step()

                running_loss += loss.item()* inputs.size(0)
                running_corrects = torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_size[phase]
            epoch_acc = running_corrects.double() / dataset_size[phase]
            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                if(is_saved):
                    torch.save(model.state_dict(), './oroginal_model{}.pth'.format(epoch))
    print('Best val Acc {:4f}'.format(best_acc))
