import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
import copy
import numpy as np

from models.lenet import LeNet


class DatasetSplit(Dataset):
    """An abstract Dataset class wrapped around Pytorch Dataset class.
    """

    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = [int(i) for i in idxs]

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return torch.tensor(image), torch.tensor(label)


class Client(object):
    def __init__(self, args, dataset, idxs, device, model):
        self.args = args
        self.trainloader = self.get_dataloader(dataset, list(idxs))
        # self.criterion = args.criterion
        self.device = device
        self.model = model
    

    def get_dataloader(self, dataset, idxs):
        idxs_train = idxs[:int(1 * len(idxs))]
        # print(len(idxs_train))
        train_dataloader = DataLoader(DatasetSplit(dataset, idxs_train), self.args.batch_size, shuffle=True)
        return train_dataloader

    
    def model_update(self, args, model):
        model.to(self.device)
        model.train()

        if self.args.optimizer == 'sgd':
            optimizer = torch.optim.SGD(model.parameters(), lr=self.args.lr, momentum=0.5)
        elif self.args.optimizer == 'adam':
            optimizer = torch.optim.Adam(model.parameters(), lr=self.args.lr, weight_decay=1e-4)
        
        criterion = nn.CrossEntropyLoss().to(self.device)
        
        for epoch in range(self.args.epochs):
            for batch_idx, (images, labels) in enumerate(self.trainloader):
                images, labels = images.to(self.device), labels.to(self.device)
                optimizer.zero_grad()
                pred = model(images)
                loss = criterion(pred, labels)
                # print(f'the loss is {loss}')
                loss.backward()
                optimizer.step()

            print(f'epoch: {epoch}, loss: {loss}')
        
        return model.state_dict()
    
