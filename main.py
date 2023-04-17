from collections import OrderedDict
import torch
import torch.nn as nn
import numpy as np
import logging
import json
import argparse
import os
import copy
import random
import torch.optim as optim
import datetime
import tqdm
from option import get_args

from utils import get_dataset
from client import Client

from models.lenet import LeNet


def main():
    args = get_args()
    experiment(args)


def experiment(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.cuda.manual_seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # data split
    train_dataset, user_groups = get_dataset(args)

    # model initialization
    global_model = LeNet()
    global_model_parameter = global_model.state_dict()
    global_model_parameter_tmp = global_model.state_dict()
    client_model = [LeNet() for i in range(args.num_users)]

    for round in range(args.rounds):
        for idx in range(args.num_users):
            # load the global model for current client
            client_model[idx].load_state_dict(global_model.state_dict())
            client = Client(args, train_dataset, user_groups[idx], device, client_model[idx])
            # local update
            model_parameter = client.model_update(args, client.model)

            # aggregate
            if idx == 0:
                for key in model_parameter:
                    global_model_parameter_tmp[key] = model_parameter[key]
            else:
                for key in model_parameter:
                    global_model_parameter_tmp[key] += model_parameter[key]
            
            print(f'client {idx} finishes training')
        
        print(f'round {round} finishes!')
        global_model_parameter = copy.deepcopy(global_model_parameter_tmp)
        
        # model parameters averaging
        for key in global_model_parameter:
            global_model_parameter[key] /= args.num_users
        
        # global model update
        global_model.load_state_dict(global_model_parameter)
        

if __name__ == '__main__':
    main()