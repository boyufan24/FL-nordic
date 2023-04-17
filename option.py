import argparse


def get_args():
    parser = argparse.ArgumentParser()
    # parser.add_argument('--model', type=str, default='cnn')
    parser.add_argument('--dataset', type=str, default='cifar10')
    parser.add_argument('--data_dir', type=str, default='./data/cifar10')
    parser.add_argument('--seed', type=int, default=2)
    parser.add_argument('--iid', type=bool, default=True)
    parser.add_argument('--rounds', type=int, default=10)
    parser.add_argument('--epochs', type=int, default=10)
    # parser.add_argument('--criterion', type=str, default='')
    # parser.add_argument('--device', type=str, default='gpu')
    # parser.add_argument('--partition', type=str, default='iid', help='choose iid or noniid')
    parser.add_argument('--num_users', type=int, default=10)
    # parser.add_argument('--num_rounds', type=int, default=10, help='communication rounds between clients and server')
    # parser.add_argument('--beta', type=float, default=0.3)
    # parser.add_argument('--sample_rate', type=float, default=1.0, help='the sample rate of client selection')
    parser.add_argument('--batch_size', type=int, default=32)
    # parser.add_argument('--alg', type=str, default='fedres')
    # parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--optimizer', type=str, default='sgd')
    # parser.add_argument('--log_dir', type=str, default='./logs')  
    # parser.add_argument('--log_name', type=str, default=None)
    # parser.add_argument('--mu', type=float, default=0.5)
    

    args = parser.parse_args()
    return args