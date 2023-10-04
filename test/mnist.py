# Source: https://pytorch.org/tutorials/intermediate/dist_tuto.html

import math
from random import Random

import torch
import torch.nn as nn
import torch.nn.functional as F
import valohai
from torch import optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

device = torch.device("cuda")
print('Current cuda device ', torch.cuda.current_device())


class Net(nn.Module):

    def __init__(self):

        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output


# class Partition:
#
#     def __init__(self, data, index):
#         self.data = data
#         self.index = index
#
#     def __len__(self):
#         return len(self.index)
#
#     def __getitem__(self, index):
#         data_idx = self.index[index]
#         return self.data[data_idx]
#
#
# class DataPartitioner:
#
#     def __init__(self, dataset, sizes=None, seed=1234):
#         if sizes is None:
#             sizes = [0.7, 0.2, 0.1]
#         self.data = dataset
#         self.partitions = []
#         rng = Random()
#         rng.seed(seed)
#         data_len = len(dataset)
#         indexes = [x for x in range(0, data_len)]
#         rng.shuffle(indexes)
#         for frac in sizes:
#             part_len = int(frac * data_len)
#             self.partitions.append(indexes[0:part_len])
#             indexes = indexes[part_len:]
#
#     def use(self, partition):
#         return Partition(self.data, self.partitions[partition])


def partition_dataset():
    dataset = datasets.MNIST(
        root='./data',
        train=True,
        download=True,
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ]),
    )
    # size = dist.get_world_size()
    # bsz = int(128 / float(size))
    # partition_sizes = [1.0 / size for _ in range(size)]
    # partition = DataPartitioner(dataset, partition_sizes)
    # partition = partition.use(dist.get_rank())
    train_set = DataLoader(
        dataset,
        batch_size=64,
        shuffle=True,
    )
    return train_set, 64


# def average_gradients(model):
#     size = float(dist.get_world_size())
#     for param in model.parameters():
#         dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM)
#         param.grad.data /= size


def run():
    print('in run on device ', device)
    torch.manual_seed(1234)
    train_set, bsz = partition_dataset()
    # train_set.to(device)
    model = Net().to(device)
    optimizer = optim.SGD(
        model.parameters(),
        lr=0.01,
        momentum=0.5,
    )
    num_batches = math.ceil(len(train_set.dataset) / float(bsz))
    for epoch in range(2):
        print('started epoch ', epoch)
        epoch_loss = 0.0
        for data, target in train_set:
            optimizer.zero_grad()
            output = model(data.to(device))
            loss = F.nll_loss(output, target.to(device))
            epoch_loss += loss.item()
            loss.backward()
            # average_gradients(model)
            optimizer.step()
        # print(f'Rank {dist.get_rank()}, epoch {epoch}: {epoch_loss / num_batches}')

    torch.save(model.state_dict(), '/valohai/outputs/model_weights.pth')


# def init(master_url, my_rank, world_size, fn):
#     dist.init_process_group(init_method=master_url, rank=my_rank, world_size=world_size, backend='nccl')
#     print('init process group')
#     fn(my_rank, world_size)


if __name__ == '__main__':
    run()
    # master_port = 1235
    # master_ip = valohai.distributed.master().primary_local_ip
    # url = f"tcp://{master_ip}:{master_port}"
    # print('url', url)
    # size = valohai.distributed.required_count
    # rank = valohai.distributed.me().rank
    #
    # print('rank ', rank)
    # print('size ', size)
    #
    # mp.set_start_method('spawn')
    # p = mp.Process(target=init, args=(url, rank, size, run))
    # p.start()
    # print('p start')
    # p.join()
    # print('p join')

