import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import valohai


def run(my_rank, world_size):
    group = dist.new_group(list(range(world_size)))
    tensor = torch.ones(1)
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM, group=group)
    print(f'Rank {my_rank} has data {tensor} on host {valohai.distributed.me().identity}')


def init(master_url, my_rank, world_size, fn):
    dist.init_process_group(init_method=master_url, rank=my_rank, world_size=world_size, backend='nccl')
    fn(my_rank, world_size)


if __name__ == '__main__':

    master_port = 1234
    master_ip = valohai.distributed.master().primary_local_ip
    url = f"tcp://{master_ip}:{master_port}"

    size = valohai.distributed.required_count
    rank = valohai.distributed.me().rank

    mp.set_start_method('spawn')
    p = mp.Process(target=init, args=(url, rank, size, run))
    p.start()
    p.join()