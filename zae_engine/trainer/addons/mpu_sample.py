import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel


def get_args_parser():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--epoch", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--port", type=int, default=8888)
    parser.add_argument("--root", type=str, default="./cifar")
    parser.add_argument("--local_rank", type=int)
    return parser


def init_distributed_training(rank, opts):
    # 1. setting for distributed training
    opts.rank = rank
    opts.gpu = opts.rank % torch.cuda.device_count()
    local_gpu_id = int(opts.gpu_ids[opts.rank])
    torch.cuda.set_device(local_gpu_id)

    if opts.rank is not None:
        print("Use GPU: {} for training".format(local_gpu_id))

    # 2. init_process_group
    torch.distributed.init_process_group(
        backend="nccl", init_method="tcp://127.0.0.1:" + str(opts.port), world_size=opts.ngpus_per_node, rank=opts.rank
    )

    # if put this function, the all processes block at all.
    torch.distributed.barrier()

    # convert print fn iif rank is zero
    setup_for_distributed(opts.rank == 0)
    print("opts :", opts)


def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__

    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop("force", False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


def main(rank, opts):
    init_distributed_training(rank, opts)
    local_gpu_id = opts.gpu

    tmp_in = torch.zeros((4))
    tmp_out = torch.ones((1))
    train_set = [[tmp_in, tmp_out]] * 16

    train_sampler = DistributedSampler(dataset=train_set, shuffle=True)

    batch_sampler_train = torch.utils.data.BatchSampler(train_sampler, opts.batch_size, drop_last=True)
    train_loader = DataLoader(train_set, batch_sampler=batch_sampler_train, num_workers=opts.num_workers)

    model = nn.Sequential(nn.Linear(4, 1))
    model = model.cuda(local_gpu_id)
    model = DistributedDataParallel(module=model, device_ids=[local_gpu_id])

    criterion = torch.nn.CrossEntropyLoss().to(local_gpu_id)
    optimizer = torch.optim.SGD(params=model.parameters(), lr=0.01, weight_decay=0.0005, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer=optimizer)

    print(f"[INFO] : 학습 시작")
    for epoch in range(opts.epoch):

        model.train()
        train_sampler.set_epoch(epoch)

        for i, (tmp_in, tmp_out) in enumerate(train_loader):

            tmp_in = tmp_in.to(local_gpu_id)
            tmp_out = tmp_out.to(local_gpu_id)
            # tmp_in = torch.zeros((2, 3, 4))
            # tmp_out = torch.ones((2, 3, 1))
            outputs = model(tmp_in)
            optimizer.zero_grad()
            loss = F.mse_loss(outputs, tmp_out)
            loss.backward()
            optimizer.step()
            scheduler.step()

            model.eval()
            with torch.no_grad():
                result = model(tmp_in)

        print(f"[INFO] : {epoch} 번째 epoch 완료")

    print(f"[INFO] : Distributed 학습 테스트완료")


if __name__ == "__main__":

    parser = argparse.ArgumentParser("Distributed training test", parents=[get_args_parser()])
    opts = parser.parse_args()
    opts.ngpus_per_node = torch.cuda.device_count()
    opts.gpu_ids = list(range(opts.ngpus_per_node))
    opts.num_workers = opts.ngpus_per_node * 4

    torch.multiprocessing.spawn(main, args=(opts,), nprocs=opts.ngpus_per_node, join=True)
