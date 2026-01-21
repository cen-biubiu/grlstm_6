from GRLSTM_Model import GRLSTM
from pars_args import args
from Data_Loader import TrainValueDataLoader
from Trainer import Trainer
from logg import setup_logger

import numpy as np
import torch
import random
torch.cuda.empty_cache()

import torch
print("Visible GPUs:", torch.cuda.device_count())
for i in range(torch.cuda.device_count()):
    print(i, torch.cuda.get_device_name(i))


def train():
    if args.gpu >= 0:
        device = torch.device(f'cuda:{args.gpu}')
        print(f"Using GPU: cuda:{args.gpu}")
    else:
        device = torch.device('cpu')
        print("Using CPU")

    train_data_loader = TrainValueDataLoader(
        args.train_file, args.poi_file, args.batch_size)

    model = GRLSTM(args, device, batch_first=True)

    trainer = Trainer(
        model=model,
        train_data_loader=train_data_loader,
        val_data_loader=None,
        n_epochs=args.n_epochs,
        lr=args.lr,
        save_epoch_int=args.save_epoch_int,
        model_folder=args.save_folder,
        device=device,
        grad_accum_steps=args.grad_accum_steps,
        aug_weight=args.aug_weight, 

    )

    trainer.train()


if __name__ == '__main__':
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    setup_logger('GRLSTM_train.log')
    train()
