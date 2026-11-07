import os
import torch
import argparse
import numpy as np

from engine.logger import Logger
from engine.solver import Trainer
from Data.build_dataloader import build_dataloader, build_dataloader_cond
from Models.interpretable_diffusion.model_utils import unnormalize_to_zero_to_one
from Utils.io_utils import load_yaml_config, seed_everything, merge_opts_to_config, instantiate_from_config
import torch
import sys
import wandb
import time


def parse_args():
    parser = argparse.ArgumentParser(description='PyTorch Training Script')
    parser.add_argument('--name', type=str, default=None)

    parser.add_argument('--config_file', type=str, default=None, 
                        help='path of config file')
    parser.add_argument('--output', type=str, default='OUTPUT', 
                        help='directory to save the results')
    parser.add_argument('--tensorboard', action='store_true', 
                        help='use tensorboard for logging')

    # args for random

    parser.add_argument('--cudnn_deterministic', action='store_true', default=False,
                        help='set cudnn.deterministic True')
    parser.add_argument('--seed', type=int, default=12345, 
                        help='seed for initializing training.')
    parser.add_argument('--gpu', type=int, default=None,
                        help='GPU id to use. If given, only the specific gpu will be'
                        ' used, and ddp will be disabled')
    
    # args for training
    parser.add_argument('--benchmark', action='store_true', default=False, help='Traing and then generate unconditional dataset.')
    parser.add_argument('--train', action='store_true', default=False, help='Train or Test.')
    parser.add_argument('--sample', type=int, default=0, 
                        choices=[0, 1], help='Condition or Uncondition.')
    parser.add_argument('--max_samples', type=int, default=10000)
    parser.add_argument('--mode', type=str, default='infill',
                        help='Infilling or Forecasting.')
    parser.add_argument('--milestone', type=int, default=10)

    parser.add_argument('--missing_ratio', type=float, default=0., help='Ratio of Missing Values.')
    parser.add_argument('--pred_len', type=int, default=0, help='Length of Predictions.')
    
    # args for modify config
    parser.add_argument('opts', help='Modify config options using the command-line',
                        default=None, nargs=argparse.REMAINDER)  

    args = parser.parse_args()
    args.save_dir = os.path.join(args.output, f'{args.name}')

    return args

def main():
    args = parse_args()

    if args.seed is not None:
        seed_everything(args.seed)

    if args.gpu is not None:
        torch.cuda.set_device(args.gpu)
    
    config = load_yaml_config(args.config_file)
    config = merge_opts_to_config(config, args.opts)

    logger = Logger(args)
    logger.save_config(config)

    model = instantiate_from_config(config['model']).cuda()
    if args.sample == 1 and args.mode in ['infill', 'predict']:
        test_dataloader_info = build_dataloader_cond(config, args)
    dataloader_info = build_dataloader(config, args)
    trainer = Trainer(config=config, args=args, model=model, dataloader=dataloader_info, logger=logger)

    if args.benchmark:
        dataset_name = config['dataloader']['train_dataset']['params']['name']
        seq_len = config['dataloader']['train_dataset']['params']['window']
        wandb.init(
            project='tddpm',
            group='diffusionts',
            config={'dataset': dataset_name, 'seq_len': seq_len}
        )
        try:
            start_training = time.time()
            trainer.train()
            stop_training = time.time()
            wandb.run.summary['training_time'] = stop_training - start_training
            dataset = dataloader_info['dataset']
            start_sampling = time.time()
            max_samples = args.max_samples
            n_samples = min(len(dataset), max_samples)
            samples = trainer.sample(num=n_samples, size_every=2001, shape=[dataset.window, dataset.var_num])
            if dataset.auto_norm:
                samples = unnormalize_to_zero_to_one(samples)
            stop_sampling = time.time()
            wandb.run.summary['sampling_time'] = stop_sampling - start_sampling
            filename = f'diffusionts-synthetic-{dataset_name}-{seq_len}-experiment.npy'
            full_path = os.path.join(args.save_dir, filename)
            np.save(full_path, samples)
            wandb.run.summary['synthetic_data_path'] = full_path
        except torch.cuda.OutOfMemoryError as e:
            print(f"Out of memory error: {e}")
        except KeyboardInterrupt as e:
            print(f"KeyboardInterrupt")
    elif args.train:
        trainer.train()
    elif args.sample == 1 and args.mode in ['infill', 'predict']:
        trainer.load(args.milestone)
        dataloader, dataset = test_dataloader_info['dataloader'], test_dataloader_info['dataset']
        coef = config['dataloader']['test_dataset']['coefficient']
        stepsize = config['dataloader']['test_dataset']['step_size']
        sampling_steps = config['dataloader']['test_dataset']['sampling_steps']
        samples, *_ = trainer.restore(dataloader, [dataset.window, dataset.var_num], coef, stepsize, sampling_steps)
        if dataset.auto_norm:
            samples = unnormalize_to_zero_to_one(samples)
            dataset_name = config['dataloader']['test_dataset']['params']['name']
            seq_len = config['dataloader']['test_dataset']['params']['window']
            filename = f'diffusionts-synthetic-{dataset_name}-{seq_len}.npy'
            np.save(os.path.join(args.save_dir, filename), samples)
            print(f"Output saved to: {os.path.join(args.save_dir, filename)}")
    else:
        trainer.load(args.milestone)
        dataset = dataloader_info['dataset']
        samples = trainer.sample(num=len(dataset), size_every=2001, shape=[dataset.window, dataset.var_num])
        if dataset.auto_norm:
            samples = unnormalize_to_zero_to_one(samples)
        dataset_name = config['dataloader']['train_dataset']['params']['name']
        seq_len = config['dataloader']['train_dataset']['params']['window']
        filename = f'diffusionts-synthetic-{dataset_name}-{seq_len}.npy'
        np.save(os.path.join(args.save_dir, filename), samples)
        print(f"Output saved to: {os.path.join(args.save_dir, filename)}")

if __name__ == '__main__':
    main()
