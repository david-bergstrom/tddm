# Original work: TimeGAN (Yoon et al., 2019)
# Copyright (c) 2019 Jinsung Yoon, Daniel Jarrett, Mihaela van der Schaar
# Licensed under the Apache License, Version 2.0
# 
# MODIFICATIONS:
# - Updated to load data from benchmark (2025)
# - Windows no longer reversed in time by default; --seed option (2026)

"""Time-series Generative Adversarial Networks (TimeGAN) Codebase.

Reference: Jinsung Yoon, Daniel Jarrett, Mihaela van der Schaar, 
"Time-series Generative Adversarial Networks," 
Neural Information Processing Systems (NeurIPS), 2019.

Paper link: https://papers.nips.cc/paper/8789-time-series-generative-adversarial-networks

Last updated Date: April 24th 2020
Code author: Jinsung Yoon (jsyoon0823@gmail.com)

-----------------------------

main_timegan.py

(1) Import data
(2) Generate synthetic data
(3) Evaluate the performances in three ways
  - Visualization (t-SNE, PCA)
  - Discriminative score
  - Predictive score
"""

## Necessary packages
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import random
import numpy as np
import warnings
warnings.filterwarnings("ignore")

# 1. TimeGAN model
from timegan import timegan
# 2. Data loading
from data_loading import benchmark_data_loading
# 3. Metrics
from metrics.discriminative_metrics import discriminative_score_metrics
from metrics.predictive_metrics import predictive_score_metrics
from metrics.visualization_metrics import visualization

import wandb


def main (args):
  """Main function for timeGAN experiments.
  
  Args:
    - data_name: sine, stock, or energy
    - seq_len: sequence length
    - Network parameters (should be optimized for different datasets)
      - module: gru, lstm, or lstmLN
      - hidden_dim: hidden dimensions
      - num_layer: number of layers
      - iteration: number of training iterations
      - batch_size: the number of samples in each batch
    - metric_iteration: number of iterations for metric computation
  
  Returns:
    - ori_data: original data
    - generated_data: generated synthetic data
    - metric_results: discriminative and predictive scores
  """
  print(f"TimeGAN,{args.data_name},{args.seq_len},seed={args.seed}")
  if args.seed is not None:
    random.seed(args.seed)
    np.random.seed(args.seed)
  ori_data = benchmark_data_loading(args.data_name, args.seq_len, reverse_time=args.reverse_time)
  print("Data loaded from benchmark loader with shape:", ori_data.shape)

  """
  ## Data loading
  if args.data_name in ['stock', 'energy']:
    # ori_data = real_data_loading(args.data_name, args.seq_len)
  elif args.data_name == 'sine':
    # Set number of samples and its dimensions
    no, dim = 10000, 5
    ori_data = sine_data_generation(no, args.seq_len, dim)
  """

  print(args.data_name + ' dataset is ready.')
    
  ## Synthetic data generation by TimeGAN
  # Set newtork parameters
  parameters = dict()  
  parameters['module'] = args.module
  parameters['hidden_dim'] = args.hidden_dim
  parameters['num_layer'] = args.num_layer
  parameters['iterations'] = args.iteration
  parameters['batch_size'] = args.batch_size
  parameters['dataset_str'] = args.data_name
  parameters['seq_len'] = args.seq_len
  parameters['seed'] = args.seed

  # logger = Logger('timegan', args.data_name, args.seq_len)
      
  generated_data = timegan(ori_data, parameters)
  print('Finish Synthetic Data Generation')

  wandb.init(
    project='tddpm',
    group='TimeGAN',
    config=args)
  
  seed_suffix = f"-seed{args.seed}" if args.seed is not None else ""
  direction = "" if args.reverse_time else "-forward"
  filename = f"{wandb.run.name}-timegan-synthetic-{args.data_name}-{args.seq_len}{direction}{seed_suffix}.npy"
  np.save(filename, generated_data)
  print("Saved synthetic data to:", filename)

  wandb.run.summary['synthetic_data_path'] = filename
  
  ## Performance metrics   
  # Output initialization
  metric_results = dict()
  
  """
  # 1. Discriminative Score
  discriminative_score = list()
  for _ in range(args.metric_iteration):
    temp_disc = discriminative_score_metrics(ori_data, generated_data)
    discriminative_score.append(temp_disc)
      
  metric_results['discriminative'] = np.mean(discriminative_score)
      
  # 2. Predictive score
  predictive_score = list()
  for tt in range(args.metric_iteration):
    temp_pred = predictive_score_metrics(ori_data, generated_data)
    predictive_score.append(temp_pred)   
      
  metric_results['predictive'] = np.mean(predictive_score)     
          
  # 3. Visualization (PCA and tSNE)
  visualization(ori_data, generated_data, 'pca')
  visualization(ori_data, generated_data, 'tsne')
  
  ## Print discriminative and predictive scores
  print(metric_results)
  """

  return ori_data, generated_data, metric_results


if __name__ == '__main__':  
  
  # Inputs for the main function
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--data_name',
      type=str)
  parser.add_argument(
      '--seq_len',
      help='sequence length',
      default=24,
      type=int)
  parser.add_argument(
      '--module',
      choices=['gru','lstm','lstmLN'],
      default='gru',
      type=str)
  parser.add_argument(
      '--hidden_dim',
      help='hidden state dimensions (should be optimized)',
      default=24,
      type=int)
  parser.add_argument(
      '--num_layer',
      help='number of layers (should be optimized)',
      default=3,
      type=int)
  parser.add_argument(
      '--iteration',
      help='Training iterations (should be optimized)',
      default=50000,
      type=int)
  parser.add_argument(
      '--batch_size',
      help='the number of samples in mini-batch (should be optimized)',
      default=128,
      type=int)
  parser.add_argument(
      '--metric_iteration',
      help='iterations of the metric computation',
      default=10,
      type=int)
  parser.add_argument(
      '--seed',
      help='random seed for numpy/random/tensorflow; appended to the output filename as -seed<N>',
      default=None,
      type=int)
  parser.add_argument(
      '--reverse_time',
      help='reverse every window in time, as an earlier version of this loader did; without it the output name gets -forward',
      action='store_true')
  
  args = parser.parse_args() 
  
  # Calls main function  
  ori_data, generated_data, metrics = main(args)
