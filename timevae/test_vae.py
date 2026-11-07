
import os, warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}
warnings.filterwarnings('ignore') 

# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # disabling gpu usage because my cuda is corrupted, needs to be fixed. 

import argparse
import random
import sys
import numpy as np
import pandas as pd
import time
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from vae_dense_model import VariationalAutoencoderDense as VAE_Dense
from vae_conv_model import VariationalAutoencoderConv as VAE_Conv
from vae_conv_I_model import VariationalAutoencoderConvInterpretable as TimeVAE
import utils
sys.path.append("../")
from data import load_dataset_from_str, minmaxscale
import wandb
import time


def parse_arguments():
    parser = argparse.ArgumentParser(description='Argument Parser')
    parser.add_argument('--dataset', type=str, help='Dataset name', required=True)
    parser.add_argument('--seq_len', type=int, help='Sequence length', required=True)
    parser.add_argument('--epochs', type=int, help='Number of training epochs', default=500)
    parser.add_argument('--seed', type=int, default=None, help='random seed for random/numpy/tensorflow; appended to the output filename as -seed<N>')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_arguments()
    
    data_dir = './'
    # ----------------------------------------------------------------------------------
    # choose model
    vae_type = 'timeVAE'           # vae_dense, vae_conv, timeVAE
    # ----------------------------------------------------------------------------------
    # read data    
    dataset = args.dataset
    perc_of_train_used = 100     # 5, 10, 20, 100    
    valid_perc = 0.1
    seq_len = args.seq_len

    print(f"TimeVAE,{dataset},{seq_len},seed={args.seed}")
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        tf.random.set_seed(args.seed)
    wandb.init(
        project='tddpm',
        group='timevae',
        config={'dataset': dataset, 'seq_len': seq_len, 'seed': args.seed}
    )

    # full_train_data = utils.get_training_data(data_dir + input_file)
    data = load_dataset_from_str(dataset_str=dataset, prefix='../data/')
    data = minmaxscale(data)

    train_data = data

    N, T, D = train_data.shape   
    print('data shape:', N, T, D) 

    # ----------------------------------------------------------------------------------
    # further split the training data into train and validation set - same thing done in forecasting task
    # N_train = int(N * (1 - valid_perc))
    # N_valid = N - N_train

    # # Shuffle data
    np.random.shuffle(train_data)

    # train_data = full_train_data[:N_train]
    # valid_data = full_train_data[N_train:]   
    # print("train/valid shapes: ", train_data.shape, valid_data.shape)    
    # import pdb; pdb.set_trace()
    
    # ----------------------------------------------------------------------------------
    # min max scale the data    
    scaler = utils.MinMaxScaler()        
    scaled_train_data = scaler.fit_transform(train_data)

    # scaled_valid_data = scaler.transform(valid_data)
    # joblib.dump(scaler, 'scaler.save')  

    # ----------------------------------------------------------------------------------
    # instantiate the model     
    
    latent_dim = 8

    if vae_type == 'vae_dense': 
        vae = VAE_Dense( seq_len=T,  feat_dim = D, latent_dim = latent_dim, hidden_layer_sizes=[200,100], )
    elif vae_type == 'vae_conv':
        vae = VAE_Conv( seq_len=T,  feat_dim = D, latent_dim = latent_dim, hidden_layer_sizes=[100, 200] )
    elif vae_type == 'timeVAE':
        vae = TimeVAE( seq_len=T,  feat_dim = D, latent_dim = latent_dim, hidden_layer_sizes=[50, 100, 200],        #[80, 200, 250] 
                reconstruction_wt = 3.0,
                # ---------------------
                # disable following three arguments to use the model as TimeVAE_Base. Enabling will convert to Interpretable version.
                # Also set use_residual_conn= False if you want to only have interpretable components, and no residual (non-interpretable) component. 
                
                # trend_poly=2, 
                # custom_seas = [ (6,1), (7, 1), (8,1), (9,1)] ,     # list of tuples of (num_of_seasons, len_per_season)
                # use_scaler = True,
                
                #---------------------------
                use_residual_conn = True
            )   
    else:  raise Exception('wut')

    
    vae.compile(optimizer=Adam())

    early_stop_loss = 'loss'
    early_stop_callback = EarlyStopping(monitor=early_stop_loss, min_delta = 1e-1, patience=10) 
    reduceLR = ReduceLROnPlateau(monitor='loss', factor=0.1, patience=5)

    start_training = time.time()

    vae.fit(
        scaled_train_data, 
        batch_size = 32,
        epochs=args.epochs,
        shuffle = True,
        callbacks=[early_stop_callback, reduceLR],
        verbose = 1
    )

    stop_training = time.time()
    wandb.run.summary['training_time'] = stop_training - start_training

    # ----------------------------------------------------------------------------------    
    # save model 
    model_dir = './model/'
    file_pref = f'vae_{vae_type}_{dataset}_perc_{perc_of_train_used}_iter_{0}_'
    vae.save(model_dir, file_pref)
    wandb.run.summary["model_weights_prefix"] = file_pref
    
    # # ----------------------------------------------------------------------------------
    # draw random prior samples
    num_samples = min(len(train_data), 10000)
    # print("num_samples: ", num_samples)

    start_sampling = time.time()
    samples = vae.get_prior_samples(num_samples=num_samples)
    stop_sampling = time.time()
    wandb.run.summary['sampling_time'] = stop_sampling - start_sampling

    # The model generates in the scaler's space; map back to the data frame before saving
    samples = scaler.inverse_transform(samples)

    seed_suffix = f"-seed{args.seed}" if args.seed is not None else ""
    filename = f"timevae-synthetic-{dataset}-{seq_len}-experiment-{wandb.run.name}{seed_suffix}.npy"
    np.save(filename, samples)
    wandb.run.summary['synthetic_data_path'] = filename
