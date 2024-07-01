# general
import yaml
import argparse
import os
import subprocess
import anndata as ad
import torch
from termcolor import colored

# model
from utils import console_print, console_print_bold, \
                  load_config, parse_config, print_config, \
                  block_timer
from preprocessing import preprocess_adata
from unimodal_vae import get_X_matrix, split_dataset, \
                modded_vae


def main():

    # argparser
    console_print("Parsing arguments...", bold=True)
    with block_timer("Argument parsing"):    
        parser=argparse.ArgumentParser(description="Train single VAE for either RNA-seq or smRNA-seq")
        parser.add_argument("config_path", type=str, help="Path to the YAML configuration file")
        args=parser.parse_args()

    # load config file and get params
    console_print("Loading and parsing config file...", bold=True)
    with block_timer("Config loading and parsing"):
        print(f"Getting config file: {args.config_path}")
        config=load_config(args.config_path)
        config_dict=parse_config(config)\
        
        input_config=config_dict['inputs']
        preprocessing_config=config_dict['preprocessing']
        model_config=config_dict['model']

        print_config(config_dict)

    #### ADD SPECIFIC CONFIG_DICT NESTED AS VARS AS DONE IN PREPROCESSING STEP ####

    # create output directory
    console_print("Creating output directory...", bold=True)
    output_path=f"{config_dict['inputs']['output_dir']}/{config_dict['inputs']['model_name']}"
    print("Output path: ", output_path)
    if not os.path.exists(output_path):
        try:
            subprocess.run(['mkdir', '-p', output_path], check=True)
            print(f"Directory created: {output_path}\n")
        except subprocess.CalledProcessError as e:
            print(f"Failed to create directory {output_path}: {e}\n")
    else:
        print(f"Directory already exists: {output_path}\n")

    #### SAVE CONFIG FILE AS WELL ####

    # read input data
    console_print("Reading input data...", bold=True)
    with block_timer("Reading input data"):
        print("Input data: ", config_dict['inputs']['data'])
        adata=ad.read_h5ad(config_dict['inputs']['data'])

    # preprocess data
    console_print("Preprocess data...", bold=True)
    with block_timer("Preprocessing data"):
        filter_genes_threshold=preprocessing_config['filter_genes_threshold'] # filter low expressed genes
        norm_per_sample=preprocessing_config['norm_per_sample'] # sample norm
        log=preprocessing_config['log'] # log norm conditional on norm_per_sample
        isolate_top=preprocessing_config['isolate_top'] # done via seurat
        N=preprocessing_config['N'] # number of top genes to isolate
        adata=preprocess_adata(adata, filter_genes_threshold, norm_per_sample, log, isolate_top, N)

    # dataloading
    console_print("Loading data...", bold=True)
    with block_timer("Loading data"):
        X, X_raw=get_X_matrix(adata, scale_data=False)
        train_loader, val_loader, test_loader, train_data, val_data, test_data=split_dataset(X, X_raw, validation_split=model_config['validation_split'],\
                                                                                            test_split=model_config['test_split'], batch_size=model_config['batch_size'])
        
    # init model
    console_print("Initializing model...", bold=True)
    with block_timer("Initializing model"):
        input_dim=adata.n_vars  # number of genes
        hidden_dims=model_config['hidden_dims']
        latent_dim=model_config['latent_dim']
        loss_type=model_config['loss_type']
        learning_rate=model_config['learning_rate']

        # init model
        model=modded_vae(input_dim, hidden_dims, latent_dim, loss_type)

        # optimizer
        optimizer=torch.optim.Adam(model.parameters(), lr=learning_rate) # 1e-4 seemed to work quite well with sample normed

        # output model
        print(model)

        #### SAVE MODEL ARCHITECTURE IN A SEPARATE FILE ####




if __name__ == "__main__":
    main()

