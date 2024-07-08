# general
import yaml
import argparse
import os
import subprocess
import anndata as ad
import torch
from termcolor import colored, cprint
# print(colored('test', 'green', attrs=['bold']))
import io
import sys
import time
# from simple_colors import *
# print(green('test', 'bold'))
# os.system('color')


# model
from utils import load_config, parse_config, print_config, block_timer
from _logger import set_log_file, console_print, save_log, save_model_architecture
from preprocessing import preprocess_adata
from unimodal_vae import get_X_matrix, split_dataset, modded_vae, train_vae, validate_vae, recon_corr, calculate_mae_list, \
                         select_and_plot_samples, plot_journal_histogram, plot_and_save_loss_wt_val, plot_training_validation_loss


#### REFORMAT ALL PRINT STATEMENTS FOR LOGGING ####

def main():

    # argparser
    # console_print("Parsing arguments...", bold=True)
    cprint("Parsing arguments...", "magenta", attrs=["bold"], file=sys.stderr)
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
        output_path=f"{config_dict['inputs']['output_dir']}/{config_dict['inputs']['model_name']}"
        preprocessing_config=config_dict['preprocessing']
        model_config=config_dict['model']

        set_log_file(f"{output_path}/run.log")

        print_config(config_dict)

    #### ADD SPECIFIC CONFIG_DICT NESTED AS VARS AS DONE IN PREPROCESSING STEP ####

    # create output directory
    # console_print("Creating output directory...", bold=True)
    cprint("Creating output directory...", "magenta", attrs=["bold"], file=sys.stderr)
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
    # console_print("Reading input data...", bold=True)
    cprint("Reading input data...", "magenta", attrs=["bold"], file=sys.stderr)
    with block_timer("Reading input data"):
        print("Input data: ", config_dict['inputs']['data'])
        adata=ad.read_h5ad(config_dict['inputs']['data'])

    # preprocess data
    # console_print("Preprocess data...", bold=True)
    cprint("Preprocess data...", "magenta", attrs=["bold"], file=sys.stderr)
    with block_timer("Preprocessing data"):
        filter_genes_threshold=preprocessing_config['filter_genes_threshold'] # filter low expressed genes
        norm_per_sample=preprocessing_config['norm_per_sample'] # sample norm
        log=preprocessing_config['log'] # log norm conditional on norm_per_sample
        isolate_top=preprocessing_config['isolate_top'] # done via seurat
        N=preprocessing_config['N'] # number of top genes to isolate
        adata=preprocess_adata(adata, filter_genes_threshold, norm_per_sample, log, isolate_top, N)

    #### SAVE SAMPLE AND GENE LEVEL MEAN PLOTS ####

    # dataloading
    # console_print("Loading data...", bold=True)
    cprint("Loading data...", "magenta", attrs=["bold"], file=sys.stderr)
    with block_timer("Loading data"):
        X, X_raw=get_X_matrix(adata, scale_data=False)
        train_loader, val_loader, test_loader, train_data, val_data, test_data=split_dataset(X, X_raw, validation_split=model_config['validation_split'],\
                                                                                            test_split=model_config['test_split'], batch_size=model_config['batch_size'])
    # init model
    # console_print("Initializing model...", bold=True)
    cprint("Initializing model...", "magenta", attrs=["bold"], file=sys.stderr)
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

        # save architecture
        save_model_architecture(model, f"{output_path}/model_architecture.txt")

    #### SAVE MODEL ARCHITECTURE IN A SEPARATE FILE ####
    
    # train model
    # console_print("Training model...", bold=True)
    cprint("Training model...", "magenta", attrs=["bold"], file=sys.stderr)
    with block_timer("Training model"):
        epochs=model_config['epochs']
        per_epoch_avg_loss, per_epoch_val_avg_loss=train_vae(model, train_loader, optimizer, epochs, with_validation=True, val_loader=val_loader)

        print("Plotting training and validation loss...")
        # plot_and_save_loss_wt_val(per_epoch_avg_loss, per_epoch_val_avg_loss, \
        #                           f"training_validation_loss.png", f"training_loss.json", f"validation_loss.json", save=False, output_path=output_path)
        plot_training_validation_loss(per_epoch_avg_loss, per_epoch_val_avg_loss, title="Training and Validation Loss", save=True, output_path=output_path)

    # run w/ validation/test sets
    # console_print("Testing model...", bold=True)
    cprint("Testing model...", "magenta", attrs=["bold"], file=sys.stderr)
    with block_timer("Testing model"):

        val_loss, val_recons=validate_vae(model, val_loader)
        print("Final validation loss: ", val_loss)

        test_loss, test_recons=validate_vae(model, test_loader)
        print("Final test loss: ", test_loss)
        print('\n')

        print("Making correlation plots...")
        # get correlations for the test/val/train data
        spear_corr_list, pear_corr_list, ccc_list=recon_corr(test_data, test_recons) 
        mae_list=calculate_mae_list(test_data, test_recons)

        # plot correlations
        select_and_plot_samples(ccc_list, test_data, test_recons, tb_N=5, nrows=2, ncols=5, save=True, output_path=output_path, filename='recon_ccc_corr.png') # SPECIFY CORR LIST
        print('\n')

        print("Making error distribution plots...")
        plot_journal_histogram(ccc_list, "Lin's CCC Distribution", bins='auto', color='#4c72b0', figsize=(8, 6), save=True, output_path=output_path, filename='ccc_test_recons_hist.png')
        plot_journal_histogram(pear_corr_list, "Pearson Corr. Distribution", bins='auto', color='#4c72b0', figsize=(8, 6), save=True, output_path=output_path, filename='pearson_test_recons_hist.png')
        plot_journal_histogram(spear_corr_list, "Spearman Corr. Distribution", bins='auto', color='#4c72b0', figsize=(8, 6), save=True, output_path=output_path, filename='spearman_test_recons_hist.png')
        plot_journal_histogram(mae_list, "MAE Distribution", bins='auto', color='#4c72b0', figsize=(8, 6), save=True, output_path=output_path, filename='mae_test_recons_hist.png')


    # save console outs
    save_log()

if __name__ == "__main__":
    main()

