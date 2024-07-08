import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
import tqdm as tqdm
import seaborn as sns
import torch
from torch.utils.data import DataLoader, TensorDataset, random_split
from torch import nn
import matplotlib.pyplot as plt
import torch.nn.functional as F
import json

# load data
def get_X_matrix(adata, scale_data=False):
    if scale_data:
        X = torch.tensor(adata.X).float()
    else:
        X = torch.tensor(adata.X.todense()).float()  # convert to dense tensor and float type if it's sparse
    print(X.shape)

    # get raw counts for reconstruction
    X_raw=torch.tensor(adata.layers['counts'].todense()).float()  

    return X, X_raw

# split dataset
def split_dataset(X, X_raw=None, validation_split=0.2, test_split=0.1, batch_size=32):
    num_samples = X.size(0)
    print("Total samples:", num_samples)
    val_size = int(num_samples * validation_split)
    test_size=int(num_samples*test_split)
    train_size = num_samples - val_size
    train_size-=test_size
    print("Train, val, test sizes:", train_size, val_size, test_size)

    if X_raw==None:
        tensor_data = torch.tensor(X, dtype=torch.float32)
        dataset = TensorDataset(tensor_data)
    else:
        tensor_data = torch.tensor(X, dtype=torch.float32)
        tensor_data_raw = torch.tensor(X_raw, dtype=torch.int32)
        dataset = TensorDataset(tensor_data, tensor_data_raw)

    # split data
    train_data, val_data, test_data = random_split(dataset, [train_size, val_size, test_size])
    print("Train, val, test sizes:", len(train_data), len(val_data), len(test_data))

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)
    test_loader=DataLoader(test_data, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader, train_data, val_data, test_data

# vae
class modded_vae(nn.Module):
    def __init__(self, input_dim, hidden_dims, latent_dim, loss_type='mse'):
        super(modded_vae, self).__init__()

        self.input_dim = input_dim
        self.loss_type = loss_type
        
        # encoder constructor
        """
        Notes:

        - Exclude dropout in decoder layers; shows improved performance and may help generative properties of the VAE. Potentially remove the last dropout layer from encoder.
        - Tune dropout_rate and other HPs w/ optuna.
    
        """
        dropout_rate=0.15 # around 0.15 to 0.2 works well 

        encoder_layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            encoder_layers.append(nn.Linear(prev_dim, h_dim))
            encoder_layers.append(nn.Dropout(dropout_rate)) # dropout added - only dropout layer
            encoder_layers.append(nn.ReLU()) # experiment with ELU and other activation functions
            # encoder_layers.append(nn.Dropout(dropout_rate)) # dropout added - remove here
            prev_dim = h_dim

        self.encoder = nn.Sequential(*encoder_layers)
        
        # latent distribution
        self.fc_mu = nn.Linear(hidden_dims[-1], latent_dim)
        self.fc_logvar = nn.Linear(hidden_dims[-1], latent_dim)

        # decoder constructor
        decoder_layers = []
        prev_dim = latent_dim
        for h_dim in reversed(hidden_dims):
            decoder_layers.append(nn.Linear(prev_dim, h_dim))
            decoder_layers.append(nn.Softplus())
            # decoder_layers.append(nn.Dropout(dropout_rate)) # dropout added after softplus, recon is still a final linear layer in the line below
            prev_dim = h_dim
        self.decoder = nn.Sequential(*decoder_layers)

        # recon and theta/pi for nb and zinb
        self.fc_recon = nn.Linear(prev_dim, input_dim)
        if loss_type in ['nb', 'zinb']:
            self.fc_theta = nn.Linear(prev_dim, input_dim)
        if loss_type == 'zinb':
            self.fc_pi = nn.Linear(prev_dim, input_dim)

    def encode(self, x):
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h = self.decoder(z)
        recon_x = self.fc_recon(h)
        
        theta = None
        pi = None
        
        if self.loss_type in ['nb', 'zinb']:
            theta = F.softplus(self.fc_theta(h)) + 1e-3  # ensure theta > 0
        
        if self.loss_type == 'zinb':
            pi = torch.sigmoid(self.fc_pi(h))  # pi between 0 and 1
        
        return recon_x, theta, pi

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon_x, theta, pi = self.decode(z)
        return recon_x, mu, logvar, theta, pi

    def loss_function(self, recon_x, x, mu, logvar, theta=None, pi=None):
        return vae_loss(recon_x, x, mu, logvar, loss_type=self.loss_type, nb_theta=theta, zinb_pi=pi)

# loss func
def vae_loss(recon_x, x, mu, logvar, loss_type='mse', kl_weight=1.0, nb_theta=None, zinb_pi=None, huber_delta=1.0):
    """
    Compute VAE loss with different reconstruction loss options.
    
    Args:
    recon_x (Tensor): Reconstructed input
    x (Tensor): Original input
    mu (Tensor): Mean of the latent distribution
    logvar (Tensor): Log variance of the latent distribution
    loss_type (str): Type of reconstruction loss ('mse', 'gaussian', 'nb', 'zinb', 'huber', 'log_cosh')
    kl_weight (float): Weight for the KL divergence term
    nb_theta (Tensor): Dispersion parameter for Negative Binomial (required for 'nb' and 'zinb')
    zinb_pi (Tensor): Mixture parameter for ZINB (required for 'zinb')
    huber_delta (float): Delta parameter for Huber loss
    
    Returns:
    Tensor: Total loss (reconstruction loss + KL divergence)
    """
    
    # kl div
    kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    
    # recon loss
    if loss_type == 'mse':
        recon_loss = F.mse_loss(recon_x, x, reduction='sum')
    
    elif loss_type == 'gaussian':
        recon_loss = 0.5 * torch.sum(torch.pow((x - recon_x), 2))
    
    elif loss_type == 'nb':
        if nb_theta is None:
            raise ValueError("nb_theta is required for Negative Binomial loss")
        
        # positive outputs only
        eps = 1e-8
        recon_x = F.softplus(recon_x) + eps

        nb_theta = F.softplus(nb_theta) + eps
        log_theta_mu_eps = torch.log(nb_theta + recon_x + eps)
        recon_loss = torch.sum(
            torch.lgamma(x + nb_theta + eps)
            - torch.lgamma(nb_theta + eps)
            - torch.lgamma(x + 1.0)
            - x * torch.log(recon_x / log_theta_mu_eps + eps)
            - nb_theta * torch.log(nb_theta / log_theta_mu_eps + eps)
        )
    
    elif loss_type == 'zinb':
        if nb_theta is None or zinb_pi is None:
            raise ValueError("nb_theta and zinb_pi are required for ZINB loss")
        
        # positive outputs only
        eps = 1e-8
        recon_x = F.softplus(recon_x) + eps

        nb_theta = F.softplus(nb_theta) + eps
        zinb_pi = torch.clamp(zinb_pi, min=eps, max=1-eps)
        
        softplus_pi = F.softplus(-zinb_pi)
        log_theta_eps = torch.log(nb_theta + eps)
        log_theta_mu_eps = torch.log(nb_theta + recon_x + eps)
        case_zero = softplus_pi + F.softplus(zinb_pi) - zinb_pi
        mul_case_zero = torch.mul((x < eps).float(), case_zero)
        
        case_non_zero = (
            torch.lgamma(x + nb_theta + eps)
            - torch.lgamma(nb_theta + eps)
            - torch.lgamma(x + 1.0)
            - x * (torch.log(recon_x + eps) - log_theta_mu_eps)
            - nb_theta * (log_theta_eps - log_theta_mu_eps)
        )
        mul_case_non_zero = torch.mul((x > eps).float(), case_non_zero)
        
        recon_loss = torch.sum(mul_case_zero + mul_case_non_zero)

    elif loss_type == 'huber':
        recon_loss = F.smooth_l1_loss(recon_x, x, reduction='sum', beta=huber_delta)
    
    elif loss_type == 'log_cosh':
        diff = recon_x - x
        recon_loss = torch.sum(torch.log(torch.cosh(diff + 1e-12)))
    
    else:
        raise ValueError(f"Unsupported loss type: {loss_type}")
    
    return recon_loss + kl_weight * kl_div

# training
def train_vae(model, dataloader, optimizer, epochs, with_validation=False, val_loader=None): # loss_func="negbin"
    per_epoch_avg_loss={}
    per_epoch_val_avg_loss={}
    # model.train() 
    for epoch in range(epochs):
        model.train() # model train moved to inside loop
        total_loss = 0
        for batch_idx, (data, raw_data) in enumerate(dataloader):
            data = data.to(next(model.parameters()).device)  
            optimizer.zero_grad()
            recon_batch, mu, logvar, theta, pi = model(data)
            loss = model.loss_function(recon_batch, data, mu, logvar, theta, pi)
            loss.backward()
            total_loss += loss.item()
            optimizer.step()
        
        avg_loss = total_loss / len(dataloader.dataset)

        print(f'Epoch {epoch + 1}, Average Loss: {avg_loss:.4f}')

        # get validation loss
        if with_validation:
            val_loss, val_recons=validate_vae(model, val_loader)
            per_epoch_val_avg_loss[epoch]=val_loss
            print(f'Epoch {epoch + 1}, Validation Loss: {val_loss:.4f}')

        # save loss
        per_epoch_avg_loss[epoch]=avg_loss

    if with_validation:
        return per_epoch_avg_loss, per_epoch_val_avg_loss
    else:
        return per_epoch_avg_loss

# validate
def validate_vae(model, val_loader): 
    model.eval()  
    total_loss = 0
    all_reconstructions = []
    with torch.no_grad():
        for data, raw_data in val_loader:
            data = data.to(next(model.parameters()).device)  
            recon_batch, mu, logvar, theta, pi = model(data)
            loss = model.loss_function(recon_batch, data, mu, logvar, theta, pi)
            total_loss += loss.item()
            all_reconstructions.append(recon_batch.cpu())
    avg_loss = total_loss / len(val_loader.dataset)
    all_reconstructions = torch.cat(all_reconstructions, dim=0)
    return avg_loss, all_reconstructions

# convert list of tensors to a dataframe (for latent_reps, rows are latent features and columns are samples)
def merge_tensors_to_df(tensor_list):
    array_list=[tensor.cpu().numpy() for tensor in tensor_list]
    combined_array=np.column_stack(array_list)
    df=pd.DataFrame(combined_array)
    return df

# extract latent layer space
def extract_latent_parameters(model, dataloader):
    model.eval() 
    latent_means = []
    latent_logvars = []
    latent_reps = []
    
    with torch.no_grad():
        for data,_ in dataloader:
            data = data.to(next(model.parameters()).device) 
            for d in data:
                mu, logvar = model.encode(d)
                z = model.reparameterize(mu, logvar)
                latent_means.append(mu)
                latent_logvars.append(logvar)
                latent_reps.append(z)
    latent_means_df=merge_tensors_to_df(latent_means)
    latent_logvars_df=merge_tensors_to_df(latent_logvars)
    latent_reps_df=merge_tensors_to_df(latent_reps)
    
    return latent_means_df, latent_logvars_df, latent_reps_df

# compute the reconstruction error for each sample after training
def compute_reconstruction_error(model, dataloader):
    model.eval()  
    reconstruction_errors = []
    with torch.no_grad():
        for data, in dataloader:
            data = data.to(next(model.parameters()).device)  
            for d in data:
                mu, logvar = model.encode(d)
                z = model.reparameterize(mu, logvar)
                recon_data = model.decode(z)

                # compute reconstruction error
                error = torch.abs(d - recon_data) 
                reconstruction_errors.append(error)

    return reconstruction_errors

# validation functions 
def ccc(x,y):
    sxy = np.sum((x - x.mean())*(y - y.mean()))/x.shape[0]
    rhoc = 2*sxy / (np.var(x) + np.var(y) + (x.mean() - y.mean())**2)
    return rhoc

def lins_ccc(y_true, y_pred):

    mean_true = torch.mean(y_true)
    mean_pred = torch.mean(y_pred)
    var_true = torch.var(y_true)
    var_pred = torch.var(y_pred)
    covar = torch.mean((y_true - mean_true) * (y_pred - mean_pred))
    
    pearson_corr = covar / (torch.sqrt(var_true) * torch.sqrt(var_pred))
    ccc = (2 * pearson_corr * torch.sqrt(var_true) * torch.sqrt(var_pred)) / (var_true + var_pred + (mean_true - mean_pred)**2)
    ccc=float(ccc)
    
    return ccc

def recon_corr(dataset, recons):
    spear_corr_list=[]
    pear_corr_list=[]
    ccc_list=[]
    for (data, raw_data), recon in zip(dataset, recons):
        spear_corr, spear_p=spearmanr(data, recon) # data and raw_data or recon
        pear_corr, pear_p=pearsonr(data, recon) # data and raw_data or recon
        # ccc_corr=ccc(data, recon)
        ccc_corr=lins_ccc(data, recon)
        spear_corr_list.append(spear_corr)
        pear_corr_list.append(pear_corr)
        ccc_list.append(ccc_corr)
    return spear_corr_list, pear_corr_list, ccc_list


# mae
def compute_mae(y_true, y_pred):
    return torch.mean(torch.abs(y_true - y_pred)).item()

def calculate_mae_list(input_tensor, reconstructed_tensor):
    mae_list = []
    for i in range(len(input_tensor)):
        y_true = input_tensor[i][0]  # relative to norm data
        y_pred = reconstructed_tensor[i]
        mae = compute_mae(y_true, y_pred)
        mae_list.append(mae)
    return mae_list

# def select_and_plot_samples(correlation_list, input_tensor, reconstructed_tensor, tb_N=5, nrows=2, ncols=5, save=False, output_path=None, filename='corr_plots.png'):
#     """
#     Selects the top and bottom N samples based on correlation and plots them.
    
#     Parameters:
#     correlation_list: list
#         List of correlation values (Pearson, Spearman, or CCC).
#     input_tensor: torch.utils.data.TensorDataset
#         Original input data tensor (paired data).
#     reconstructed_tensor: torch.Tensor
#         Reconstructed data tensor.
#     tb_N: int
#         Number of top and bottom correlated samples to select.
#     nrows: int
#         Number of rows in the grid plot.
#     ncols: int
#         Number of columns in the grid plot.
#     """
#     correlation_array = np.array(correlation_list)
    
#     # get the indices for the top N and bottom N samples
#     top_indices = np.argsort(correlation_array)[-tb_N:]
#     bottom_indices = np.argsort(correlation_array)[:tb_N]
#     selected_indices = np.concatenate((top_indices, bottom_indices))
    
#     # top samples
#     selected_input = torch.stack([input_tensor[i][0] for i in selected_indices]) # index 0 to choose normalized data, index 1 to choose raw data
#     selected_reconstructed = reconstructed_tensor[selected_indices]
    
#     # plotting
#     plt.figure(figsize=(15, 7)) # was 15 by 10 but let to rectangualar plots
#     for i, idx in enumerate(selected_indices):
#         plt.subplot(nrows, ncols, i + 1)
#         sns.scatterplot(x=selected_input[i].numpy(), y=selected_reconstructed[i].numpy(), s=10)
#         plt.xlabel('True')
#         plt.ylabel('Reconstructed')
#         plt.title(f'Sample {idx} - Correlation: {correlation_array[idx]:.2f}')
#     plt.tight_layout()
#     # plt.show()

#     if save:
#         plt.savefig(f"{output_path}/{filename}", dpi=300)

def select_and_plot_samples(correlation_list, input_tensor, reconstructed_tensor, tb_N=5, nrows=2, ncols=5, save=False, output_path=None, filename='corr_plots.png'):
    """
    Selects the top and bottom N samples based on correlation and plots them in a journal-style format.
    
    Parameters:
    correlation_list: list
        List of correlation values (Pearson, Spearman, or CCC).
    input_tensor: torch.utils.data.TensorDataset
        Original input data tensor (paired data).
    reconstructed_tensor: torch.Tensor
        Reconstructed data tensor.
    N: int
        Number of top and bottom correlated samples to select.
    nrows: int
        Number of rows in the grid plot.
    ncols: int
        Number of columns in the grid plot.
    """
    correlation_array = np.array(correlation_list)
    
    top_indices = np.argsort(correlation_array)[-tb_N:]
    bottom_indices = np.argsort(correlation_array)[:tb_N]
    selected_indices = np.concatenate((top_indices, bottom_indices))
    
    selected_input = torch.stack([input_tensor[i][0] for i in selected_indices])
    selected_reconstructed = reconstructed_tensor[selected_indices]
    
    fig, axes = plt.subplots(nrows, ncols, figsize=(15, 7), squeeze=False)
    
    for i, idx in enumerate(selected_indices):
        row = i // ncols
        col = i % ncols
        ax = axes[row, col]
        
        max_val = max(selected_input[i].max(), selected_reconstructed[i].max())
        ax.plot([0, max_val], [0, max_val], 'k--', alpha=0.75, zorder=2)
        
        ax.scatter(selected_input[i].numpy(), selected_reconstructed[i].numpy(), 
                   s=30, alpha=0.7, color='#7fbfff', edgecolor='#4c7399', linewidth=0.5, zorder=1)
        
        ax.set_xlabel('True', fontsize=10)
        ax.set_ylabel('Reconstructed', fontsize=10)
        ax.set_title(f'Sample {idx}\nCorr: {correlation_array[idx]:.2f}', fontsize=12)
        
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        ax.tick_params(axis='both', which='major', labelsize=8)
        
        ax.set_aspect('equal', 'box')
        
        ax.set_xlim(0, max_val)
        ax.set_ylim(0, max_val)
    
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.4, wspace=0.3)
    
    # plt.show()
    if save:
        plt.savefig(f"{output_path}/{filename}", dpi=300)

def plot_journal_histogram(data, title, bins=30, color='#4c72b0', figsize=(8, 6), save=False, output_path=None, filename='histogram.png'):
    """
    Plot a histogram with journal-style formatting.

    Parameters:
    data (list or numpy array): The data to plot.
    title (str): The title of the histogram.
    bins (int): Number of bins in the histogram (default: 30).
    color (str): Color of the histogram bars (default: '#4c72b0').
    figsize (tuple): Figure size (width, height) in inches (default: (8, 6)).
    """
    plt.figure(figsize=figsize)
    
    sns.set_style("ticks")
    
    sns.histplot(data, bins=bins, color=color, kde=True, edgecolor='black')
    
    plt.title(title, fontsize=24, fontweight='bold', pad=20)
    # plt.xlabel('Value', fontsize=20)
    # plt.ylabel('Frequency', fontsize=20)
    
    sns.despine()
    
    plt.tick_params(axis='both', which='major', labelsize=20)
    
    # plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    
    if save:
        plt.savefig(f"{output_path}/{filename}", dpi=300)

# plot loss and validation
def plot_and_save_loss_wt_val(loss_dict, val_loss_dict, plot_filename, dict_filename, val_filename, save=False, output_path=None):
    """
    Plots the average loss per epoch and saves the figure and the associated dictionary.

    Parameters:
    loss_dict (dict): Dictionary with epoch as the key and average loss as the value.
    plot_filename (str): Filename to save the plot (e.g., 'loss_plot.png').
    dict_filename (str): Filename to save the dictionary (e.g., 'loss_dict.json').

    Returns:
    None
    """
    # Plot the average loss per epoch
    epochs = list(loss_dict.keys())
    losses = list(loss_dict.values())

    val_epochs = list(val_loss_dict.keys())
    val_losses = list(val_loss_dict.values())

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, losses, linestyle='-', color='black', label='Train')
    plt.plot(val_epochs, val_losses, linestyle='-', color='orange', label='Val')
    plt.xlabel('Epoch')
    plt.ylabel('Average Loss')
    plt.title('Average loss per epoch')
    plt.grid(True)
    plt.legend()
    # plt.close()

    if save:
        plt.savefig(f"{output_path}/{plot_filename}")

        # Save the dictionary to a JSON file
        with open(f"{output_path}/{dict_filename}", 'w') as f:
            json.dump(loss_dict, f, indent=4)

        with open(f"{output_path}/{val_filename}", 'w') as f:
            json.dump(val_loss_dict, f, indent=4)

def plot_training_validation_loss(train_loss_dict, val_loss_dict, title="Training and Validation Loss", save=False, output_path=None):
    """
    Plot training and validation loss on the same graph with journal-style formatting.

    Parameters:
    train_loss_dict (dict): Dictionary with epochs as keys and training loss as values.
    val_loss_dict (dict): Dictionary with epochs as keys and validation loss as values.
    title (str): Title of the plot.
    """
    plt.figure(figsize=(10, 6))
    
    epochs = list(train_loss_dict.keys())
    train_losses = list(train_loss_dict.values())
    val_losses = list(val_loss_dict.values())
    
    plt.plot(epochs, train_losses, 'black', label='Training Loss', linewidth=2)
    plt.plot(epochs, val_losses, 'orange', label='Validation Loss', linewidth=2)
    
    plt.title(title, fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.legend(fontsize=10)
    
    plt.tick_params(axis='both', which='major', labelsize=10, direction='out', length=6, width=1)
    
    x_min, x_max = plt.xlim()
    y_min, y_max = plt.ylim()
    plt.xlim(x_min, x_max + (x_max - x_min) * 0.02)
    plt.ylim(y_min, y_max + (y_max - y_min) * 0.02)
    
    plt.gca().xaxis.set_major_locator(plt.MaxNLocator(nbins=6, prune=None))
    plt.gca().yaxis.set_major_locator(plt.MaxNLocator(nbins=6, prune=None))
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    if save:
        plt.savefig(f"{output_path}/training_validation_loss_plot.png")

        # Save the dictionary to a JSON file
        with open(f"{output_path}/training_loss.json", 'w') as f:
            json.dump(train_loss_dict, f, indent=4)

        with open(f"{output_path}/validation_loss.json", 'w') as f:
            json.dump(val_loss_dict, f, indent=4)

# 