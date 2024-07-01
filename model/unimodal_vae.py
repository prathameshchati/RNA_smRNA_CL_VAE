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
        encoder_layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            encoder_layers.append(nn.Linear(prev_dim, h_dim))
            encoder_layers.append(nn.ReLU())
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
    model.train() 
    for epoch in range(epochs):
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

def select_and_plot_samples(correlation_list, input_tensor, reconstructed_tensor, tb_N=5, nrows=2, ncols=5, save=False, output_path=None, filename='corr_plots.png'):
    """
    Selects the top and bottom N samples based on correlation and plots them.
    
    Parameters:
    correlation_list: list
        List of correlation values (Pearson, Spearman, or CCC).
    input_tensor: torch.utils.data.TensorDataset
        Original input data tensor (paired data).
    reconstructed_tensor: torch.Tensor
        Reconstructed data tensor.
    tb_N: int
        Number of top and bottom correlated samples to select.
    nrows: int
        Number of rows in the grid plot.
    ncols: int
        Number of columns in the grid plot.
    """
    correlation_array = np.array(correlation_list)
    
    # get the indices for the top N and bottom N samples
    top_indices = np.argsort(correlation_array)[-tb_N:]
    bottom_indices = np.argsort(correlation_array)[:tb_N]
    selected_indices = np.concatenate((top_indices, bottom_indices))
    
    # top samples
    selected_input = torch.stack([input_tensor[i][0] for i in selected_indices]) # index 0 to choose normalized data, index 1 to choose raw data
    selected_reconstructed = reconstructed_tensor[selected_indices]
    
    # plotting
    plt.figure(figsize=(15, 7)) # was 15 by 10 but let to rectangualar plots
    for i, idx in enumerate(selected_indices):
        plt.subplot(nrows, ncols, i + 1)
        sns.scatterplot(x=selected_input[i].numpy(), y=selected_reconstructed[i].numpy(), s=10)
        plt.xlabel('True')
        plt.ylabel('Reconstructed')
        plt.title(f'Sample {idx} - Correlation: {correlation_array[idx]:.2f}')
    plt.tight_layout()
    # plt.show()

    if save:
        plt.savefig(f"{output_path}/{filename}", dpi=300)
