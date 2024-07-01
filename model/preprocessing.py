import numpy as np
import tqdm as tqdm
import scanpy as sc
import seaborn as sns
import torch


# ft threshold, 0.8 for rnaseq and 0.5 for smrnaseq
def preprocess_adata(seq_adata, filter_genes_threshold=0.01, norm_per_sample=False, log=False, isolate_top=False, N=10000):
    # add counts layer
    seq_adata.layers['counts']=seq_adata.X.copy()

    # drop low expressed genes
    print(seq_adata.shape)
    sc.pp.filter_genes(seq_adata, min_cells=int(seq_adata.shape[0] * filter_genes_threshold)) # 0.8 worked best # adjusted from default 0.01 to 0.1 since we have less samples
    print(seq_adata.shape)

    # normalize per sample - this seems to work best for training
    if norm_per_sample:
        seq_adata, seq_adata_raw=normalize_per_sample(seq_adata, scaling_factor=10000, log=log)
        
    # add mean and var of expression per gene to determine outlier genes (axis=0)
    seq_adata.var['mean_expression']=np.mean(seq_adata.X.todense(), axis=0).tolist()[0]
    seq_adata.var['var_expression']=np.var(seq_adata.X.todense(), axis=0).tolist()[0]

    # get mean and var expression per SAMPLE (axis=1)
    seq_adata.obs['mean_expression']=sum(np.mean(seq_adata.X, axis=1).tolist(), [])
    seq_adata.obs['var_expression']=sum(np.var(seq_adata.X.todense(), axis=1).tolist(), [])

    # compute cutoff thresholds for filtering outliers
    mean_expression_threshold=np.percentile(seq_adata.var.mean_expression, 99)
    var_expression_threshold=np.percentile(seq_adata.var.var_expression, 99)
    print("mean and var thresholds:", mean_expression_threshold, var_expression_threshold)

    # determine what to filter
    seq_adata.var[seq_adata.var.mean_expression>=mean_expression_threshold]
    seq_adata.var[seq_adata.var.var_expression>=var_expression_threshold]
    genes_to_keep=seq_adata.var.mean_expression<=mean_expression_threshold

    # filter genes - counts layer is also filtered automatically
    seq_adata=seq_adata[:, genes_to_keep].copy()

    # sns.histplot(seq_adata.var.mean_expression)

    # set if we are min max scaling (0 to 1 range for rna-seq)
    isolate_top=isolate_top # for selecting gene sets # 10K WORKED BEST

    # isolate top N features
    # format rna-seq object
    if isolate_top:
        seq_adata.raw=seq_adata  # keep full dimension safe
        sc.pp.highly_variable_genes(
            seq_adata,
            flavor="seurat_v3",
            n_top_genes=N,
            layer="counts",
            batch_key="batch_id",
            subset=True,
        )

        # sample N random genes for later; benchmarking the model
        # rnaseq_ad_form_random = subsample_genes(rnaseq_ad_form, N)

    # dims
    scale_data=False
    if scale_data:
        print(torch.tensor(seq_adata.X).float().size())
    else:
        print(torch.tensor(seq_adata.X.todense()).float().size())
    print(torch.tensor(seq_adata.layers['counts'].todense()).float().size())

    return seq_adata

# mm scaling
def min_max_scale(adata):
    if isinstance(adata.X, np.ndarray):
        data_matrix = adata.X
    else:
        data_matrix = adata.X.toarray()

    # data_matrix=data_matrix.T
    
    data_min = data_matrix.min(axis=0)
    data_range = data_matrix.max(axis=0) - data_min
    
    data_range[data_range == 0] = 1
    
    adata.X = (data_matrix - data_min) / data_range

# random gene sampling
def subsample_genes(adata, N):
    """ Randomly subsamples N genes from an anndata object. """
    
    if N > adata.n_vars:
        raise ValueError("N is greater than the number of genes in the dataset!")
    
    np.random.seed(456)  
    random_indices = np.random.choice(adata.n_vars, N, replace=False)
    
    adata_subsampled = adata[:, random_indices].copy()
    
    return adata_subsampled

# sample normalization
def normalize_per_sample(adata, scaling_factor=10000, log=False):
    """
    Normalizes the counts in the AnnData object per sample (cell).
    
    Parameters:
    adata (AnnData): The AnnData object containing the count matrix.
    scaling_factor (float): The scaling factor to multiply the normalized counts. Default is 10000.
    
    Returns:
    AnnData: The AnnData object with normalized counts.
    """
    if adata.X is None:
        raise ValueError("The AnnData object does not contain a count matrix in the X attribute.")
    
    adata_norm = adata.copy()
    sc.pp.normalize_total(adata_norm, target_sum=scaling_factor)

    if log:
        sc.pp.log1p(adata_norm)
    
    return adata_norm, adata

# outlier filtering
def remove_outliers_per_feature(data, threshold=1.5):
    """
    Removes outliers from the data based on the IQR method for each feature.
    
    Parameters:
    data (torch.Tensor): The input data tensor (samples x features).
    threshold (float): The IQR threshold to determine outliers. Default is 1.5.
    
    Returns:
    torch.Tensor: The data tensor with outliers removed.
    """
    data_np = data.numpy()
    Q1 = np.percentile(data_np, 25, axis=0)
    Q3 = np.percentile(data_np, 75, axis=0)
    IQR = Q3 - Q1

    lower_bound = Q1 - (IQR * threshold)
    upper_bound = Q3 + (IQR * threshold)

    mask = (data_np >= lower_bound) & (data_np <= upper_bound)
    filtered_data = np.where(mask, data_np, np.nan)
    nan_threshold = 0.5  
    nan_counts = np.isnan(filtered_data).sum(axis=1)
    filtered_data = filtered_data[nan_counts < nan_threshold * data_np.shape[1]]
    col_means = np.nanmean(filtered_data, axis=0)
    inds = np.where(np.isnan(filtered_data))
    filtered_data[inds] = np.take(col_means, inds[1])

    return torch.tensor(filtered_data, dtype=data.dtype)

def remove_outlier_genes(data, threshold=1.5):
    """
    Removes outlier genes from the data based on the IQR method for each gene.
    
    Parameters:
    data (torch.Tensor): The input data tensor (samples x features).
    threshold (float): The IQR threshold to determine outliers. Default is 1.5.
    
    Returns:
    torch.Tensor: The data tensor with outlier genes removed.
    """
    data_np = data.numpy()
    Q1 = np.percentile(data_np, 25, axis=0)
    Q3 = np.percentile(data_np, 75, axis=0)
    IQR = Q3 - Q1

    lower_bound = Q1 - (IQR * threshold)
    upper_bound = Q3 + (IQR * threshold)

    outlier_mask = (data_np < lower_bound) | (data_np > upper_bound)

    outlier_proportion = outlier_mask.mean(axis=0)

    gene_mask = outlier_proportion < 0.5  
    
    filtered_data = data_np[:, gene_mask]

    return torch.tensor(filtered_data, dtype=data.dtype), gene_mask

