# RNA_smRNA_CL_VAE

__Quickstart__

Call from this directory w/ proper config file. 

```bash
$ python model/train.py configs/rna_config.yml
```

__Notes__

- Add data download pipeline for TCGA (note, smrna-seq indexing done elsewhere)

- Add tests and module info, add requirements file, setup... see algs formatting guidelines. 
- Add directory map and structure for module packaging
- Clean imports
- Need to fix zinb, nb for unimodal vae; particularly for smRNA implementation
- add github make file