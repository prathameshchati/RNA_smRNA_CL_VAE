    # load config file and get params
    print(f"Getting config file: {args.config_path}")
    config=utils.load_config(args.config_path)
    filepath, model_name, output_dir, filter_low_features, min_max_scaling, \
           isolate_top_genes, hidden_dims, latent_dim, epochs, batch_size, \
           learning_rate, save_model=utils.extract_config(config)
    print(filepath, model_name, output_dir, filter_low_features, min_max_scaling, \
           isolate_top_genes, hidden_dims, latent_dim, epochs, batch_size, \
           learning_rate, save_model)
    print("\n")

    # check if output directory exists, otherwise create one
    if not os.path.exists(output_dir):
        # directory does not exist, create it using mkdir command
        try:
            subprocess.run(['mkdir', '-p', output_dir], check=True)
            print(f"Directory created: {output_dir}\n")
        except subprocess.CalledProcessError as e:
            print(f"Failed to create directory {output_dir}: {e}\n")
    else:
        print(f"Directory already exists: {output_dir}\n")

    # preprocessing
    print("Preprocessing data...\n")
    adata=preprocess(filepath, filter_low_features, min_max_scaling, isolate_top_genes)

    # format data
    print("Formatting data...")
    X, dataset, dataloader=load_data(adata, batch_size)
    print("Finished formatting data...\n")

    # setup model
    print("Setting up model...")
    input_dim=adata.n_vars
    model=rnaseqAutoencoder(input_dim, hidden_dims, latent_dim)
    optimizer=torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_function=nn.MSELoss()
    print("Finished setting up model...\n")

    #  train
    print("COMMENCE TRAINING...")
    epoch_loss=train(model, dataloader, optimizer, loss_function, epochs=epochs, normalize=False, print_loss=False)
    print("Training finished, congrats...\n")

    # plot recon loss
    print("Plotting recon loss...")
    sns.lineplot(x=epoch_loss.keys(), y=epoch_loss.values())
    plt.title("per epoch loss")
    plt.xlabel("epoch")
    plt.ylabel("mse loss")
    plt.savefig(f"{output_dir}/recon_loss.png", dpi=300)
    print("Saved recon loss fig...\n")

    # extract latent space and save as df
    print("Extracting latent space for training samples...")



    # save; check if model exists under same name when saving,
    # prompt user to overwrite or not save


    





    # CHECK IF FILE EXISTS AND PRINT CONFIG FILE

    # SAVE MODEL, CONFIG AND ALL WEIGHTS/INFO IN OUTPUT DIRECTORY


