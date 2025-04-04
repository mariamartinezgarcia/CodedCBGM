import argparse
import yaml
import os

import torch

import os
import itertools
import pickle
import wandb
import yaml

import src.nn.modules as modules
from src.concept_coded_vae import CBCodedVAE
from datasets.color_mnist import get_confounded_color_mnist, get_color_mnist
from datasets.celeba import get_celeba_dataloader

#WANDB CONFIGURATIONS FOR CLUSTER
os.environ["WANDB__SERVICE_WAIT"] = "300"
os.environ['SSL_CERT_DIR'] = '/etc/ssl/certs'
os.environ['REQUESTS_CA_BUNDLE'] = '/etc/ssl/certs/ca-certificates.crt'
import ssl
ssl._create_default_https_context = ssl._create_unverified_context


def main():
    # We only specify the yaml file from argparse and handle rest
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-c", "--config", help="path to configuration file")

    args = parser.parse_args()

    with open(args.config, "r") as stream:
        cfg = yaml.safe_load(stream)
    print(f"Loaded configuration file {args.config}")

    # --- Read Configuration File --- #
    # Dataset
    dataset = cfg['dataset']['name']
    dataset_root = cfg['dataset']['root']
    selected_attr = cfg['dataset']['attr']
    # Model
    likelihood = cfg['model']['likelihood']
    type_encoder =  cfg['model']['type_encoder']
    type_decoder =  cfg['model']['type_decoder']
    beta = cfg['model']['beta']
    n_concepts = cfg['model']['n_concepts']
    concept_inf = cfg['model']['concept_inf']
    sc_type = cfg['model']['sc_type']
    sc_inf = cfg['model']['sc_inf']
    sc_dim = cfg['model']['sc_dim']
    # Codes
    concept_code_root = cfg['concept_code']['root']
    concept_code_file = cfg['concept_code']['file']
    concept_bits_info = cfg['concept_code']['bits_info']
    concept_bits_code = cfg['concept_code']['bits_code']
    sc_code_root = cfg['sc_code']['root']
    sc_code_file = cfg['sc_code']['file']
    sc_bits_info = cfg['sc_code']['bits_info']
    sc_bits_code = cfg['sc_code']['bits_code']
    # Checkpoint
    checkpoint_path = cfg['checkpoint']['path']
    # Train
    n_epochs = cfg['train']['n_epochs']
    batch_size = cfg['train']['batch_size']
    lr = cfg['train']['lr']
    n_samples = cfg['train']['n_samples']
    train_enc = cfg['train']['train_enc']
    train_dec = cfg['train']['train_dec']
    w_concept = cfg['train']['w_concept']
    w_orth =  cfg['train']['w_orth']
    save_model = cfg['train']['save']
    # Log
    wb = cfg['log']['wandb']
    wb_project = cfg['log']['wandb_project']
    wb_user = cfg['log']['wandb_user']
    # GPU
    use_cuda = cfg['gpu'] and torch.cuda.is_available()
    if use_cuda:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    # --- Debug --- #
    torch.autograd.set_detect_anomaly(True)

    # --- Dataset --- #
    if dataset == 'celeba':
        trainloader, attr_names, attr_indices = get_celeba_dataloader(root_dir=dataset_root, selected_attributes=selected_attr, batch_size=batch_size, image_size=64, split='train', num_workers=4)

    if dataset == 'color_mnist':
        trainloader = get_confounded_color_mnist(root=dataset_root,  batch_size=128, istesting=False)
        attr_names = None 
        attr_indices = None

    # --- Repetition Codes --- #
    # Concepts
    G_concept=None
    if concept_inf == 'rep':

        assert n_concepts == concept_bits_info, "The number of 'concept' information bits must be equal to the number of concepts."

        # Load matrices
        if concept_code_file == 'default':
            concept_code_path = os.path.join(concept_code_root, 'rep_matrices_'+str(n_concepts)+'_'+str(concept_bits_code)+'.pkl')
        else:
            concept_code_path = os.path.join(concept_code_root, concept_code_file)
        
        with open(concept_code_path, 'rb') as file:
            rep_matrices = pickle.load(file)

        G_concept = rep_matrices['G']

    # Side Channel
    G_sc=None
    if sc_type=='binary' and sc_inf == 'rep':
        # Load matrices
        if sc_code_file == 'default':
            sc_code_path = os.path.join(sc_code_root, 'rep_matrices_'+str(sc_bits_info)+'_'+str(sc_bits_code)+'.pkl')
        else:
            sc_code_path = os.path.join(sc_code_root, sc_code_file)
        
        with open(sc_code_path, 'rb') as file:
            rep_matrices = pickle.load(file)

        G_sc = rep_matrices['G']

    # ---- Obtain Codebook ---- #
    if concept_bits_info<15:
        # Generate all possible words with bits_info
        all_concept_words = torch.FloatTensor(list(map(list, itertools.product([0, 1], repeat=concept_bits_info))))
        n_concept_words = all_concept_words.shape[0]

    if sc_type=='binary':
        if sc_bits_info<15:
            # Generate all possible words with bits_info
            all_sc_words = torch.FloatTensor(list(map(list, itertools.product([0, 1], repeat=sc_bits_info))))
            n_sc_words = all_sc_words.shape[0]

    # --- Weights and Biases --- #
    if wb:   
        if sc_type is None:
            wandb.init(
                project = wb_project,
                entity = wb_user,
                name = 'concept '+concept_inf+', sc None ,'+dataset.lower(),
                group = dataset,
                job_type = 'eval',
                config=cfg
            )
        else:
            wandb.init(
                project = wb_project,
                entity = wb_user,
                name = 'concept '+concept_inf+', sc '+sc_type+', '+dataset.lower(),
                group = dataset,
                job_type = 'eval',
                config=cfg
            )

    # ---- Get encoder and decoder networks---- #
    if sc_type is None:
        enc = modules.get_encoder(type_encoder, concept_bits_code, dataset)
        dec = modules.get_decoder(type_decoder, concept_bits_code, dataset)
    if sc_type == 'continuous':  
        enc = modules.get_encoder(type_encoder, concept_bits_code + sc_dim*2, dataset)
        dec = modules.get_decoder(type_decoder, concept_bits_code + sc_dim, dataset)
    if sc_type == 'binary':  
        if sc_inf == 'uncoded':
            enc = modules.get_encoder(type_encoder, concept_bits_code + sc_dim, dataset)
            dec = modules.get_decoder(type_decoder, concept_bits_code + sc_dim, dataset)
        if sc_inf == 'rep':
            enc = modules.get_encoder(type_encoder, concept_bits_code + sc_bits_code, dataset)
            dec = modules.get_decoder(type_decoder, concept_bits_code + sc_bits_code, dataset)


    # ---- Declare the model ---- #
    model = CBCodedVAE(
        enc, 
        dec, 
        concept_bits_info,
        concept_inf=concept_inf,
        likelihood=likelihood,
        sc_type=sc_type,
        sc_inf=sc_inf,
        sc_dim=sc_dim,
        G_concept=G_concept,
        G_sc=G_sc,
        beta=beta,
        lr=lr,
        seed=0,
        ) 

    # ---- Train ---- #
    elbo_evol, concept_evol, orth_evol, kl_concept_evol, kl_sc_evol, rec_evol = model.train(
        trainloader, 
        n_epochs=n_epochs, 
        n_samples=n_samples,
        train_enc=train_enc, 
        train_dec=train_dec,
        w_concept=w_concept,
        w_orth=w_orth,
        wb=wb
        )

    if save_model:
        model.save(checkpoint_path)
        print('Model saved!')

    if wb:  
        wandb.finish()



if __name__ == "__main__":
    main()
