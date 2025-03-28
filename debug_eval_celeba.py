import torch
from torchvision import datasets, transforms
import numpy as np
import itertools
import matplotlib.pyplot as plt

import random

from torch import nn

import wandb
import pickle
import yaml
import os

from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision.datasets import CelebA
import os

import src.nn.modules as modules
from src.concept_coded_vae import CBCodedVAE
from datasets.color_mnist import get_confounded_color_mnist
from datasets.celeba import get_celeba_dataloader

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


# --- Random Seed --- #
g = torch.Generator()
g.manual_seed(0)


# ---- Configuration ---- #
with open('./configs/config_sc_None_train.yml', "r") as file:
    cfg = yaml.safe_load(file)

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
# GPU
gpu = cfg['gpu']['device']
# Checkpoint
checkpoint_path = cfg['checkpoint']['path']
# Log
wb = cfg['log']['wandb']
wb_project = cfg['log']['wandb_project']
wb_user = cfg['log']['wandb_user']


# --- GPU --- #
os.environ["CUDA_VISIBLE_DEVICES"]=str(cfg['gpu']['device'])

# --- Debug --- #
torch.autograd.set_detect_anomaly(True)

# --- Dataset --- #
if dataset == 'celeba':
    testloader, attr_names, attr_indices = get_celeba_dataloader(root_dir=dataset_root, selected_attributes=selected_attr, image_size=64, split='test', num_workers=4)

if dataset == 'color_mnist':
    testloader = get_confounded_color_mnist(root=dataset_root,  batch_size=128, istesting=True)
    attr_names = None 
    attr_indices = None

# --- Repetition Codes --- #
# Concepts
if concept_inf == 'rep':
    # Load matrices
    if concept_code_file == 'default':
        concept_code_path = os.path.join(concept_code_root, 'rep_matrices_'+str(concept_bits_info)+'_'+str(concept_bits_code)+'.pkl')
    else:
        concept_code_path = os.path.join(concept_code_root, concept_code_file)
    
    with open(concept_code_path, 'rb') as file:
        rep_matrices = pickle.load(file)

    G_concept = rep_matrices['G']

# Side Channel
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
    enc = modules.get_encoder(type_encoder, concept_bits_code + sc_dim, dataset)
    dec = modules.get_decoder(type_decoder, concept_bits_code + sc_dim, dataset)

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
    seed=0,
    wb=wb
    ) 

# Load pre-trained model
checkpoint = torch.load(checkpoint_path, map_location=model.device)
model.load_state_dict(checkpoint)
print('Model loaded!')

# Evaluation mode
model.encoder.eval()
model.decoder.eval()

# ---- Reconstruction ---- #
with torch.no_grad():

    # 1. Obtain a batch of test data
    images, concepts = next(iter(testloader))
    # 2. Forward model
    latent_sample, concept_probs, reconstructed = model.forward(images)
    # 3. Plot reconstructed images
    images=images*0.5 + 0.5
    reconstructed=reconstructed*0.5 + 0.5
    fig, axes = plt.subplots(nrows=2, ncols=20, sharex=True, sharey=True, figsize=(40,4))
    for i in range(20):
        axes[0,i].imshow(np.transpose(images[i], (1,2,0)))
        axes[0,i].axis('off')
        axes[1,i].imshow(np.transpose(reconstructed[i].cpu().data.numpy(), (1,2,0)))
        axes[1,i].axis('off')
    
    plt.tight_layout(pad=0.00)
    #fig.suptitle('m_probs = '+str(concept_probs.detach().cpu().numpy()), fontsize=30)


    # W&B
    if cfg['setting']['wb']:
        wandb.log({"reconstructed": fig})
        wandb.log({"m_probs_reconstructed": concept_probs.detach().cpu().numpy()})

    plt.close(fig)


# ---- Reconstruction BCE ---- #
#BCE Loss
bce = nn.BCELoss(reduction='mean')
bce_sum = 0
for x, concepts in testloader:
    _, concept_probs, _ = model.forward(x)
    bce_sum += bce(concept_probs, concepts.type(torch.FloatTensor).to(concept_probs.device))

bce_final = bce_sum.item()/len(testloader)

# W&B
if wb:
    wandb.log({"BCE TEST": bce_final})


# ---- Generation ---- #
with torch.no_grad():

    # 1. Generate random images
    generated_imgs, generated_concepts = model.generate(n_samples=100)
    generated_imgs=generated_imgs*0.5 + 0.5
    # 2. Plot generated images
    fig, axes = plt.subplots(nrows=10, ncols=10, sharex=True, sharey=True, figsize=(20,20))
    for i in range(10):
        for j in range(10):
            ax = axes[i, j]
            ax.imshow(np.transpose(generated_imgs[i * 10 + j].cpu().data.numpy(), (1,2,0))) 
            ax.axis('off')
    plt.tight_layout(pad=0.00)

    # W&B
    if wb:
        wandb.log({"generated": fig})

    plt.close(fig)


# ---- Intervention ---- #







