import torch
import torch.nn as nn
from torch import optim

from src.nn.encoder import Encoder
from src.nn.decoder import Decoder

from src.utils.sampling import sample_from_qz_given_x, sample_from_qc_given_x, modulate_words
from src.train.loss import compute_word_logprobs
from src.train.train import trainloop
from src.utils.functions import check_args, set_random_seed



class CBCodedVAE(nn.Module):

    """
    Class implementing the Coded-DVAE.
    """

    def __init__(self, enc, dec, n_concepts, concept_inf='uncoded', likelihood='gauss', sc_type=None, sc_inf='uncoded', sc_dim=None, G_concept=None, G_sc=None, beta=10, lr=1e-4, weight_decay=1e-4, seed=None):
                 
        super(CBCodedVAE, self).__init__()
        """
        Initialize an instance of the class.

        Parameters
        ----------
        enc : torch.nn.Module
            Module with the architecture of the encoder neural network without the output activation.
        dec : torch.nn.Module
            Module with the architecture of the decoder neural network.
        n_concepts : int
            Number of binary concepts.
        concept_inf : str, optional
            Inference type for the binary concepts. Default 'uncoded'.
            - 'uncoded' for the uncoded case.
            - 'rep' for the coded case with inference at bit level using repetition codes.
        likelihood: string, optional
            Distribution used to compute the reconstruction term. Default 'gauss'.
            - 'gauss': Gaussian likelihood.
            - 'ber': Bernoulli likelihood.
        sc_type: str, optional
            Type of side channel. Default None.
            - None: do not use any side channel.
            - 'binary: binary side channel.
            - 'continuous': continuous side channel.
        sc_inf: str, optional
            Inference type for the binary side channel. Default 'uncoded'.
            - 'uncoded' for the uncoded case.
            - 'rep' for the coded case with inference at bit level using repetition codes.
        sc_dim: int, optional
            Latent dimension of the side channel. In the binary case, this indicates the number of information bits in the side channel.
        G_concept : torch.tensor, optional
            Matrix used to encode information words.
        G_sc : torch.tensor, optional
            Matrix used to encode information words in the binary side channel.
        beta: float, optional
            Temperature term that controls the decay of the exponentials in the smoothing transformation. Default to 10.
        lr: float, optional
            Learning rate. Default to 1e-4.
        weight_decay: float
            Weight decay. Default to 1e-4.
        seed: int
            Seed for reproducibility.
        """

        # Hyperparameters
        self.beta = torch.tensor(beta)
     
        # Main Configuration
        self.likelihood = likelihood
        self.n_concepts = n_concepts
        self.concept_inf = concept_inf
        
        # Side Channel Configuration
        self.sc_type = sc_type
        self.sc_dim = sc_dim
        self.sc_inf = sc_inf

        # Concept Code
        self.G_concept = G_concept
        if G_concept != None:
            # G is not a list of tensors but rather one tensor
            self.H_concept = G_concept.T
            self.bits_code_concept = G_concept.shape[1]
        else:
            self.H_concept = None
            self.bits_code_concept = n_concepts

        # Side Channel Code
        self.G_sc = G_sc
        if G_sc != None:
            # G is not a list of tensors but rather one tensor
            self.H_sc= G_sc.T
            self.bits_code_sc = G_sc.shape[1]
        else:
            self.H_sc = None
            self.bits_code_sc = sc_dim
       
        # Check arguments
        # UPDATE!!!!!!
        #check_args(self.concept_inf, G = self.G_concept, H = self.H_concept) # UPDATE!!!!!!

        # Encoder
        self.encoder = Encoder(enc, self.bits_code_concept, sc_type=self.sc_type)
        # Decoder
        self.decoder = Decoder(dec)

        # Optimizers
        self.optimizer_encoder = optim.Adam(self.encoder.parameters(), lr=lr, weight_decay=weight_decay)
        self.optimizer_decoder = optim.Adam(self.decoder.parameters(), lr=lr, weight_decay=weight_decay)

        # Set device (GPU if available; otherwise CPU)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)

        # Set random seed
        if not (seed is None):
            set_random_seed(seed)
        

    def forward(self, x):

        """
        Forward pass.

        Parameters
        ----------
        x: torch.tensor
            Batch of data.
        """

        x = x.to(self.device)
        batch_size = x.shape[0]
        
        # Forward encoder
        out_concept, out_sc = self.encoder.forward(x)

        # Sanity check
        assert torch.any(torch.isinf(out_concept))==False, "Invalid probs value (inf)."
        assert torch.any(torch.isnan(out_concept))==False, "Invalid probs value (nan)."

        # Sample from the concept latent distribution

        # Uncoded case
        if self.concept_inf == 'uncoded':
            # Sample z 
            concept_probs = out_concept
            z_sample = sample_from_qz_given_x(concept_probs, beta=self.beta, n_samples=1)

        # Coded case
        if self.concept_inf == 'rep':
            # Compute the information marginals
            logpm1 = torch.matmul(torch.log(out_concept), self.H_concept.to(self.device))
            logpm0 = torch.matmul(torch.log(1-out_concept), self.H_concept.to(self.device))
            log_marginals = torch.stack((logpm0, logpm1), dim=2)
            log_marginals_norm = log_marginals - torch.logsumexp(log_marginals, dim=-1, keepdim=True)

            # Introduce code structure
            concept_probs = torch.exp(log_marginals_norm[:,:,1])
            qc = torch.matmul(concept_probs, self.G_concept.to(self.device))

            # Sample z 
            z_sample = sample_from_qz_given_x(qc, beta=self.beta, n_samples=1)

        # Sample from the side channel
        sc_sample = None
        if self.sc_type == 'continuous':
            # Sample z 
            mean = out_sc[:,:self.sc_dim]             # First half of the output corresponds to the mean vector
            var = torch.exp(out_sc[:,self.sc_dim:])   # Second half of the output corresponds to the logvar vector
            sc_sample = torch.randn(batch_size, self.sc_dim).to(self.device)*var + mean
       
        if self.sc_type == 'binary':

            if self.sc_inf == 'uncoded':
                #Sample z
                sc_sample = sample_from_qz_given_x(out_sc, beta=self.beta, n_samples=1)[:,:,0]

            if self.sc_inf == 'rep':
                # Compute the information marginals
                logpm1 = torch.matmul(torch.log(out_sc), self.H_sc.to(self.device))
                logpm0 = torch.matmul(torch.log(1-out_sc), self.H_sc.to(self.device))
                log_marginals = torch.stack((logpm0, logpm1), dim=2)
                log_marginals_norm = log_marginals - torch.logsumexp(log_marginals, dim=-1, keepdim=True)

                # Introduce code structure
                qc = torch.matmul(torch.exp(log_marginals_norm[:,:,1]), self.G_sc.to(self.device))
                # Sample z
                sc_sample = sample_from_qz_given_x(qc, beta=self.beta, n_samples=1)[:,:,0]

        # Forward decoder
        if self.sc_type is None:
            latent_sample = z_sample[:,:,0]
        else:
            latent_sample = torch.cat((z_sample[:,:,0], sc_sample), dim=1)
        reconstructed = self.decoder.forward(latent_sample)

        return latent_sample, concept_probs, reconstructed
    
    
    def train(self, train_dataloader, n_epochs=100, n_samples=1, train_enc=True, train_dec=True, w_concept=1., w_orth=1.,verbose=True, wb=False):

        """
        Train the model for a given number of epochs.
            
            Parameters
            ----------
            train_dataloader : torch Dataloader
                Dataloader with the training set.
            n_epochs: int, optional
                Number of epochs. Default 100.
            n_samples : int, optional
                Number of samples used for computing the ELBO. The number of samples is 1 by default.
            train_enc : boolean, optional
                Flag to indicate if the parameters of the encoder need to be updated. True by default.
            train_dec : boolean, optional
                Flag to indicate if the parameters of the decoder need to be updated. True by default.
            w_concept: float, optional
                Weight for the concept loss term. Default 1.
            w_orth: float, optional
                Weight for the orthogonality loss term. Default 1.   
            verbose: boolean, optional
                Flag to print the ELBO during training. True by default.
            wb: boolean, optional
                Flag to log the ELBO, KL term and reconstruction term to Weights&Biases.

            Returns
            -------
            elbo_evolution : list
                List containing the ELBO values obtained during training (1 value per epoch).
            kl_concepts_evolution : list
                List containing the Kullback-Leibler divergence values for the concept distribution obtained during training (1 value per epoch).
            kl_sc_evolution : list
                List containing the Kullback-Leibler divergence values for the side channel distribution obtained during training (1 value per epoch).
            rec_evolution : list
                List containing reconstruction term values obtained during training (1 value per epoch).
            concept_loss_evolution : list
                List containing BCE values obtained during training (1 value per epoch).
            orth_loss_evolution : list
                List containing orthogobality loss values obtained during training (1 value per epoch).
        """

        # Neural Networks in training mode
        self.encoder.train()
        self.decoder.train()

        # Train!
        print('Starting training...')
        elbo_evol_train, concept_evol_train, orth_evol_train, kl_concept_evol_train, kl_sc_evol_train, rec_evol_train = trainloop(
            self, 
            train_dataloader, 
            n_epochs, 
            n_samples=n_samples, 
            train_enc=train_enc,
            train_dec=train_dec,
            w_concept=w_concept,
            w_orth=w_orth,
            verbose=verbose,
            wb=wb 
        )

        print('Training finished!')

        return  elbo_evol_train, concept_evol_train, orth_evol_train, kl_concept_evol_train, kl_sc_evol_train, rec_evol_train
    

    def generate(self, n_samples=100, m_probs=None):

        """
        Generate new samples following the generative model.

        Parameters
        ----------
        n_samples: int, optional
            Number of samples to generate.
        m_probs: list, optional
            List with fixed concept pribability vectors. 

        Returns
        -------
        Generated samples and the sampled concepts used for generation.

        """

        if m_probs is None:
            m_probs = torch.ones((n_samples, self.n_concepts))*0.5
        else:
            m_probs = torch.flatten(m_probs).repeat(n_samples,1)

        # Uncoded case
        if self.concept_inf == 'uncoded':
            # Sample m
            m_sample = m_probs.bernoulli()
            # Sample z
            z_sample = modulate_words(m_sample.to(self.device), beta=self.beta)

        # Coded case
        if self.concept_inf == 'rep':
            # Sample m
            m_sample = m_probs.bernoulli()
            # Obtain a codeword
            c = torch.matmul(m_sample, self.G_concept)
            # Sample z
            z_sample = modulate_words(c.to(self.device), beta=self.beta)

        # Sample from the side channel
        sc_sample = None
        if self.sc_type == 'continuous':
            # Sample from the side channel
            sc_z_sample = torch.randn(n_samples, self.sc_dim).to(self.device)
       
        if self.sc_type == 'binary':
            sc_probs = torch.ones((n_samples, self.sc_dim))*0.5

            if self.sc_inf == 'uncoded':
                # Sample m
                sc_sample = sc_probs.bernoulli()
                # Sample z
                sc_z_sample = modulate_words(sc_sample.to(self.device), beta=self.beta)

            if self.sc_inf == 'rep':
                # Sample m
                sc_sample = sc_probs.bernoulli()
                # Obtain a codeword
                c = torch.matmul(sc_sample, self.G_sc)
                # Sample z
                sc_z_sample = modulate_words(c.to(self.device), beta=self.beta)

        # Forward decoder
        if self.sc_type is None:
            latent_sample = z_sample
        else:
            latent_sample = torch.cat((z_sample, sc_z_sample), dim=1)
        generated = self.decoder.forward(latent_sample)

        return generated, m_sample
    

    def save(self, path):

        """
        Save model.

        Parameters
        ----------
        path: str
           Path where the model will be saved.

        """

        torch.save(self.state_dict(), path)
        print('Model saved at ' + path)





