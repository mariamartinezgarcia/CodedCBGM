import torch
import torch.nn as nn
import numpy as np
from scipy.stats import entropy
from src.utils.sampling import sample_from_qc_given_x, sample_from_qz_given_x, modulate_words
from src.train.loss import compute_word_logprobs
from src.nn.modules import dclamp
from src.train.loss import log_gaussian


def compute_binary_entropy(bit_probs, words):

    """
    Compute entropy in base 2 of the distribution over words.

    Parameters
    ----------
    bit_probs : torch.tensor
        Bit probabilities.
    words : torch.tensor
        Codebook.

    Returns
    -------
    Entropy in base 2 of the distribution over words.
    """

    # Compute the entropy in base 2 of the distribution over words.  
    _, logq_norm = compute_word_logprobs(bit_probs, words)

    # Transform to probs to compute entropy in base 2
    probsq = torch.exp(logq_norm)
    entropy_qm = entropy(probsq.cpu().data.numpy(), base=2, axis=1)

    return torch.mean(torch.tensor(entropy_qm))


def get_latent_projections(model, dataset):

    """
    Obtain latent projections of data points.

    Parameters
    ----------
    model : CodedVAE instance
        Model.
    dataset : torch Dataset
        Dataset for which the projections will be computed.

    Returns
    -------
    projections : numpy.ndarray
        Latent projections of the observations in dataset.
    labels_projections:  numpy.ndarray
        Labels of the projected observations.
    """

    # Generate an auxiliar dataloader
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1000, shuffle=True)

    # Encode training data
    first=True
    for images, labels in dataloader:

        if model.inference == 'uncoded':
            # Encode and sample
            bit_probs = model.encoder.forward(images.to(model.device))
            z_sample = sample_from_qz_given_x(bit_probs, beta=model.beta)

        if model.inference == 'rep':
            # Encode and sample
            encoder_out = model.encoder.forward(images.to(model.device))
            logpm1 = torch.matmul(torch.log(encoder_out), model.H.to(model.device))
            logpm0 = torch.matmul(torch.log(1-encoder_out), model.H.to(model.device))

            log_marginals = torch.stack((logpm0, logpm1), dim=2)
            log_marginals_norm = log_marginals - torch.logsumexp(log_marginals, dim=-1, keepdim=True)

            bit_probs = torch.exp(log_marginals_norm[:,:,1])
            z_sample = sample_from_qz_given_x(bit_probs, beta=model.beta, n_samples=1)
        
        if model.inference == 'hier':
            c_dims = model.G.shape[1]
            # Encode and sample
            encoder_out = model.encoder.forward(images.to(model.device))

            logpm1_1 = torch.matmul(torch.log(encoder_out[:,:c_dims]), model.H.to(model.device))
            logpm1_0 = torch.matmul(torch.log(1-encoder_out[:,:c_dims]), model.H.to(model.device))

            log_marginals_1 = torch.stack((logpm1_0, logpm1_1), dim=2)

            log_marginals_norm_1 = log_marginals_1 - torch.logsumexp(log_marginals_1, dim=-1, keepdim=True)

            # Obtain q((m1+m2)|x)
            logpm12_1 = torch.matmul(torch.log(encoder_out[:,c_dims:]), model.H.to(encoder_out.device))
            logpm12_0 = torch.matmul(torch.log(1-encoder_out[:,c_dims:]), model.H.to(encoder_out.device))

            log_marginals_12 = torch.stack((logpm12_0, logpm12_1), dim=2)
            log_marginals_norm_12 = log_marginals_12 - torch.logsumexp(log_marginals_12, dim=-1, keepdim=True)

            # Obtain q(m2|x)
            combination1 = log_marginals_norm_12[:,:,1]+log_marginals_norm_1[:,:,0] # log q((m1+m2)=1|x)) + log q(m1=0|x)
            combination2 = log_marginals_norm_12[:,:,0]+log_marginals_norm_1[:,:,1] # log q((m1+m2)=0|x)) + log q(m1=1|x)
            combination = torch.stack((combination1, combination2), dim=2)

            logpm2_1 = torch.logsumexp(combination, dim=-1)
            # Clamp to avoid numerical instabilities
            logpm2_1 = torch.log(dclamp(torch.exp(logpm2_1), 0.0001, 0.9999))
            logpm2_0 = torch.log(dclamp(1-torch.exp(logpm2_1), 0.0001, 0.9999))

            log_marginals_2 = torch.stack((logpm2_0, logpm2_1), dim=2)
            log_marginals_norm_2 = log_marginals_2 - torch.logsumexp(log_marginals_2, dim=-1, keepdim=True)

            bit_probs_1 = torch.exp(log_marginals_norm_1)[:,:,1]
            bit_probs_2 = torch.exp(log_marginals_norm_2)[:,:,1]

            # Obtain n_samples from q(z|x) for each observed x
            qz1_sample = sample_from_qz_given_x(bit_probs_1, beta=model.beta, n_samples=1)  # shape [N, K, n_samples]
            qz12_sample = sample_from_qz_given_x(bit_probs_2, beta=model.beta, n_samples=1)  # shape [N, K, n_samples]
            z_sample = torch.cat((qz1_sample, qz12_sample), dim=1) # shape [N, K*2, n_samples]

        if model.inference == 'word':
            # Encode and sample
            bit_probs = model.encoder.forward(images.to(model.device))
            logq, _ = compute_word_logprobs(bit_probs, code_words=model.code_words.to(model.device))
            c_sample, _ = sample_from_qc_given_x(logq, model.code_words.to(model.device))
            z_sample = modulate_words(c_sample, beta=model.beta) 

        # Concatenate the projections
        if first:
            labels_projections = labels.data.numpy()
            projections = z_sample.cpu().data.numpy()
            first=False
        else:
            projections = np.concatenate((projections, z_sample.cpu().data.numpy()), axis=0)
            labels_projections = np.concatenate((labels_projections, labels.data.numpy()), axis=0)

    return projections, labels_projections


def get_confident_latent_projections(model, dataset, words, threshold=0.8):

    """
    Obtain latent projections of data points. We only consider projections with a probability larger than an established threshold.

    Parameters
    ----------
    model : CodedVAE instance
        Model.
    dataset : torch Dataset
        Dataset for which the projections will be computed.
    words : torch.tensor
        Codebook.
    threshold : float
        Threshold to consider a projection confident.

    Returns
    -------
    projections : numpy.ndarray
        Latent projections of the observations in dataset.
    labels_projections:  numpy.ndarray
        Labels of the projected observations.
    """

    # Generate an auxiliar dataloader
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1000, shuffle=True)

    # Encode training data
    first=True
    for images, labels in dataloader:

        if model.inference == 'uncoded':
            # Encode and sample
            bit_probs = model.encoder.forward(images.to(model.device))
            _, logq_norm = compute_word_logprobs(bit_probs, words.to(model.device))
            row_indices = (torch.exp(logq_norm).max(dim=1).values > threshold).nonzero(as_tuple=True)[0]
            z_sample = sample_from_qz_given_x(bit_probs[row_indices,:], beta=model.beta)
            labels = labels[row_indices.cpu()]
            
        if model.inference == 'rep':
            # Encode and sample
            encoder_out = model.encoder.forward(images.to(model.device))
            logpm1 = torch.matmul(torch.log(encoder_out), model.H.to(model.device))
            logpm0 = torch.matmul(torch.log(1-encoder_out), model.H.to(model.device))

            log_marginals = torch.stack((logpm0, logpm1), dim=2)
            log_marginals_norm = log_marginals - torch.logsumexp(log_marginals, dim=-1, keepdim=True)

            bit_probs = torch.exp(log_marginals_norm[:,:,1])

            _, logq_norm = compute_word_logprobs(bit_probs, words.to(model.device))
            row_indices = (torch.exp(logq_norm).max(dim=1).values > threshold).nonzero(as_tuple=True)[0]
            z_sample = sample_from_qz_given_x(bit_probs[row_indices,:], beta=model.beta)
            labels = labels[row_indices.cpu()]

        if model.inference == 'word':
            # Encode and sample
            bit_probs = model.encoder.forward(images.to(model.device))
            logq, logq_norm = compute_word_logprobs(bit_probs, code_words=model.code_words.to(model.device))
            row_indices = (torch.exp(logq_norm).max(dim=1).values > threshold).nonzero(as_tuple=True)[0]
            c_sample, _ = sample_from_qc_given_x(logq[row_indices,:], model.code_words.to(model.device))
            z_sample = modulate_words(c_sample, beta=model.beta) 
            labels = labels[row_indices.cpu()]

        # Concatenate the projections
        if first:
            labels_projections = labels.data.numpy()
            projections = z_sample.cpu().data.numpy()
            first=False
        else:
            projections = np.concatenate((projections, z_sample.cpu().data.numpy()), axis=0)
            labels_projections = np.concatenate((labels_projections, labels.data.numpy()), axis=0)

    return projections, labels_projections


def get_latent_words(model, dataset):

    """
    Obtain latent words of data points. 

    Parameters
    ----------
    model : CodedVAE instance
        Model.
    dataset : torch Dataset
        Dataset for which the projections will be computed.

    Returns
    -------
    projections : numpy.ndarray
        Latent words of the observations in dataset.
    labels_projections:  numpy.ndarray
        Labels of the projected observations.
    """

    # Generate an auxiliar dataloader
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1000, shuffle=True)

    # Encode training data
    first=True
    for images, labels in dataloader:

        if model.inference == 'word':
            # Encode and sample
            bit_probs = model.encoder.forward(images.to(model.device))
            logq, _ = compute_word_logprobs(bit_probs, code_words=model.code_words.to(model.device))
            words, _ = sample_from_qc_given_x(logq, model.code_words.to(model.device))

        if model.inference == 'rep':
            # Encode and sample
            encoder_out = model.encoder.forward(images.to(model.device))
            logpm1 = torch.matmul(torch.log(encoder_out), model.H.to(model.device))
            logpm0 = torch.matmul(torch.log(1-encoder_out), model.H.to(model.device))

            log_marginals = torch.stack((logpm0, logpm1), dim=2)
            log_marginals_norm = log_marginals - torch.logsumexp(log_marginals, dim=-1, keepdim=True)

            words = torch.bernoulli(torch.exp(log_marginals_norm[:,:,1]))

  
        # Concatenate the projections
        if first:
            labels_projections = labels.data.numpy()
            projections = words.cpu().data.numpy()
            first=False
        else:
            projections = np.concatenate((projections, words.cpu().data.numpy()), axis=0)
            labels_projections = np.concatenate((labels_projections, labels.data.numpy()), axis=0)
        
    return projections, labels_projections


def compute_marginal_likelihood(model, dataloader, n_bits_info, n_samples=50):

    """
    Estimate the marginal log-likelihood via Importance Sampling

    Parameters
    ----------
    model : CodedVAE instance
        Model.
    dataloader : torch Dataloader
        Dataloader of a given dataset.
    n_bits_info : int
        Number of information bits.
    n_samples: int
        Number of samples used to estimate the marginal log-likelihood.

    Returns
    -------
    loglik : estimated log-likelihood

    """

    loglik = []

    batch=0 # For printing
    a = 0.2
    b = 0.8
    beta = 15.

    for images, _ in dataloader:
        batch += 1
        print(f'Batch: {batch}/{len(dataloader)}')
        for x in images:
            
            posterior = model.encoder.forward(x.unsqueeze(0).to(model.device))

            if model.inference == 'rep':
                logqm1 = torch.matmul(torch.log(posterior), model.H.to(model.device))
                logqm0 = torch.matmul(torch.log(1-posterior), model.H.to(model.device))

                log_marginals = torch.stack((logqm0, logqm1), dim=2)
                log_marginals_norm = log_marginals - torch.logsumexp(log_marginals, dim=-1, keepdim=True)
                bit_probs = torch.exp(log_marginals_norm[:,:,1])
            if model.inference == 'uncoded':
                bit_probs = posterior

            # Generate a mask 
            mask = torch.zeros_like(bit_probs, dtype=torch.float32)
            # Set values to 0 where the tensor is between 0 and a
            mask[bit_probs < a] = 0
            # Set values to 1 where the tensor is between b and 1
            mask[bit_probs > b] = 1
            # Set values to 2 where the tensor is between a and b
            mask[(bit_probs >= a) & (bit_probs <= b)] = 2

            # Sample from the three distributions
            zeros = torch.zeros(n_samples, n_bits_info)
            ones = torch.ones(n_samples, n_bits_info)

            exp0 = modulate_words(zeros, beta=torch.tensor(beta))
            exp1 = modulate_words(ones, beta=torch.tensor(beta))
            uniform = torch.FloatTensor(n_samples,n_bits_info).uniform_(a, b)

            # Compute importance weights
            logp_exp1 = -torch.tensor(beta)*(exp1-1) - torch.log(1-torch.exp(-torch.tensor(beta))) + torch.log(torch.tensor(beta))
            logp_exp0 = -torch.tensor(beta)*exp0 - torch.log(1-torch.exp(-torch.tensor(beta))) + torch.log(torch.tensor(beta))
            logp_uni = torch.ones_like(uniform)*(-torch.log(torch.tensor(b)-torch.tensor(a)))

            # Index with mask
            theta = torch.zeros_like(zeros)
            theta[:, (mask.cpu()==0)[0,:]] = exp0[:, (mask.cpu()==0)[0,:]]
            theta[:, (mask.cpu()==1)[0,:]] = exp1[:, (mask.cpu()==1)[0,:]]
            theta[:, (mask.cpu()==2)[0,:]] = uniform[:, (mask.cpu()==2)[0,:]]

            logq_theta = torch.zeros_like(zeros)
            logq_theta[:, (mask.cpu()==0)[0,:]] = logp_exp0[:, (mask.cpu()==0)[0,:]]
            logq_theta[:, (mask.cpu()==1)[0,:]] = logp_exp1[:, (mask.cpu()==1)[0,:]]
            logq_theta[:, (mask.cpu()==1)[0,:]] = logp_uni[:, (mask.cpu()==1)[0,:]]
            logq_theta = torch.sum(logq_theta, dim=1)

            # Compute prior
            logp_theta_exp1 = -torch.tensor(beta)*(theta-1) - torch.log(1-torch.exp(-torch.tensor(beta))) + torch.log(torch.tensor(beta)) + torch.log(torch.tensor(0.3))
            logp_theta_exp0 = -torch.tensor(beta)*(theta) - torch.log(1-torch.exp(-torch.tensor(beta))) + torch.log(torch.tensor(beta)) + torch.log(torch.tensor(0.3))
            logp_theta_uni = -torch.ones_like(theta)*(torch.log(torch.tensor(1.)-torch.tensor(0.)) + torch.log(torch.tensor(0.4)))
            logp_theta = torch.stack([logp_theta_exp0, logp_theta_exp1, logp_theta_uni], dim=-1)
            logp_theta = torch.logsumexp(logp_theta, dim=-1)
            logp_theta = torch.sum(logp_theta, dim=1)

            # Compute normalized importance weights
            logw = (logp_theta - logq_theta) - torch.logsumexp((logp_theta-logq_theta), dim=0)

            # Sample z using the thetas sampled
            if model.inference == 'rep':
                theta = torch.matmul(theta, model.G)
            theta = dclamp(theta, 0, 1-1e-3)  
            z = sample_from_qz_given_x(theta, n_samples=1, beta=model.beta)[:,:,0]

            # Forward decoder
            out_decoder = model.decoder.forward(z.to(model.device)).view(n_samples, -1)
            x_flat = x.view(1, -1)
            if model.likelihood == 'ber':
                bce = torch.nn.BCELoss(reduction='none')
                logp_x_z = -torch.sum(bce(out_decoder, x_flat.repeat(n_samples,1).to(model.device)), dim=1) 
            if model.likelihood == 'gauss':
                covar = torch.ones(out_decoder.shape[1]).to(model.device) * 0.1
                logp_x_z = log_gaussian(x_flat.to(model.device), out_decoder, covar)

            logp_x = torch.logsumexp((logw.to(model.device) + logp_x_z), dim=0)

            loglik.append(logp_x)
    
    return torch.mean(torch.tensor(loglik))



# ---- Class error in reconstruction ---- #

class ClassifierNetwork(nn.Module):
    """
    Class implementing a CNN-based classifier.
    """

    def __init__(self):
        super(ClassifierNetwork, self).__init__()

        """
        Initialize an instance of the class.
        """

        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        
        self.fc1 = nn.Linear(in_features=64*6*6, out_features=1024)
        self.drop = nn.Dropout(0.25)
        self.fc2 = nn.Linear(in_features=1024, out_features=128)
        self.fc3 = nn.Linear(in_features=128, out_features=10)

        self.logsoftmax = nn.LogSoftmax(dim=1)  

    def forward(self, x):

        """
        Forward pass.

        Parameters
        ----------
        x: torch.tensor
            Batch of data.
        """

        out = self.layer1(x)
        out = self.layer2(out)
        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        out = self.drop(out)
        out = self.fc2(out)
        out = self.fc3(out)
        out = self.logsoftmax(out)
        
        return out


def eval_reconstruction(classifier_network, model, dataloader, code_words=None, threshold=None):

    """
    Evaluate the reconstruction accuracy.

    Parameters
    ----------
    classifier_network : ClassifierNetwork instance
        Classifier.
    model : CodedVAE instance
        Model we are evaluating.
    dataloader : torch Dataloader
        Dataloader of a given dataset.
    code_words: torch.tensor, optional
        Codebook.
    threshold: float
        Threshold to consider a projection confident.

    Returns
    -------
    Reconstruction accuracy.

    """

    # Evaluation mode
    classifier_network.eval()

    # Turn off gradients for validation, saves memory and computations
    with torch.no_grad():

        same_class = 0
        num_data_points = 0
        for images, labels in dataloader:
                
            if (not (threshold is None)) and (not (code_words is None)):

                bit_probs = model.encoder.forward(images.to(model.device))
                if model.inference == 'rep':
                    logpm1 = torch.matmul(torch.log(bit_probs), model.H.to(model.device))
                    logpm0 = torch.matmul(torch.log(1-bit_probs), model.H.to(model.device))
                    log_marginals = torch.stack((logpm0, logpm1), dim=2)
                    log_marginals_norm = log_marginals - torch.logsumexp(log_marginals, dim=-1, keepdim=True)
                    bit_probs = torch.exp(log_marginals_norm[:,:,1])
                if model.inference == 'hier':
                    c_dims = model.G.shape[1]
                    # Obtain q(m1|x)
                    logpm1_1 = torch.matmul(torch.log(bit_probs[:,:c_dims]), model.H.to(model.device))
                    logpm1_0 = torch.matmul(torch.log(1-bit_probs[:,:c_dims]), model.H.to(model.device))

                    log_marginals_1 = torch.stack((logpm1_0, logpm1_1), dim=2)
                    log_marginals_norm_1 = log_marginals_1 - torch.logsumexp(log_marginals_1, dim=-1, keepdim=True)

                    # Obtain q((m1+m2)|x)
                    logpm12_1 = torch.matmul(torch.log(bit_probs[:,c_dims:]), model.H.to(model.device))
                    logpm12_0 = torch.matmul(torch.log(1-bit_probs[:,c_dims:]), model.H.to(model.device))

                    log_marginals_12 = torch.stack((logpm12_0, logpm12_1), dim=2)
                    log_marginals_norm_12 = log_marginals_12 - torch.logsumexp(log_marginals_12, dim=-1, keepdim=True)

                    # Obtain q(m2|x)
                    combination1 = log_marginals_norm_12[:,:,1]+log_marginals_norm_1[:,:,0] # log q((m1+m2)=1|x)) + log q(m1=0|x)
                    combination2 = log_marginals_norm_12[:,:,0]+log_marginals_norm_1[:,:,1] # log q((m1+m2)=0|x)) + log q(m1=1|x)
                    combination = torch.stack((combination1, combination2), dim=2)

                    logpm2_1 = torch.logsumexp(combination, dim=-1)
                    # Clamp to avoid numerical instabilities
                    logpm2_1 = torch.log(torch.clamp(torch.exp(logpm2_1), 0.0001, 0.9999))
                    logpm2_0 = torch.log(torch.clamp(1-torch.exp(logpm2_1), 0.0001, 0.9999))

                    log_marginals_2 = torch.stack((logpm2_0, logpm2_1), dim=2)
                    log_marginals_norm_2 = log_marginals_2 - torch.logsumexp(log_marginals_2, dim=-1, keepdim=True)

                    bit_probs = torch.cat((torch.exp(log_marginals_norm_1[:,:,1]), torch.exp(log_marginals_norm_2[:,:,1])), dim=1)

                _, logq_norm = compute_word_logprobs(bit_probs, code_words.to(model.device))
                row_indices = (torch.exp(logq_norm).max(dim=1).values > threshold).nonzero(as_tuple=True)[0]
                if len(row_indices) == 0:
                    continue
                images = images[row_indices.cpu()]
                labels = labels[row_indices.cpu()]
        
            _, reconstructed = model.forward(images)
            
            probs = classifier_network(reconstructed)
            pred = np.argmax(probs.cpu().detach().numpy(), axis=1)

            same_class += (labels.data.numpy()==pred).sum()
            num_data_points += images.shape[0]

    return same_class/num_data_points



