from dataclasses import dataclass
from typing import List, Tuple
from typing_extensions import Literal

import torch
import torch.nn as nn
from torch.nn import Parameter
import numpy as np
import math

from copy import deepcopy

from fs_mol.modules.graph_feature_extractor import (
    GraphFeatureExtractor,
    GraphFeatureExtractorConfig,
)
from fs_mol.data.dkt import DKTBatch

from fs_mol.utils.gp_utils import ExactGPLayer

import gpytorch
from gpytorch.distributions import MultivariateNormal

#from fs_mol.utils._stateless import functional_call

FINGERPRINT_DIM = 2048
PHYS_CHEM_DESCRIPTORS_DIM = 42


@dataclass(frozen=True)
class ADKTPriorModelConfig:
    # Model configuration:
    graph_feature_extractor_config: GraphFeatureExtractorConfig = GraphFeatureExtractorConfig()
    used_features: Literal[
        "gnn", "ecfp", "pc-descs", "gnn+ecfp", "ecfp+fc", "pc-descs+fc", "gnn+ecfp+pc-descs+fc"
    ] = "gnn+ecfp+fc"
    #distance_metric: Literal["mahalanobis", "euclidean"] = "mahalanobis"

class ADKTPriorModel(nn.Module):
    def __init__(self, config: ADKTPriorModelConfig):
        super().__init__()
        self.config = config

        # Create GNN if needed:
        if self.config.used_features.startswith("gnn"):
            self.graph_feature_extractor = GraphFeatureExtractor(
                config.graph_feature_extractor_config
            )

        self.use_fc = self.config.used_features.endswith("+fc")

        # Create MLP if needed:
        if self.use_fc:
            self.fc_out_dim = 2048
            # Determine dimension:
            fc_in_dim = 0
            if "gnn" in self.config.used_features:
                fc_in_dim += self.config.graph_feature_extractor_config.readout_config.output_dim
            if "ecfp" in self.config.used_features:
                fc_in_dim += FINGERPRINT_DIM
            if "pc-descs" in self.config.used_features:
                fc_in_dim += PHYS_CHEM_DESCRIPTORS_DIM

            self.fc = nn.Sequential(
                nn.Linear(fc_in_dim, 2048),
                nn.ReLU(),
                nn.Linear(2048, self.fc_out_dim),
            )

            # for name, param in self.named_parameters():
            #     # if not name.endswith("_nn") and not name.startswith("gp_"):
            #     if "weight" in name and not "mp_norm_layer" in name:
            #         setattr(self, name.replace(".", "_")+"_mu_nn", Parameter(nn.init.normal_(torch.Tensor(*tuple(param.shape)), mean=0.0, std=1.0)))
            #         setattr(self, name.replace(".", "_")+"_logsigma_nn", Parameter(torch.full(tuple(param.shape), math.log(0.1))))
            #     elif "bias" in name and not "mp_norm_layer" in name:
            #         setattr(self, name.replace(".", "_")+"_mu_nn", Parameter(torch.zeros(param.shape)))
            #         setattr(self, name.replace(".", "_")+"_logsigma_nn", Parameter(torch.full(tuple(param.shape), math.log(0.1))))

            for name, param in self.named_parameters():
                if name.endswith("weight"):
                    if "norm_layer" in name:
                        setattr(self, name.replace(".", "_")+"_mu_nn", Parameter(torch.ones(param.shape)))
                        setattr(self, name.replace(".", "_")+"_logsigma_nn", Parameter(torch.full(tuple(param.shape), math.log(0.1))))
                    else:
                        setattr(self, name.replace(".", "_")+"_mu_nn", Parameter(nn.init.xavier_uniform_(torch.Tensor(*tuple(param.shape)))))
                        setattr(self, name.replace(".", "_")+"_logsigma_nn", Parameter(torch.full(tuple(param.shape), math.log(0.1))))
                elif name.endswith("bias"):
                    setattr(self, name.replace(".", "_")+"_mu_nn", Parameter(torch.zeros(param.shape)))
                    setattr(self, name.replace(".", "_")+"_logsigma_nn", Parameter(torch.full(tuple(param.shape), math.log(0.1))))
                elif name.endswith("alpha"):
                    setattr(self, name.replace(".", "_")+"_mu_nn", nn.Parameter(torch.full(size=(1,), fill_value=1e-7)))
                    setattr(self, name.replace(".", "_")+"_logsigma_nn", Parameter(torch.full(tuple(param.shape), math.log(0.1))))
                else:
                    raise ValueError("Unexpected parameter with name {}.".format(name))

        self.__create_tail_GP(kernel_type=self.config.gp_kernel)
        
        if self.config.gp_kernel == "cossim":
            self.normalizing_features = True
        else:
            self.normalizing_features = False

    def feature_extractor_params(self):
        fe_params = []
        for name, param in self.named_parameters():
            if not name.endswith("_nn") and not name.startswith("gp_"):
                fe_params.append(param)
        return fe_params

    # def gnn_params(self):
    #     fe_params = []
    #     for name, param in self.named_parameters():
    #         if not name.endswith("_nn") and not name.startswith("gp_") and not name.startswith("fc"):
    #             fe_params.append(param)
    #     return fe_params

    def gp_params(self):
        gp_params = []
        for name, param in self.named_parameters():
            if name.startswith("gp_"):
                gp_params.append(param)
        return gp_params

    def prior_params(self):
        prior_params = []
        for name, param in self.named_parameters():
            if name.endswith("_nn"):
                prior_params.append(param)
        return prior_params

    # def gnn_prior_params(self):
    #     fe_params = []
        
    #     fe_params = self.gnn_params() + self.prior_params()
    #     return fe_params

    def reinit_gp_params(self, gp_input, use_lengthscale_prior=False):

        self.gp_model.load_state_dict(self.init_gp_model_params)
        self.gp_likelihood.load_state_dict(self.init_gp_likelihood_params)

        if self.config.gp_kernel == 'matern' or self.config.gp_kernel == 'rbf' or self.config.gp_kernel == 'RBF':
            median_lengthscale_init = self.compute_median_lengthscale_init(gp_input)
            if use_lengthscale_prior:
                scale = 0.25
                loc = torch.log(median_lengthscale_init).item() + scale**2 # make sure that mode=median_lengthscale_init
                self.gp_model.covar_module.base_kernel.lengthscale_prior.base_dist.loc = torch.tensor(loc).to(self.device)
                self.gp_model.covar_module.base_kernel.lengthscale_prior.base_dist.scale = torch.tensor(scale).to(self.device)
                # lengthscale_prior = gpytorch.priors.LogNormalPrior(loc=loc, scale=scale)
                # self.gp_model.covar_module.base_kernel.register_prior(
                #     "lengthscale_prior", lengthscale_prior, lambda m: m.lengthscale, lambda m, v: m._set_lengthscale(v)
                # )
            self.gp_model.covar_module.base_kernel.lengthscale = torch.ones_like(self.gp_model.covar_module.base_kernel.lengthscale) * median_lengthscale_init

    def __create_tail_GP(self, kernel_type):
        dummy_train_x = torch.ones(64, self.fc_out_dim)
        dummy_train_y = torch.ones(64)

        if self.config.use_ard:
            ard_num_dims = self.fc_out_dim
        else:
            ard_num_dims = None

        if self.config.use_numeric_labels:
            scale = 0.25
            loc = np.log(0.01) + scale**2 # make sure that mode=0.01
            noise_prior = gpytorch.priors.LogNormalPrior(loc=loc, scale=scale)
        else:
            scale = 0.25
            loc = np.log(0.1) + scale**2 # make sure that mode=0.1
            noise_prior = gpytorch.priors.LogNormalPrior(loc=loc, scale=scale)
        
        self.gp_likelihood = gpytorch.likelihoods.GaussianLikelihood(noise_prior=noise_prior).to(self.device)
        #self.gp_likelihood = gpytorch.likelihoods.GaussianLikelihood().to(self.device)
        self.gp_model = ExactGPLayer(
            train_x=dummy_train_x, train_y=dummy_train_y, likelihood=self.gp_likelihood, 
            kernel=kernel_type, ard_num_dims=ard_num_dims, use_numeric_labels=self.config.use_numeric_labels,
            use_lengthscale_prior=self.config.use_lengthscale_prior
        ).to(self.device)
        self.mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.gp_likelihood, self.gp_model).to(self.device)

        self.init_gp_model_params = deepcopy(self.gp_model.state_dict())
        self.init_gp_likelihood_params = deepcopy(self.gp_likelihood.state_dict())

    def compute_median_lengthscale_init(self, gp_input):
        dist_squared = torch.cdist(gp_input, gp_input) ** 2
        dist_squared = torch.triu(dist_squared, diagonal=1)
        return torch.sqrt(0.5 * torch.median(dist_squared[dist_squared>0.0]))

    def reinit_feature_extractor_params(self):
        # for name, param in self.named_parameters():
        #     if not name.endswith("_nn") and not name.startswith("gp_") and not name.endswith("alpha") and not "mp_norm_layer" in name:
        #         param.data = getattr(self, name.replace(".", "_")+"_mu_nn") + torch.exp(getattr(self, name.replace(".", "_")+"_logsigma_nn")) * torch.randn_like(getattr(self, name.replace(".", "_")+"_logsigma_nn")).to(self.device)
        
        for name, param in self.named_parameters():
            if not name.endswith("_nn") and not name.startswith("gp_"):
                param.data = getattr(self, name.replace(".", "_")+"_mu_nn") + torch.exp(getattr(self, name.replace(".", "_")+"_logsigma_nn")) * torch.randn_like(getattr(self, name.replace(".", "_")+"_logsigma_nn")).to(self.device)

    def log_prob(self, loc, logscale, value):
        # compute the variance
        var = (torch.exp(logscale) ** 2)
        return -((value - loc) ** 2) / (2 * var) - logscale - math.log(math.sqrt(2 * math.pi))

    def log_prior(self):
        logprob_prior = torch.tensor(0.0).to(self.device)
        # for name, param in self.named_parameters():
        #     if not name.endswith("_nn") and not name.startswith("gp_") and not name.endswith("alpha") and not "mp_norm_layer" in name:
        #         logprob_prior -= self.log_prob(getattr(self, name.replace(".", "_")+"_mu_nn"), getattr(self, name.replace(".", "_")+"_logsigma_nn"), param).sum()

        for name, param in self.named_parameters():
            if not name.endswith("_nn") and not name.startswith("gp_"):
                logprob_prior -= self.log_prob(getattr(self, name.replace(".", "_")+"_mu_nn"), getattr(self, name.replace(".", "_")+"_logsigma_nn"), param).sum()
        #         print(logprob_prior)
        #breakpoint()
        return logprob_prior

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def forward(self, input_batch: DKTBatch, train_loss: bool, predictive_val_loss: bool=False, is_functional_call: bool=False, reinit_gp_params: bool=False):
        support_features: List[torch.Tensor] = []
        query_features: List[torch.Tensor] = []

        if "gnn" in self.config.used_features:
            support_features.append(self.graph_feature_extractor(input_batch.support_features))
            query_features.append(self.graph_feature_extractor(input_batch.query_features))
        if "ecfp" in self.config.used_features:
            support_features.append(input_batch.support_features.fingerprints.float())
            query_features.append(input_batch.query_features.fingerprints.float())
        if "pc-descs" in self.config.used_features:
            support_features.append(input_batch.support_features.descriptors)
            query_features.append(input_batch.query_features.descriptors)

        support_features_flat = torch.cat(support_features, dim=1)
        query_features_flat = torch.cat(query_features, dim=1)

        if self.use_fc:
            support_features_flat = self.fc(support_features_flat)
            query_features_flat = self.fc(query_features_flat)

        if self.normalizing_features:
            support_features_flat = torch.nn.functional.normalize(support_features_flat, p=2, dim=1)
            query_features_flat = torch.nn.functional.normalize(query_features_flat, p=2, dim=1)

        if self.config.use_numeric_labels:
            support_labels_converted = input_batch.support_numeric_labels.float()
            query_labels_converted = input_batch.query_numeric_labels.float()
        else:
            support_labels_converted = self.__convert_bool_labels(input_batch.support_labels)
            query_labels_converted = self.__convert_bool_labels(input_batch.query_labels)

        # compute train/val loss if the model is in the training mode
        if self.training:
            assert train_loss is not None
            if train_loss: # compute train loss (on the support set)
                if is_functional_call: # return loss directly
                    if reinit_gp_params:
                        self.reinit_gp_params(support_features_flat.detach(), self.config.use_lengthscale_prior)
                    self.gp_model.set_train_data(inputs=support_features_flat, targets=support_labels_converted, strict=False)
                    logits = self.gp_model(support_features_flat)
                    logits = -self.mll(logits, self.gp_model.train_targets) + self.log_prior()
                else:
                    #self.reinit_gp_params(support_features_flat.detach(), self.config.use_lengthscale_prior)
                    self.gp_model.set_train_data(inputs=support_features_flat.detach(), targets=support_labels_converted.detach(), strict=False)
                    logits = None
            else: # compute val loss (on the query set)
                assert is_functional_call == True
                if predictive_val_loss:
                    self.gp_model.eval()
                    self.gp_likelihood.eval()
                    with gpytorch.settings.detach_test_caches(False):
                        self.gp_model.set_train_data(inputs=support_features_flat, targets=support_labels_converted, strict=False)
                        # return sum of the log predictive losses for all data points, which converges better than averaged loss
                        logits = -self.gp_likelihood(self.gp_model(query_features_flat)).log_prob(query_labels_converted) #/ self.predictive_targets.shape[0]
                    self.gp_model.train()
                    self.gp_likelihood.train()
                else:
                    self.gp_model.set_train_data(inputs=query_features_flat, targets=query_labels_converted, strict=False)
                    logits = self.gp_model(query_features_flat)
                    logits = -self.mll(logits, self.gp_model.train_targets)

        # do GP posterior inference if the model is in the evaluation mode
        else:
            assert train_loss is None
            self.gp_model.set_train_data(inputs=support_features_flat, targets=support_labels_converted, strict=False)

            with torch.no_grad():
                logits = self.gp_likelihood(self.gp_model(query_features_flat))

        return logits

    def __convert_bool_labels(self, labels):
        # True -> 1.0; False -> -1.0
        return (labels.float() - 0.5) * 2.0
