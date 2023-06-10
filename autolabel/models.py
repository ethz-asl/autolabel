import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_ngp.gridencoder import GridEncoder
from torch_ngp.encoding import get_encoder
from torch_ngp.activation import trunc_exp
from torch_ngp.ffmlp import FFMLP
import tinycudann as tcnn

from torch_ngp.nerf.renderer import NeRFRenderer


class FreqEncoder(nn.Module):

    def __init__(self, input_dim):
        super().__init__()
        self.encoder = tcnn.Encoding(input_dim, {
            "otype": "Frequency",
            "n_frequencies": 10
        })
        self.n_output_dims = self.encoder.n_output_dims

    def forward(self, x, bound):
        normalized = (x + bound) / (2.0 * bound)
        return self.encoder(normalized)


class HGFreqEncoder(nn.Module):

    def __init__(self, input_dim):
        super().__init__()
        self.encoder = tcnn.Encoding(input_dim, {
            "otype": "Frequency",
            "n_frequencies": 2
        })
        self.grid_encoding = tcnn.Encoding(
            input_dim, {
                "otype": "Grid",
                "type": "Hash",
                "n_levels": 16,
                "n_features_per_level": 2,
                "log2_hashmap_size": 19,
                "base_resolution": 16,
                "per_level_scale": 2.0,
                "interpolation": "Linear"
            })
        self.n_output_dims = self.encoder.n_output_dims + self.grid_encoding.n_output_dims

    def forward(self, x, bound):
        freq = self.encoder(x)
        normalized = (x + bound) / (2.0 * bound)
        # Sometimes samples might leak a bit outside the bounds.
        # This produces NaNs in the grid encoding, so we simply clip those points
        # assuming there aren't many of these.
        normalized = torch.clip(normalized, 0.0, 1.0)
        grid = self.grid_encoding(normalized)
        return torch.cat([freq, grid], dim=-1)


class ALNetwork(NeRFRenderer):

    def __init__(self,
                 encoding='hg',
                 num_layers=2,
                 hidden_dim=64,
                 geo_feat_dim=15,
                 num_layers_color=3,
                 hidden_dim_color=64,
                 hidden_dim_semantic=64,
                 semantic_classes=2,
                 bound=1,
                 **kwargs):
        super().__init__(bound, **kwargs)

        # sigma network
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.geo_feat_dim = geo_feat_dim

        self.encoder, self.in_dim = self._get_encoder(encoding)

        self.sigma_net = tcnn.Network(n_input_dims=self.in_dim,
                                      n_output_dims=1 + self.geo_feat_dim,
                                      network_config={
                                          "otype": "FullyFusedMLP",
                                          "activation": "ReLU",
                                          "output_activation": "None",
                                          "n_neurons": self.hidden_dim,
                                          "n_hidden_layers": self.num_layers
                                      })

        # color network
        self.num_layers_color = num_layers_color
        self.hidden_dim_color = hidden_dim_color
        self.encoder_dir = tcnn.Encoding(n_input_dims=3,
                                         encoding_config={
                                             "otype": "SphericalHarmonics",
                                             "degree": 4
                                         })
        self.color_features = self.encoder_dir.n_output_dims + self.geo_feat_dim

        self.color_net = tcnn.Network(
            n_input_dims=self.color_features,
            n_output_dims=3,
            network_config={
                "otype": "FullyFusedMLP",
                "activation": "ReLU",
                "output_activation": "None",
                "n_neurons": self.hidden_dim_color,
                "n_hidden_layers": self.num_layers_color
            })

        self.hidden_dim_semantic = hidden_dim_semantic
        self.semantic_classes = semantic_classes
        self.semantic_features = tcnn.Network(
            n_input_dims=self.geo_feat_dim,
            n_output_dims=self.hidden_dim_semantic,
            network_config={
                "otype": "CutlassMLP",
                "activation": "ReLU",
                "output_activation": "None",
                "n_neurons": self.hidden_dim_semantic,
                "n_hidden_layers": 2
            })
        self.semantic_out = tcnn.Network(n_input_dims=self.hidden_dim_semantic +
                                         self.geo_feat_dim,
                                         n_output_dims=semantic_classes,
                                         network_config={
                                             "otype": "FullyFusedMLP",
                                             "activation": "ReLU",
                                             "output_activation": "None",
                                             "n_neurons": 64,
                                             "n_hidden_layers": 1
                                         })

    def _get_encoder(self, encoding):
        if encoding == 'freq':
            encoder = FreqEncoder(3)
            return encoder, encoder.n_output_dims
        elif encoding == 'hg':
            return get_encoder('hashgrid', desired_resolution=2**18)
        elif encoding == 'hg+freq':
            encoder = HGFreqEncoder(3)
            return encoder, encoder.n_output_dims
        else:
            raise NotImplementedError(f"Unknown input encoding {encoding}")

    def forward(self, x, d):
        """
        x: [N, 3], in [-bound, bound] points
        d: [N, 3], normalized to [-1, 1] viewing directions
        """
        x = self.encoder(x, bound=self.bound)
        h = self.sigma_net(x)

        sigma = trunc_exp(h[..., 0])
        geo_feat = F.relu(h[..., 1:])

        d = self.encoder_dir(d)

        h = torch.cat([d, geo_feat], dim=-1)
        h = self.color_net(h)

        rgb = torch.sigmoid(h)

        features = self.semantic_features(geo_feat)
        semantic = self.semantic_out(
            torch.cat([F.relu(features), geo_feat], dim=-1))
        semantic = F.softmax(semantic, dim=-1)

        return sigma, rgb, semantic

    def density(self, x):
        """
        x: [N, 3] points in [-bound, bound]
        """
        x = self.encoder(x, bound=self.bound)
        h = self.sigma_net(x)

        sigma = trunc_exp(h[..., 0])
        geo_feat = h[..., 1:]

        return {
            'sigma': sigma,
            'geo_feat': geo_feat,
        }

    def color(self, x, d, mask=None, geo_feat=None, **kwargs):
        """
        x: [N, 3] in [-bound, bound]
        mask: [N,], bool, indicates where we actually needs to compute rgb.
        """
        if mask is not None:
            rgbs = torch.zeros(mask.shape[0], 3, dtype=x.dtype,
                               device=x.device)  # [N, 3]
            # in case of empty mask
            if not mask.any():
                return rgbs
            x = x[mask]
            d = d[mask]
            geo_feat = geo_feat[mask]

        # TinyCudaNN SH encoding requires inputs to be in [0, 1].
        d = (d + 1) / 2
        d = self.encoder_dir(d)

        h = torch.cat([d, geo_feat], dim=-1)

        h = self.color_net(h)

        h = torch.sigmoid(h)

        if mask is not None:
            rgbs[mask] = h.to(rgbs.dtype)
        else:
            rgbs = h

        return rgbs

    def get_params(self, lr):
        params = [{
            'params': self.encoder.parameters(),
            'lr': lr
        }, {
            'params': self.sigma_net.parameters(),
            'lr': lr
        }, {
            'params': self.encoder_dir.parameters(),
            'lr': lr
        }, {
            'params': self.color_net.parameters(),
            'lr': lr
        }, {
            'params': self.semantic_features.parameters(),
            'lr': lr
        }, {
            'params': self.semantic_out.parameters(),
            'lr': lr
        }]
        if self.bg_radius > 0:
            params.append({'params': self.encoder_bg.parameters(), 'lr': lr})
            params.append({'params': self.bg_net.parameters(), 'lr': lr})

        return params

    def semantic(self, geo_features, sigma):
        """
        features: [N, D] geometric features
        sigma: [N, 1] density outputs
        returns: [N, C] semantic head outputs
        """
        sem_features = self.semantic_features(geo_features)
        features = torch.cat([F.relu(sem_features), geo_features], dim=1)
        return self.semantic_out(features), sem_features

    def network_parameters(self):
        """
        return: list of parameters in the neural networks, excluding encoder parameters
        """
        return (list(self.sigma_net.parameters()) +
                list(self.color_net.parameters()) +
                list(self.semantic_features.parameters()) +
                list(self.semantic_out.parameters()))


class Autoencoder(nn.Module):

    def __init__(self, in_features, bottleneck):
        super().__init__()
        self.encoder = tcnn.Network(n_input_dims=in_features,
                                    n_output_dims=bottleneck,
                                    network_config={
                                        "otype": "CutlassMLP",
                                        "activation": "ReLU",
                                        "output_activation": "ReLU",
                                        "n_neurons": 128,
                                        "n_hidden_layers": 1
                                    })
        self.decoder = tcnn.Network(n_input_dims=bottleneck,
                                    n_output_dims=in_features,
                                    network_config={
                                        "otype": "CutlassMLP",
                                        "activation": "ReLU",
                                        "output_activation": "None",
                                        "n_neurons": 128,
                                        "n_hidden_layers": 1
                                    })

    def forward(self, x, p=0.1):
        code = self.encoder(x)
        out = self.decoder(F.dropout(code, 0.1))
        return out, code
