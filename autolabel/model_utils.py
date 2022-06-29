import torch
import glob
from torch_ngp.nerf.network_ff import NeRFNetwork

def load_checkpoint(model, checkpoint_dir, device='cuda:0'):
    checkpoint_list = sorted(glob.glob(f'{checkpoint_dir}/*.pth'))
    checkpoint = checkpoint_list[-1]
    checkpoint_dict = torch.load(checkpoint, map_location=device)
    model.load_state_dict(checkpoint_dict['model'])
    return model

def create_model(min_bounds, max_bounds):
    extents = max_bounds - min_bounds
    bound = (extents - (min_bounds + max_bounds) * 0.5).max()
    return NeRFNetwork(num_layers=2, num_layers_color=2,
            hidden_dim_color=64,
            hidden_dim=64,
            geo_feat_dim=15,
            encoding="hashgrid",
            bound=float(bound),
            cuda_ray=False,
            density_scale=1)
