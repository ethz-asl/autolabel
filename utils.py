from torch_ngp.nerf.network_ff import NeRFNetwork

def create_model(dataset):
    extents = dataset.max_bounds - dataset.min_bounds
    bound = (extents - (dataset.min_bounds + dataset.max_bounds) * 0.5).max()
    return NeRFNetwork(num_layers=2, num_layers_color=2,
            hidden_dim_color=64,
            hidden_dim=64,
            geo_feat_dim=15,
            encoding="hashgrid",
            bound=float(bound),
            cuda_ray=False,
            density_scale=1)
