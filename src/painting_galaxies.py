import h5py
import scipy
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import SubsetRandomSampler
from torch_geometric.data import Data
from torch_geometric.loader import NeighborLoader, DataLoader, RandomNodeLoader, ClusterData, ClusterLoader
from torch_geometric.nn import MessagePassing, SAGEConv

from torch_geometric.utils import index_to_mask, to_undirected, remove_self_loops
from torch_cluster import radius_graph
from torch_scatter import scatter_add

import numpy as np
import pandas as pd
from pathlib import Path
import pickle
import matplotlib.pyplot as plt
import illustris_python as il

from tqdm import tqdm
from easyquery import Query
import gc
import argparse
import random

parser = argparse.ArgumentParser(description='Supply aggregation function and whether loops are used.')
parser.add_argument('--aggr', help='Aggregation function: "sum", "max", or "multi"', required=True, type=str)
parser.add_argument('--loops', help='Whether to use self-loops: "True" or "False"', required=True, type=int)
parser.add_argument('--mode', help='Training dark matter versus hydro : "DMO" or "Hydro"', required=True, type=str)

args = parser.parse_args()

ROOT = Path(__file__).parent.parent.resolve()
tng_base_path = f"{ROOT}/illustris_data/TNG300-1/output"

seed = 255
rng = np.random.RandomState(seed)
random.seed(seed)
torch.manual_seed(seed)

c0, c1, c2, c3, c4 = '#003f5c', '#58508d', '#bc5090', '#ff6361', '#ffa600'

device = "cuda" if torch.cuda.is_available() else "cpu"

### params for generating data products, visualizations, etc.
recompile_data = False
retrain = True
revalidate = True
make_plots = True
save_models = True

cuts = {
    "minimum_log_stellar_mass": 9,
    "minimum_log_halo_mass": 11,
    "minimum_n_star_particles": 50
}

boxsize = (205 / 0.6774)    # box size in comoving kpc/h
h = 0.6774                  # reduced Hubble constant
snapshot = 99,              # z = 0

normalization_params = dict(
    norm_half_mass_radius=8., 
    norm_velocity=100.
)

# training and optimization params
training_params = dict(
    min_batch_size=128,
    batch_size=1024,
    n_epochs=1000,
    learning_rate=1e-2,
    weight_decay=1e-2,
    num_parts=48,
    augment=True
)

model_params = dict(
    n_layers=1,
    n_hidden=64,
    n_latent=16,
    n_unshared_layers=8,
)

# GNN params
undirected = True
periodic = True


def load_subhalos(hydro=True, normalization_params=normalization_params, cuts=cuts, snapshot=99):
    
    use_cols = ['subhalo_x', 'subhalo_y', 'subhalo_z', 'subhalo_vx', 'subhalo_vy', 'subhalo_vz', 'subhalo_loghalomass', 'subhalo_logvmax'] 
    y_cols = ['subhalo_logstellarmass']

 
    base_path = tng_base_path.replace("TNG300-1", "TNG300-1-Dark") if not hydro else tng_base_path

    subhalo_fields = [
        "SubhaloPos", "SubhaloMassType", "SubhaloLenType", "SubhaloHalfmassRadType", 
        "SubhaloVel", "SubhaloVmax", "SubhaloGrNr", 
    ]

    if hydro:
        subhalo_fields += ["SubhaloFlag"]

    subhalos = il.groupcat.loadSubhalos(base_path, snapshot, fields=subhalo_fields) 

    pos = subhalos["SubhaloPos"][:,:3]

    halo_fields = ["Group_M_Crit200", "GroupFirstSub", "GroupPos", "GroupVel"]
    halos = il.groupcat.loadHalos(base_path, snapshot, fields=halo_fields)

    subhalo_pos = subhalos["SubhaloPos"][:] / (h*1e3)
    subhalo_stellarmass = subhalos["SubhaloMassType"][:,4]
    subhalo_halomass = subhalos["SubhaloMassType"][:,1]
    subhalo_n_stellar_particles = subhalos["SubhaloLenType"][:,4]
    subhalo_stellarhalfmassradius = subhalos["SubhaloHalfmassRadType"][:,4]  / normalization_params["norm_half_mass_radius"]
    subhalo_vel = subhalos["SubhaloVel"][:] /  normalization_params["norm_velocity"]
    subhalo_vmax = subhalos["SubhaloVmax"][:] / normalization_params["norm_velocity"]
    subhalo_flag = subhalos["SubhaloFlag"][:] if hydro else np.ones_like(subhalo_halomass) # note dummy values of 1 if DMO
    halo_id = subhalos["SubhaloGrNr"][:].astype(int)

    halo_mass = halos["Group_M_Crit200"][:]
    halo_primarysubhalo = halos["GroupFirstSub"][:].astype(int)
    group_pos = halos["GroupPos"][:] / (h*1e3)
    group_vel = halos["GroupVel"][:]  / normalization_params["norm_velocity"]

    halos = pd.DataFrame(
        np.column_stack((np.arange(len(halo_mass)), group_pos, group_vel, halo_mass, halo_primarysubhalo)),
        columns=['halo_id', 'halo_x', 'halo_y', 'halo_z', 'halo_vx', 'halo_vy', 'halo_vz', 'halo_mass', 'halo_primarysubhalo']
    )
    halos['halo_id'] = halos['halo_id'].astype(int)
    halos.set_index("halo_id", inplace=True)
    
    # get subhalos/galaxies      
    subhalos = pd.DataFrame(
        np.column_stack([halo_id, subhalo_flag, np.arange(len(subhalo_stellarmass)), subhalo_pos, subhalo_vel, subhalo_n_stellar_particles, subhalo_stellarmass, subhalo_halomass, subhalo_stellarhalfmassradius, subhalo_vmax]), 
        columns=['halo_id', 'subhalo_flag', 'subhalo_id', 'subhalo_x', 'subhalo_y', 'subhalo_z', 'subhalo_vx', 'subhalo_vy', 'subhalo_vz', 'subhalo_n_stellar_particles', 'subhalo_stellarmass', 'subhalo_halomass', 'subhalo_stellarhalfmassradius', 'subhalo_vmax'],
    )
    subhalos["is_central"] = (halos.loc[subhalos.halo_id]["halo_primarysubhalo"].values == subhalos["subhalo_id"].values)
    
    # DMO-hydro matching
    if hydro:
        dmo_hydro_match = h5py.File(f"{ROOT}/illustris_data/TNG300-1/postprocessing/subhalo_matching_to_dark.hdf5")
        subhalos["subhalo_l_halo_tree"] = dmo_hydro_match["Snapshot_99"]["SubhaloIndexDark_LHaloTree"]
        subhalos["subhalo_sublink"] = dmo_hydro_match["Snapshot_99"]["SubhaloIndexDark_SubLink"]      

    # only drop in hydro
    if hydro:
        subhalos = subhalos[subhalos["subhalo_flag"] != 0].copy()
    subhalos['halo_id'] = subhalos['halo_id'].astype(int)
    subhalos['subhalo_id'] = subhalos['subhalo_id'].astype(int)

    subhalos["subhalo_logstellarmass"] = np.log10(subhalos["subhalo_stellarmass"] / h)+10
    subhalos["subhalo_loghalomass"] = np.log10(subhalos["subhalo_halomass"] / h)+10
    subhalos["subhalo_logvmax"] = np.log10(subhalos["subhalo_vmax"])
    subhalos["subhalo_logstellarhalfmassradius"] = np.log10(subhalos["subhalo_stellarhalfmassradius"])
        
    if hydro:
        # DM cuts -- do not put outside `if` statement, otherwise indices will be missing during the `prepare_subhalos()` call
        subhalos.drop("subhalo_flag", axis=1, inplace=True)
        subhalos = subhalos[subhalos["subhalo_loghalomass"] > cuts["minimum_log_halo_mass"]].copy()
        # stellar mass and particle cuts
        subhalos = subhalos[subhalos["subhalo_n_stellar_particles"] > cuts["minimum_n_star_particles"]].copy()
        subhalos = subhalos[subhalos["subhalo_logstellarmass"] > cuts["minimum_log_stellar_mass"]].copy()
    
    
    return subhalos

def prepare_subhalos(dmo_link_method="sublink", cuts=cuts):
    """Helper function to load subhalos, join to DMO simulation, add cosmic web
    parameters, and impose cuts.

    Note that we force LHaloTree and sublink DMO linking methods to match.
    """
    
    subhalos = load_subhalos()
    subhalos_dmo = load_subhalos(hydro=False)

    valid_idxs_l_halo_tree = subhalos_dmo.index.isin(subhalos.subhalo_l_halo_tree)
    subhalos_linked = pd.concat(
        [
            (
                subhalos_dmo
                .loc[subhalos.subhalo_l_halo_tree[subhalos.subhalo_l_halo_tree != -1]]
                .reset_index(drop=True)
                .rename({c: c+"_DMO" for c in subhalos_dmo.columns}, axis=1)
            ),
            subhalos[subhalos.subhalo_l_halo_tree != -1].reset_index(drop=True),
        ], 
        axis=1,
    )

    # force l_halo_tree and sublink to match
    subhalos_linked = subhalos_linked[subhalos_linked.subhalo_l_halo_tree == subhalos_linked.subhalo_sublink].copy()
    
    # join with cosmic web parameters
    cw = h5py.File(f"{ROOT}/illustris_data/TNG300-1/postprocessing/disperse/disperse_099.hdf5")
    cw = pd.DataFrame(
        {k: cw[k] for k in cw.keys()}
    ).rename({"subhalo_ID": "subhalo_id"}, axis=1).set_index("subhalo_id")

    cw_normalization = dict(mean=cw.mean(0), std=cw.std(0))
    cw = (cw - cw_normalization["mean"]) / cw_normalization["std"]

    subhalos_linked = subhalos_linked.join(cw, on="subhalo_id", how="left")
    
    # reiterate halo mass cuts just in case DMO masses are different...
    subhalos_linked = subhalos_linked[subhalos_linked.subhalo_loghalomass_DMO > cuts["minimum_log_halo_mass"]].copy()
    subhalos_linked.dropna(inplace=True, axis=0)

    return subhalos_linked

def make_cosmic_graph(subhalos, D_link):
    df = subhalos.copy()

    subhalo_id = torch.tensor(df.index.values, dtype=torch.long)

    df.reset_index(drop=True)

    # DMO only properties
    x = torch.tensor(df[['subhalo_loghalomass_DMO', 'subhalo_logvmax_DMO']].values, dtype=torch.float)

    # hydro properties
    x_hydro = torch.tensor(df[["subhalo_loghalomass", 'subhalo_logvmax']].values, dtype=torch.float)

    # hydro total stellar mass
    y = torch.tensor(df[['subhalo_logstellarmass']].values, dtype=torch.float)

    # phase space coordinates
    pos = torch.tensor(df[['subhalo_x_DMO', 'subhalo_y_DMO', 'subhalo_z_DMO']].values, dtype=torch.float)
    vel = torch.tensor(df[['subhalo_vx_DMO', 'subhalo_vy_DMO', 'subhalo_vz_DMO']].values, dtype=torch.float)

    pos_hydro = torch.tensor(df[['subhalo_x', 'subhalo_y', 'subhalo_z']].values, dtype=torch.float)
    vel_hydro = torch.tensor(df[['subhalo_vx', 'subhalo_vy', 'subhalo_vz']].values, dtype=torch.float)

    is_central = torch.tensor(df[['is_central']].values, dtype=torch.int)
    halfmassradius = torch.tensor(df[['subhalo_logstellarhalfmassradius']].values, dtype=torch.float)

    # cw params
    cw_params = torch.tensor(df[["d_minima", "d_node", "d_saddle_1", "d_saddle_2", "d_skel"]].values, dtype=torch.float)

    # make links
    kd_tree = scipy.spatial.KDTree(pos, leafsize=25, boxsize=boxsize)
    edge_index = kd_tree.query_pairs(r=D_link, output_type="ndarray").astype(int)

    # normalize positions
    df[['subhalo_x', 'subhalo_y', 'subhalo_z']] = df[['subhalo_x', 'subhalo_y', 'subhalo_z']]/(boxsize/2)

    # add reverse pairs
    edge_index = to_undirected(torch.Tensor(edge_index).t().contiguous().type(torch.long)).numpy()

    # Write in pytorch-geometric format
    edge_index = edge_index.reshape((2,-1))
    num_pairs = edge_index.shape[1]

    row, col = edge_index

    diff = pos[row]-pos[col]
    dist = np.linalg.norm(diff, axis=1)

    use_gal = True

    if periodic:
        # Take into account periodic boundary conditions, correcting the distances
        for i, pos_i in enumerate(diff):
            for j, coord in enumerate(pos_i):
                if coord > D_link:
                    diff[i,j] -= boxsize  # Boxsize normalize to 1
                elif -coord > D_link:
                    diff[i,j] += boxsize  # Boxsize normalize to 1

    centroid = np.array(pos.mean(0)) # define arbitrary coordinate, invarinat to translation/rotation shifts, but not stretches
    # centroid+=1.2

    unitrow = (pos[row]-centroid)/np.linalg.norm((pos[row]-centroid), axis=1).reshape(-1,1)
    unitcol = (pos[col]-centroid)/np.linalg.norm((pos[col]-centroid), axis=1).reshape(-1,1)
    unitdiff = diff/dist.reshape(-1,1)
    # Dot products between unit vectors
    cos1 = np.array([np.dot(unitrow[i,:], unitcol[i,:]) for i in range(num_pairs)])
    cos2 = np.array([np.dot(unitrow[i,:], unitdiff[i,:]) for i in range(num_pairs)])

    # same invariant edge features but for velocity
    velnorm = np.vstack(df[['subhalo_vx', 'subhalo_vy', 'subhalo_vz']].to_numpy())
    vel_diff = velnorm[row]-velnorm[col]
    vel_norm = np.linalg.norm(vel_diff, axis=1)
    vel_centroid = np.array(velnorm.mean(0))

    vel_unitrow = (velnorm[row]-vel_centroid)/np.linalg.norm(velnorm[row]-vel_centroid, axis=1).reshape(-1, 1)
    vel_unitcol = (velnorm[col]-vel_centroid)/np.linalg.norm(velnorm[col]-vel_centroid, axis=1).reshape(-1, 1)
    vel_unitdiff = vel_diff / vel_norm.reshape(-1,1)
    vel_cos1 = np.array([np.dot(vel_unitrow[i,:], vel_unitcol[i,:]) for i in range(num_pairs)])
    vel_cos2 = np.array([np.dot(vel_unitrow[i,:], vel_unitdiff[i,:]) for i in range(num_pairs)])

    # build edge features
    edge_attr = np.concatenate([dist.reshape(-1,1), cos1.reshape(-1,1), cos2.reshape(-1,1), vel_norm.reshape(-1,1), vel_cos1.reshape(-1,1), vel_cos2.reshape(-1,1)], axis=1)

    # use loops here no matter what; can always remove later
    loops = np.zeros((2,pos.shape[0]),dtype=int)
    atrloops = np.zeros((pos.shape[0], edge_attr.shape[1]))
    for i, posit in enumerate(pos):
        loops[0,i], loops[1,i] = i, i
        atrloops[i,0], atrloops[i,1], atrloops[i,2] = 0., 1., 0.
    edge_index = np.append(edge_index, loops, 1)
    edge_attr = np.append(edge_attr, atrloops, 0)
    edge_index = edge_index.astype(int)

    # optimized way to compute overdensity
    source_nodes = edge_index[0]
    target_nodes = edge_index[1]
    overdensity = torch.log10(
        scatter_add(
            10**(torch.tensor(x[source_nodes, 0])), 
            index=torch.tensor(target_nodes), 
            dim=0, 
            dim_size=len(x)
        )
    )

    data = Data(
        x=x,
        edge_index=torch.tensor(edge_index, dtype=torch.long),
        edge_attr=torch.tensor(edge_attr, dtype=torch.float),
        y=y,
        pos=pos,
        vel=vel,
        is_central=is_central,
        x_hydro=x_hydro,
        pos_hydro=pos_hydro,
        vel_hydro=vel_hydro,
        halfmassradius=halfmassradius,
        subhalo_id=subhalo_id,
        overdensity=overdensity,
        cw_params=cw_params
    )

    return data

                    
def visualize_graph(data, draw_edges=True, projection="3d", edge_index=None, boxsize=302.6, fontsize=12, results_path=None):

    fig = plt.figure(figsize=(6, 6), dpi=300)

    if projection=="3d":
        ax = fig.add_subplot(projection="3d")
        pos = boxsize/2*data.pos[:,:3]
        mass = data.x[:,0]
    elif projection=="2d":
        ax = fig.add_subplot()
        pos = boxsize/2*data.pos[:,:2]
        mass = data.x[:,0]

    # Draw lines for each edge
    if data.edge_index is not None and draw_edges:
        for (src, dst) in data.edge_index.t().tolist():

            src = pos[src].tolist()
            dst = pos[dst].tolist()
            if projection=="3d":
                ax.plot([src[0], dst[0]], [src[1], dst[1]], zs=[src[2], dst[2]], linewidth=0.2/D_link, color='black')
            elif projection=="2d":
                ax.plot([src[0], dst[0]], [src[1], dst[1]], linewidth=0.2/D_link, color='black')

    # Plot nodes
    if projection=="3d":
        sc = ax.scatter(pos[:, 0], pos[:, 1], pos[:, 2], s=4**(mass - 10), zorder=1000, vmin=11, vmax=13.5, alpha=0.9, edgecolor='k', c=mass, cmap="plasma", linewidths=0.1)
    elif projection=="2d":
        sc = ax.scatter(pos[:, 0], pos[:, 1], s=4**(mass - 10), zorder=1000, alpha=0.9, edgecolor='k', c=mass, vmin=11, vmax=13.5,  cmap="plasma", linewidths=0.1)
    plt.subplots_adjust(right=0.8)

    if projection == "3d":
        cb = fig.colorbar(sc, shrink=0.8, aspect=50, location='top', pad=-0.03)
    else:
        cb = fig.colorbar(sc, shrink=0.805, aspect=40, location="top", pad=0.03)
    cb.set_label("log($M_{\\rm halo}/M_{\\odot})$", fontsize=fontsize)

    ax.xaxis.set_tick_params(labelsize=fontsize)
    ax.yaxis.set_tick_params(labelsize=fontsize)
    ax.set_xlabel("X [Mpc]", fontsize=fontsize)
    ax.set_ylabel("Y [Mpc]", fontsize=fontsize)

    if projection=="3d": 
        ax.zaxis.set_tick_params(labelsize=fontsize)
        ax.set_zlabel("Z [Mpc]", fontsize=fontsize)
        ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        ax.xaxis._axinfo["grid"]['color'] =  (0.5,0.5,0.5,0.2)
        ax.yaxis._axinfo["grid"]['color'] =  (0.5,0.5,0.5,0.2)
        ax.zaxis._axinfo["grid"]['color'] =  (0.5,0.5,0.5,0.2)
    else:
        ax.set_aspect("equal")
        ax.grid(alpha=0.15)

    fig.tight_layout()
    if projection == "3d":
        fig.savefig(f"{results_path}/cosmic-graph.png", dpi=300)
    else:
        fig.savefig(f"{results_path}/cosmic-graph-projection.png", dpi=300)
    
    plt.close()

        
class EdgeInteractionLayer(MessagePassing):
    """Graph interaction layer that combines node & edge features on edges.
    """
    def __init__(self, n_in, n_hidden, n_latent, aggr='sum', act_fn=nn.SiLU):
        super(EdgeInteractionLayer, self).__init__(aggr)

        self.mlp = nn.Sequential(
            nn.Linear(n_in, n_hidden, bias=True),
            nn.LayerNorm(n_hidden),
            act_fn(),
            nn.Linear(n_hidden, n_hidden, bias=True),
            nn.LayerNorm(n_hidden),
            act_fn(),
            nn.Linear(n_hidden, n_latent, bias=True),
        )

    def forward(self, x, edge_index, edge_attr):
        return self.propagate(edge_index, x=x, edge_attr=edge_attr)

    def message(self, x_i, x_j, edge_attr):
        inputs = torch.cat([x_i, x_j, edge_attr], dim=-1)
        return self.mlp(inputs)


class EdgeInteractionGNN(nn.Module):
    """Graph net over nodes and edges with multiple unshared layers, and sequential layers with residual connections.
    Self-loops also get their own MLP (i.e. galaxy-halo connection).
    """
    def __init__(self, n_layers, node_features=2, edge_features=6, hidden_channels=64, aggr="sum", latent_channels=64, n_out=1, n_unshared_layers=4, act_fn=nn.SiLU):
        super(EdgeInteractionGNN, self).__init__()

        self.n_in = 2 * node_features + edge_features 
        self.n_out = n_out
        self.n_pool = (len(aggr) if isinstance(aggr, list) else 1) 
        
        layers = [
            nn.ModuleList([
                EdgeInteractionLayer(self.n_in, hidden_channels, latent_channels, aggr=aggr, act_fn=act_fn)
                for _ in range(n_unshared_layers)
            ])
        ]
        for _ in range(n_layers-1):
            layers += [
                nn.ModuleList([
                    EdgeInteractionLayer(self.n_pool * (2 * latent_channels * n_unshared_layers + node_features), hidden_channels, latent_channels, aggr=aggr, act_fn=act_fn) 
                    for _ in range(n_unshared_layers)
                ])
            ]
   
        self.layers = nn.ModuleList(layers)
        
        self.galaxy_environment_mlp = nn.Sequential(
            nn.Linear(self.n_pool * n_unshared_layers * latent_channels, hidden_channels, bias=True),
            nn.LayerNorm(hidden_channels),
            act_fn(),
            nn.Linear(hidden_channels, hidden_channels, bias=True),
            nn.LayerNorm(hidden_channels),
            act_fn(),
            nn.Linear(hidden_channels, latent_channels, bias=True)
        )

        self.galaxy_halo_mlp = nn.Sequential(
            nn.Linear(node_features, hidden_channels, bias=True),
            nn.LayerNorm(hidden_channels),
            act_fn(),
            nn.Linear(hidden_channels, hidden_channels, bias=True),
            nn.LayerNorm(hidden_channels),
            act_fn(),
            nn.Linear(hidden_channels, latent_channels, bias=True)
        )

        self.fc = nn.Sequential(
            nn.Linear(2 * latent_channels, hidden_channels, bias=True),
            nn.LayerNorm(hidden_channels),
            act_fn(),
            nn.Linear(hidden_channels, hidden_channels, bias=True),
            nn.LayerNorm(hidden_channels),
            act_fn(),
            nn.Linear(hidden_channels, 2 * n_out, bias=True)
        )
    
    def forward(self, data):
        
        # determine edges by getting neighbors within radius defined by `D_link`
        edge_index = data.edge_index
        edge_attr = data.edge_attr

        # update hidden state on edge (h, or sometimes e_ij in the text)
        h = torch.cat(
            [
                unshared_layer(data.x, edge_index=edge_index, edge_attr=edge_attr)
                for unshared_layer in self.layers[0]
            ], 
            axis=1
        )
        
        for layer in self.layers[1:]:
            # if multiple layers deep, also use a residual layer
            h += torch.cat(
                [
                    unshared_layer(h, edge_index=edge_index, edge_attr=edge_attr) 
                    for unshared_layer in layer
                ], 
                axis=1
            )
                        
        out =  torch.cat([self.galaxy_environment_mlp(h), self.galaxy_halo_mlp(data.x)], axis=1)
        return self.fc(out)

class SAGEGraphConvNet(torch.nn.Module):
    """A simple GNN built using SAGEConv layers.
    """
    def __init__(self, n_in=3, n_hidden=256, n_out=1):
        super(SAGEGraphConvNet, self).__init__()
        self.n_in = n_in
        self.n_hidden = n_hidden
        self.n_out = n_out
        self.conv1 = SAGEConv(self.n_in, self.n_hidden)
        self.conv2 = SAGEConv(self.n_hidden, self.n_hidden)
        # self.fc = nn.Linear(n_hidden, n_out, bias=True)
        self.mlp = nn.Sequential(
            nn.Linear(self.n_in + 2 * self.n_hidden, self.n_hidden, bias=True),
            nn.ReLU(),
            nn.LayerNorm(self.n_hidden),
            nn.Linear(self.n_hidden, 2*self.n_out, bias=True)
        )

    def forward(self, data):
        x0, edge_index = data.x, data.edge_index

        x1 = self.conv1(x0, edge_index)
        x2 = self.conv2(F.relu(x1), edge_index)
        mlp_out = self.mlp(torch.cat([x0, F.relu(x1), F.relu(x2)], dim=-1))
        return mlp_out       

def get_train_valid_indices(data, k, K=3, boxsize=205/0.6774, pad=10, epsilon=1e-10):
    """k must be between `range(0, K)`. 

    `boxsize` and `pad` are both in units of Mpc, and it is assumed that the 
    `data` object has attribute `pos` of shape (N_rows, 3) also in units of Mpc.

    `epsilon` is there so that the modular division doesn't cause the boolean
    logic to wrap around.
    """

    # use z coordinate for train-valid split
    train_1_mask = (
        (data.pos[:, 2]  > ((k) / K * boxsize + pad) % boxsize) 
        & (data.pos[:, 2] <= ((k + 1) / K * boxsize - epsilon) % boxsize)
    )

    train_2_mask = (
        (data.pos[:, 2]  > ((k + 1)/ K * boxsize) % boxsize) 
        & (data.pos[:, 2] <= ((k + 2) / K * boxsize - pad) % boxsize)
    )

    valid_mask = (
        (data.pos[:, 2] > ((k + 2) / K * boxsize) % boxsize)
        & (data.pos[:, 2] <= ((k + 3) / K * boxsize - epsilon) % boxsize)
    )

    # this is the weird pytorch way of doing `np.argwhere`
    train_indices = (train_1_mask  | train_2_mask).nonzero(as_tuple=True)[0] 

    valid_indices = valid_mask.nonzero(as_tuple=True)[0]

    # ensure zero overlap
    assert (set(train_indices) & set(valid_indices)) == set()

    return train_indices, valid_indices


def train(dataloader, model, optimizer, device, augment=True, max_grad_norm=None):
    """Train GNN model using Gaussian NLL loss."""
    model.train()

    loss_total = 0
    for data in (dataloader):
        if augment: # add random noise
            data_node_features_scatter = 3e-4 * torch.randn_like(data.x) * torch.std(data.x, dim=0)
            data_edge_features_scatter = 3e-4 * torch.randn_like(data.edge_attr) * torch.std(data.edge_attr, dim=0)
            
            data.x += data_node_features_scatter
            data.edge_attr += data_edge_features_scatter

            assert not torch.isnan(data.x).any() 
            assert not torch.isnan(data.edge_attr).any() 

        data.to(device)

        optimizer.zero_grad()
        y_pred, logvar_pred = model(data).chunk(2, dim=1)
        assert not torch.isnan(y_pred).any() and not torch.isnan(logvar_pred).any()

        y_pred = y_pred.view(-1, model.n_out)
        logvar_pred = logvar_pred.mean()
        loss = 0.5 * (F.mse_loss(y_pred, data.y) / 10**logvar_pred + logvar_pred)

        loss.backward()
        # use gradient clipping to prevent exploding gradients
        if max_grad_norm is not None:
            nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()
        loss_total += loss.item()

    return loss_total / len(dataloader)


def validate(dataloader, model, device):
    model.eval()

    loss_total = 0

    y_preds = []
    y_trues = []


    for data in (dataloader):
        with torch.no_grad():
            data.to(device)
            y_pred, logvar_pred = model(data).chunk(2, dim=1)
            y_pred = y_pred.view(-1, model.n_out)
            logvar_pred = logvar_pred.mean()
            loss = 0.5 * (F.mse_loss(y_pred, data.y) / 10**logvar_pred + logvar_pred)

            loss_total += loss.item()
            y_preds += list(y_pred.detach().cpu().numpy())
            y_trues += list(data.y.detach().cpu().numpy())


    y_preds = np.concatenate(y_preds)
    y_trues = np.array(y_trues)

    return (
        loss_total / len(dataloader),
        y_preds,
        y_trues,
    )

def configure_optimizer(model, lr, wd,):
    """Only apply weight decay to weights, but not to other
    parameters like biases or LayerNorm. Based on minGPT version.
    """

    decay, no_decay = set(), set()
    yes_wd_modules = (nn.Linear, )
    no_wd_modules = (nn.LayerNorm, )
    for mn, m in model.named_modules():
        for pn, p in m.named_parameters():
            fpn = '%s.%s' % (mn, pn) if mn else pn
            if pn.endswith('bias'):
                no_decay.add(fpn)
            elif pn.endswith('weight') and isinstance(m, yes_wd_modules):
                decay.add(fpn)
            elif pn.endswith('weight') and isinstance(m, no_wd_modules):
                no_decay.add(fpn)
    param_dict = {pn: p for pn, p in model.named_parameters()}

    optim_groups = [
        {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": wd},
        {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.},
    ]

    optimizer = torch.optim.AdamW(
        optim_groups, 
        lr=lr, 
    )

    return optimizer



def get_metrics(p, y):
    """Returns a bunch of metrics for any model (RF, GNN) prediction"""
    rmse = np.sqrt(np.mean((p-y)**2))
    nmad = median_abs_deviation((p-y), scale="normal")
    mae = np.mean(np.absolute(p-y))
    pearson_rho = np.corrcoef(p, y)[0,1]
    r2 = 1 - (np.sum((p-y)**2) / np.sum((y - y.mean())**2))
    bias = np.mean(p - y)*1e3
    f_outlier = np.mean(np.absolute(p-y) > 3*nmad) * 100

    return rmse, nmad, mae, pearson_rho, r2, bias, f_outlier


def save_metrics(df, results_path=None):
    """Save LaTeX table of results"""
    
    with open(f"{results_path}/metrics.tex", "w") as f:
        
        metrics_Mh = get_metrics(df.p_RF_Mhalo, df.log_Mstar)
        f.write("RF - $M_{\\rm halo}$ & 1 & " + " & ".join([f"{m:.3f}" for m in metrics_Mh]) + "\\\\" + "\n")

        metrics_Vmax = get_metrics(df.p_RF_Vmax, df.log_Mstar)
        f.write("RF - $V_{\\rm max}$ & 1 & " + " & ".join([f"{m:.3f}" for m in metrics_Vmax]) + "\\\\" + "\n")

        metrics_MhVmax = get_metrics(df.p_RF_MhVmax, df.log_Mstar)
        f.write("RF - $M_{\\rm halo}+V_{\\rm max}$ & 2 & " + " & ".join([f"{m:.3f}" for m in metrics_MhVmax]) + "\\\\" + "\n")

        metrics_overdensity = get_metrics(df.p_RF_overdensity, df.log_Mstar)
        f.write("RF - $M_{\\rm halo}+V_{\\rm max}+" + f"\\delta_{D_link}$ & 2 & " + " & ".join([f"{m:.3f}" for m in metrics_overdensity]) + "\\\\" + "\n")

        metrics_GNN = get_metrics(df.p_GNN, df.log_Mstar)
        f.write("\\bf GNN & 8 & \\bf " + " & \\bf ".join([f"{m:.3f}" for m in metrics_GNN]) + "\\\\" + "\n")
        
        if "is_central" in df.columns:

            metrics_GNN_centrals = get_metrics(df[df.is_central].p_GNN, df[df.is_central].log_Mstar)
            f.write("GNN $(3d)$ - centrals & 8 & " + " & ".join([f"{m:.3f}" for m in metrics_GNN_centrals]) + "\\\\"+ "\n")

            metrics_GNN_satellites = get_metrics(df[~df.is_central].p_GNN, df[~df.is_central].log_Mstar)
            f.write("GNN $(3d)$ - satellites & 8 & " + " & ".join([f"{m:.3f}" for m in metrics_GNN_satellites]) + "\\\\"+ "\n")
    
def plot_comparison_figure(df, results_path=None):
    
    sc_kwargs = dict(edgecolor='white', s=3, linewidths=0.1, cmap=cmr.dusk, vmin=11, vmax=14)
    
    fig = plt.figure(figsize=(14, 3.75), dpi=300, constrained_layout=True)
    
    gs = fig.add_gridspec(1, 4, wspace=0.05, left=0.05, right=0.95, bottom=0.025, top=0.975, )
    ax1, ax2, ax3, ax4 = gs.subplots(sharey="row")

    ax1.scatter(df.log_Mstar, df.p_RF_Vmax, c=df.log_Mhalo, **sc_kwargs)
    ax1.text(0.025, 0.96, f"RF: $V_{{\\rm max}}$\n{np.sqrt(np.mean((df.p_RF_Vmax - df.log_Mstar)**2)):.3f} dex", va="top", transform=ax1.transAxes, fontsize=16)

    ax2.scatter(df.log_Mstar, df.p_RF_MhVmax, c=df.log_Mhalo, **sc_kwargs)
    ax2.text(0.025, 0.96, f"RF: $V_{{\\rm max}}+M_{{\\rm halo}}$\n{np.sqrt(np.mean((df.p_RF_MhVmax - df.log_Mstar)**2)):.3f} dex", va="top", transform=ax2.transAxes, fontsize=16)

    ax3.scatter(df.log_Mstar, df.p_RF_overdensity, c=df.log_Mhalo, **sc_kwargs)
    ax3.text(0.025, 0.96, f"RF: $V_{{\\rm max}}+M_{{\\rm halo}}+\\delta$\n{np.sqrt(np.mean((df.p_RF_overdensity - df.log_Mstar)**2)):.3f} dex", va="top", transform=ax3.transAxes, fontsize=16)
    
    sc = ax4.scatter(df.log_Mstar, df.p_GNN_3d, c=df.log_Mhalo, **sc_kwargs)
    ax4.text(0.025, 0.96, f"GNN\n{np.sqrt(np.mean((df.p_GNN_3d - df.log_Mstar)**2)):.3f} dex", va="top", transform=ax4.transAxes, fontsize=16)

    cb = fig.colorbar(sc, ax=[ax1, ax2, ax3, ax4], pad=0.02, shrink=0.83)
    cb.set_label("True log($M_{\\rm halo}/M_{\\odot}}$)", fontsize=14)


    for ax in [ax1, ax2, ax3, ax4]:
        ax.plot([0, 50], [0, 50], lw=1.5, c='w', zorder=9)
        ax.plot([0, 50], [0, 50], lw=1, c='0.5', zorder=10)
        ax.grid(alpha=0.15)
        ax.set_xlim(9, 12.5)
        ax.set_ylim(9, 12.5)
        ax.set_xticks([9, 10, 11, 12])
        ax.set_yticks([9, 10, 11, 12])

        if ax == ax1:
            ax.set_ylabel("Predicted log($M_{\\bigstar}/M_{\\odot}}$)", fontsize=14)
        ax.set_xlabel("True log($M_{\\bigstar}/M_{\\odot}}$)", fontsize=14)
        ax.set_aspect("equal")


    plt.savefig(f'{results_path}/GNN-vs-RF.png')
    plt.close()

    
def main(
    D_link, aggr, use_loops, mode, K=3,
    n_hidden=model_params["n_hidden"],
    n_latent=model_params["n_latent"],
    n_layers=model_params["n_layers"],
    n_unshared_layers=model_params["n_unshared_layers"],
):
    """Run the full pipeline"""

    results_path = f"{ROOT}/results/linking_length_tests/D_link{D_link}"
    
    # make paths in case they don't exist
    Path(f"{results_path}/data").mkdir(parents=True, exist_ok=True)
    Path(f"{results_path}/logs").mkdir(parents=True, exist_ok=True)
    Path(f"{results_path}/models").mkdir(parents=True, exist_ok=True)

    catalog_path = f"{results_path}/data/subhalos_DMO-matched.parquet"
    data_path = f"{results_path}/data/cosmic_graphs.pkl"
    if recompile_data or retrain or revalidate or not os.path.isfile(data_path):

        # get DMO & hydro linked catalogs with all cuts
        
        if os.path.isfile(catalog_path) and not recompile_data:
            subhalos = pd.read_parquet(catalog_path)
        else:
            subhalos = prepare_subhalos()
            subhalos.to_parquet(catalog_path)

        # make cosmic graphs
        if os.path.isfile(data_path) and not recompile_data:
            with open(data_path, "rb") as data_file:
                data = pickle.load(data_file)
        else:
            data = make_cosmic_graph(subhalos, D_link=D_link)
            with open(data_path, "wb") as data_file:
                pickle.dump(data, data_file)

        if mode.lower() == "hydro":
            data_hydro = data.clone()
            x, pos, vel = data.x, data.pos, data.vel
            x_hydro, pos_hydro, vel_hydro = data.x_hydro, data.pos_hydro, data.vel_hydro
            data_hydro.x = torch.cat((x_hydro, data.is_central.type(torch.float)), dim=-1)
            data_hydro.pos = pos_hydro
            data_hydro.vel = vel_hydro
            data = data_hydro
        else:
            data.x = torch.cat((data.x, data.is_central.type(torch.float)), dim=-1)
    else:
        subhalos = pd.read_parquet(catalog_path)
        with open(data_path, "rb") as data_file:
                data = pickle.load(data_file)

    # remove self-loops
    if not use_loops:
        data.edge_index, data.edge_attr = remove_self_loops(data.edge_index, data.edge_attr)

    # training
    node_features = data.x.shape[1]
    edge_features = data.edge_attr.shape[1]
    out_features = data.y.shape[1]

    lr = training_params["learning_rate"]
    wd = training_params["weight_decay"]
    min_batch_size = training_params["min_batch_size"]
    batch_size = training_params["batch_size"]
    n_epochs = training_params["n_epochs"]
    num_parts = training_params["num_parts"]

    for k in range(K):
        model_path = f"{results_path}/models/gnn-{training_params['n_epochs']}-{aggr}-{mode}-fold_{k+1}.pt"
        valid_results_path = f"{results_path}/gnn-{training_params['n_epochs']}-{aggr}-{mode}-fold_{k+1}.cat"
        
        if retrain or revalidate or not os.path.isfile(model_path):

            # model = SAGEGraphConvNet(n_in=node_features, n_out=out_features)
            model = EdgeInteractionGNN(
                node_features=node_features,  # note that there are 3 features counting `is_central`, but this final feature is not in `data.x`
                edge_features=edge_features, 
                n_layers=n_layers, 
                hidden_channels=n_hidden,
                latent_channels=n_latent,
                n_unshared_layers=n_unshared_layers,
                n_out=out_features,
                aggr=(["sum", "max", "mean"] if aggr == "multi" else aggr)
            )
            model.to(device)

            optimizer = configure_optimizer(model, lr, wd)
            
            train_indices, valid_indices = get_train_valid_indices(data, k=k, K=K)

            train_data = ClusterData(
                data.subgraph(train_indices), 
                num_parts=num_parts, 
                recursive=False,
            )
            valid_data = ClusterData(
                data.subgraph(train_indices), 
                num_parts=num_parts // 2, 
                recursive=False,
            )
            train_loader = ClusterLoader(
                train_data,
                shuffle=True,
                batch_size=1,
            )
            valid_loader = ClusterLoader(
                valid_data,
                shuffle=False,
                batch_size=2,
            )

            gc.collect()

            
            if retrain or not os.path.isfile(model_path):
                # training log
                train_losses = []
                valid_losses = []
                with open(f'{results_path}/logs/training-{mode}-fold_{k+1}.log', 'a') as f:

                    for epoch in range(n_epochs):
                        if epoch == int(n_epochs * 0.25):
                            optimizer = configure_optimizer(model, lr/5, wd)
                        elif epoch == (n_epochs * 0.5):
                            optimizer = configure_optimizer(model, lr/25, wd)
                        elif epoch == (n_epochs * 0.75):
                            optimizer = configure_optimizer(model, lr/125, wd)

                        train_loss = train(train_loader, model, optimizer, device)
                        valid_loss, p, y  = validate(valid_loader, model, device)

                        train_losses.append(train_loss)
                        valid_losses.append(valid_loss)


                        f.write(f"{epoch + 1: >4d}    {train_loss: >9.5f}    {valid_loss: >9.5f}    {np.sqrt(np.mean((p - y.flatten())**2)): >10.6f}\n")
                        f.flush()

                torch.save(model.state_dict(), model_path)

            if revalidate or not os.path.isfile(valid_results_path):
                # load previously trained model
                model.load_state_dict(torch.load(model_path))
                model.to(device);

                valid_loader = RandomNodeLoader(
                    data.subgraph(valid_indices),
                    num_parts=np.ceil(len(valid_indices) / batch_size),
                    shuffle=False,
                    pin_memory=True
                )

                valid_loss, p, y  = validate(valid_loader, model, device)
                assert len(data.y[valid_indices]) == len(p)

                # helper function to select the validation indices for 1-d tensors in `data`
                # and return as numpy
                select_valid = lambda data_tensor: data_tensor[valid_indices].numpy().flatten()
                results = pd.DataFrame({
                    "subhalo_id": select_valid(data.subhalo_id),
                    "log_Mstar": select_valid(data.y),
                    f"p_GNN_{mode}": p,
                    "log_Mhalo_dmo": select_valid(data.x[:, 0]),
                    "log_Vmax_dmo": select_valid(data.x[:, 1]),
                    "x_dmo": select_valid(data.pos[:, 0]),
                    "y_dmo": select_valid(data.pos[:, 1]),
                    "z_dmo": select_valid(data.pos[:, 2]),
                    "vx_dmo": select_valid(data.vel[:, 0]),
                    "vy_dmo": select_valid(data.vel[:, 1]),
                    "vz_dmo": select_valid(data.vel[:, 2]),
                    "is_central": select_valid(data.is_central).astype(int),
                    "log_Mhalo_hydro": select_valid(data.x_hydro[:, 0]),
                    "log_Vmax_hydro": select_valid(data.x_hydro[:, 1]),
                    "x_hydro": select_valid(data.pos_hydro[:, 0]),
                    "y_hydro": select_valid(data.pos_hydro[:, 1]),
                    "z_hydro": select_valid(data.pos_hydro[:, 2]),
                    "vx_hydro": select_valid(data.vel_hydro[:, 0]),
                    "vy_hydro": select_valid(data.vel_hydro[:, 1]),
                    "vz_hydro": select_valid(data.vel_hydro[:, 2]),
                    "Rstar_50": select_valid(data.halfmassradius),
                    "overdensity": select_valid(data.overdensity),
                    "d_minima": select_valid(data.cw_params[:, 0]),
                    "d_node": select_valid(data.cw_params[:, 1]),
                    "d_saddle_1": select_valid(data.cw_params[:, 2]),
                    "d_saddle_2": select_valid(data.cw_params[:, 3]),
                    "d_skel": select_valid(data.cw_params[:, 4]),
                })

                assert torch.allclose(torch.tensor(y), data.y[valid_indices])

                results.to_csv(valid_results_path, index=False)



if __name__ == "__main__":
    aggr = args.aggr
    use_loops = args.loops
    mode = args.mode

    # for D_link in [3]:
    for D_link in [0.3, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 5, 7.5, 10]:
        print(f"Working on D_link = {D_link}")
        main(D_link=D_link, aggr=aggr, use_loops=use_loops, mode=mode)
