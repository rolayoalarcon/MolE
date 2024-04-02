import os
import yaml
import numpy as np
import pandas as pd

import torch
from torch_geometric.data import Data, Dataset, Batch

from rdkit import Chem
from rdkit.Chem.rdchem import HybridizationType
from rdkit.Chem.rdchem import BondType as BT
from rdkit.Chem.Scaffolds.MurckoScaffold import MurckoScaffoldSmiles
from rdkit.Chem import AllChem
from rdkit.Chem import DataStructs
from rdkit import RDLogger                                                                                                                                                               
RDLogger.DisableLog('rdApp.*')  


ATOM_LIST = list(range(1,119))
CHIRALITY_LIST = [
    Chem.rdchem.ChiralType.CHI_UNSPECIFIED,
    Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW,
    Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW,
    Chem.rdchem.ChiralType.CHI_OTHER
]
BOND_LIST = [
    BT.SINGLE, 
    BT.DOUBLE, 
    BT.TRIPLE, 
    BT.AROMATIC
]
BONDDIR_LIST = [
    Chem.rdchem.BondDir.NONE,
    Chem.rdchem.BondDir.ENDUPRIGHT,
    Chem.rdchem.BondDir.ENDDOWNRIGHT
]

def read_smiles(data_path, smile_col="smiles"):
    
    # Read the data
    smile_df = pd.read_csv(data_path)

    # Remove NaN
    smile_df = smile_df.dropna()

    # Remove invalid smiles
    smile_df = smile_df[smile_df["smiles"].apply(lambda x: Chem.MolFromSmiles(x) is not None)]

    # Add chem_id column
    smile_df["chem_id"] = [f"chem_{i}" for i in range(smile_df.shape[0])]

    return smile_df

# Create molecular graphs
class MoleculeDataset(Dataset):
    def __init__(self, smile_df):
        super(Dataset, self).__init__()

        # Gather the SMILES and the corresponding IDs
        self.smiles_data = smile_df["smiles"].tolist()
        self.id_data = smile_df["chem_id"].tolist()

    def __getitem__(self, index):
        # Get the molecule
        mol = Chem.MolFromSmiles(self.smiles_data[index])
        mol = Chem.AddHs(mol)

        #########################
        # Get the molecule info #
        #########################
        type_idx = []
        chirality_idx = []
        atomic_number = []

        for atom in mol.GetAtoms():
            if atom.GetAtomicNum() == 0:
                print(self.id_data[index])

            type_idx.append(ATOM_LIST.index(atom.GetAtomicNum()))
            chirality_idx.append(CHIRALITY_LIST.index(atom.GetChiralTag()))
            atomic_number.append(atom.GetAtomicNum())

        x1 = torch.tensor(type_idx, dtype=torch.long).view(-1,1)
        x2 = torch.tensor(chirality_idx, dtype=torch.long).view(-1,1)
        x = torch.cat([x1, x2], dim=-1)

        row, col, edge_feat = [], [], []
        for bond in mol.GetBonds():
            start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            row += [start, end]
            col += [end, start]
            edge_feat.append([
                BOND_LIST.index(bond.GetBondType()),
                BONDDIR_LIST.index(bond.GetBondDir())
            ])
            edge_feat.append([
                BOND_LIST.index(bond.GetBondType()),
                BONDDIR_LIST.index(bond.GetBondDir())
            ])

        edge_index = torch.tensor([row, col], dtype=torch.long)
        edge_attr = torch.tensor(np.array(edge_feat), dtype=torch.long)

        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, 
                    chem_id=self.id_data[index])
        
        return data

    def __len__(self):
        return len(self.smiles_data)
    
    def get(self, index):
        self.__getitem__(index)

    def len():
        pass
    

# Function to generate the molecular scaffolds
def _generate_scaffold(smiles, include_chirality=False):
    mol = Chem.MolFromSmiles(smiles)
    scaffold = MurckoScaffoldSmiles(mol=mol, includeChirality=include_chirality)
    return scaffold


# Function to separate structures based on scaffolds
def generate_scaffolds(smile_list):
    scaffolds = {}
    data_len = len(smile_list)

    print("About to generate scaffolds")
    for ind, smiles in enumerate(smile_list):
        scaffold = _generate_scaffold(smiles)
        if scaffold not in scaffolds:
            scaffolds[scaffold] = [ind]
        else:
            scaffolds[scaffold].append(ind)

    # Sort from largest to smallest scaffold sets
    scaffolds = {key: sorted(value) for key, value in scaffolds.items()}
    scaffold_sets = [
        scaffold_set for (scaffold, scaffold_set) in sorted(
            scaffolds.items(), key=lambda x: (len(x[1]), x[1][0]), reverse=True)
    ]
    return scaffold_sets

# Separate train, validation and test sets based on scaffolds
def scaffold_split(data_df, valid_size, test_size):

    # Determine molecular scaffolds
    dataset = data_df["smiles"].tolist()
    train_size = 1.0 - valid_size - test_size
    scaffold_sets = generate_scaffolds(dataset)

    # Determine splits
    train_cutoff = train_size * len(dataset)
    valid_cutoff = (train_size + valid_size) * len(dataset)
    train_inds: List[int] = []
    valid_inds: List[int] = []
    test_inds: List[int] = []

    print("About to sort in scaffold sets")
    for scaffold_set in scaffold_sets:
        if len(train_inds) + len(scaffold_set) > train_cutoff:
            if len(train_inds) + len(valid_inds) + len(scaffold_set) > valid_cutoff:
                test_inds += scaffold_set
            else:
                valid_inds += scaffold_set
        else:
            train_inds += scaffold_set

    # Gather chem_ids based on 
    chemical_ids = data_df["chem_id"].tolist()
    train_ids = [chemical_ids[ind] for ind in train_inds]
    valid_ids = [chemical_ids[ind] for ind in valid_inds]
    test_ids = [chemical_ids[ind] for ind in test_inds]

    return train_ids, valid_ids, test_ids
    

def split_dataset(smile_df, valid_size, test_size, split_strategy):

    # Determine the splitting strategy
    if split_strategy == "random":

        # Determine the number of samples
        n_samples = smile_df.shape[0]

        # Randomly shuffle the indices
        chem_ids = smile_df["chem_id"].values
        np.random.shuffle(chem_ids)

        # Grab the validation ids
        valid_split = int(np.floor(valid_size * n_samples))
        valid_ids = chem_ids[:valid_split]

        # Grab the test ids
        test_split = int(np.floor(test_size * n_samples))
        test_ids = chem_ids[valid_split:(valid_split + test_split)]

        # Grab the train ids
        train_ids = chem_ids[(valid_split + test_split):]

    elif split_strategy == "scaffold":
        train_ids, valid_ids, test_ids = scaffold_split(smile_df, valid_size, test_size)

    # Add column with split information
    smile_df["split"]  = smile_df["chem_id"].apply(lambda x: "train" if x in train_ids else "valid" if x in valid_ids else "test")

    return smile_df


def batch_representation(smile_df, dl_model, batch_size= 10_000, id_is_str=True, device="cuda:0"):
    
    """
    Generate molecular representations in batches using a pre-trained mdoel.

    Args:
        smile_df (DataFrame): DataFrame containing SMILES representations of molecules.
        dl_model (torch.nn.Module): Deep learning model used for generating representations.
        batch_size (int, optional): Batch size for processing molecules (default is 10_000).
        id_is_str (bool, optional): Whether the molecule identifiers are strings (default is True).
        device (str, optional): Device to use for computation (default is "cuda:0").

    Returns:
        DataFrame: DataFrame containing molecular representations.

    """
    
    # First we create a list of graphs
    molecular_graph_dataset = MoleculeDataset(smile_df)
    graph_list = [g for g in molecular_graph_dataset]

    # Determine number of loops to do given the batch size
    n_batches = len(graph_list) // batch_size

    # Are all molecules accounted for?
    remaining_molecules = len(graph_list) % batch_size

    # Starting indices
    start, end = 0, batch_size

    # Determine number of iterations
    if remaining_molecules == 0:
        n_iter = n_batches
    
    elif remaining_molecules > 0:
        n_iter = n_batches + 1
    
    # A list to store the batch dataframes
    batch_dataframes = []

    # Iterate over the batches
    for i in range(n_iter):
        # Start batch object
        batch_obj = Batch()
        graph_batch = batch_obj.from_data_list(graph_list[start:end])
        graph_batch = graph_batch.to(device)

        # Gather the representation
        with torch.no_grad():
            dl_model.eval()
            h_representation, _ = dl_model(graph_batch)
            chem_ids = graph_batch.chem_id
        
        batch_df = pd.DataFrame(h_representation.cpu().numpy(), index=chem_ids)
        batch_dataframes.append(batch_df)

        # Get the next batch
        ## In the final iteration we want to get all the remaining molecules
        if i == n_iter - 2:
            start = end
            end = len(graph_list)
        else:
            start = end
            end = end + batch_size
    
    # Concatenate the dataframes
    chem_representation = pd.concat(batch_dataframes)

    return chem_representation

def load_pretrained_model(pretrain_architecture, pretrained_model, pretrained_dir = "./ckpt", device="cuda:0"):

    """
    Load a pretrained model based on the given architecture and model name.

    Args:
        pretrain_architecture (str): Architecture of the pretrained model.
        pretrained_model (str): Name of the pretrained model.
        pretrained_dir (str): Directory containing the pretrained model checkpoints.
        device (str): Device to load the model on (default is "cuda:0").

    Returns:
        torch.nn.Module: Pretrained model loaded with its weights.

    """

    # Read model configuration
    config = yaml.load(open(os.path.join(pretrained_dir, pretrained_model, "checkpoints/config.yaml"), "r"), Loader=yaml.FullLoader)
    model_config = config["model"]

    # Determine if model is MolCLR
    if pretrained_model == "MolCLR":
        from models.gin_molclr import GINet
        model = GINet(**model_config).to(device)

    # Instantiate MolE models
    elif pretrain_architecture == "gin_concat":
        from models.ginet_concat import GINet
        model = GINet(**model_config).to(device)

    elif pretrain_architecture == "gin_noconcat":
        from models.ginet_noconcat import GINet
        model = GINet(**model_config).to(device)

    elif pretrain_architecture == "gcn_concat":
        from models.gcn_concat import GCN
        model = GCN(**model_config).to(device)

    elif pretrain_architecture == "gcn_noconcat":
        from models.gcn_noconcat import GCN
        model = GCN(**model_config).to(device)
    
    # Load pre-trained weights
    model_pth_path = os.path.join(pretrained_dir, pretrained_model, "checkpoints/model.pth")
    print(model_pth_path)

    state_dict = torch.load(model_pth_path, map_location=device)
    model.load_my_state_dict(state_dict)

    return model

def fp_array(fingerprin_object):

    # Initialise an array full of zeros
    array = np.zeros((0,), dtype=np.int8)

    # Dump fingerprint info into array
    DataStructs.ConvertToNumpyArray(fingerprin_object, array)

    return array


def generate_fps(smile_df):

    # Generate fingerprints
    mol_objs = [Chem.MolFromSmiles(smile) for smile in smile_df["smiles"].tolist()]
    fp_objs = [AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024) for mol in mol_objs]

    # Place fingerprints in array
    fps_arrays = [fp_array(fp) for fp in fp_objs]

    # Create dataframe
    fps_matrix = np.stack(fps_arrays, axis=0 )
    fps_dataframe = pd.DataFrame(fps_matrix, index=smile_df.chem_id.tolist())

    return fps_dataframe

def process_dataset(dataset_path, 
                    dataset_split = True,
                    pretrained_model=None,
                    pretrain_architecture=None, 
                    split_approach=None, 
                    validation_proportion=None, 
                    test_proportion=None,
                    device="cuda:0"):
    """
    Process the dataset by reading, splitting, and generating static representations.

    Args:
        dataset_path (str): Path to the dataset file.
        dataset_split (bool): Whether to split the dataset into train, validation, and test sets.
        pretrained_model (str): Path to the pretrained model file or the name of the pretraining architecture. Can also be "MolCLR" or "ECFP4".
        pretrain_architecture (str): Pretraining architecture used for generating representations.
        split_approach (str): Method for splitting the dataset into train, validation, and test sets.
        validation_proportion (float): Proportion of the dataset to be used for validation.
        test_proportion (float): Proportion of the dataset to be used for testing.
        device (str): Device to use for computation (default is "cuda:0"). Can also be "cpu".

    Returns:
        tuple: A tuple containing the splitted dataset and its representation.

    """

    # First we read in the smiles as a dataframe
    smiles_df = read_smiles(dataset_path)

    # We split the dataset into train, validation and test if requested
    if dataset_split:
        splitted_smiles_df = split_dataset(smiles_df, validation_proportion, test_proportion, split_approach)

    # Determine the representation
    if pretrained_model == "ECFP4":
        udl_representation = generate_fps(smiles_df)
    
    else:
        # Now we load our pretrained model
        pmodel = load_pretrained_model(pretrain_architecture, pretrained_model, device=device)
        # Obtain the requested representation
        udl_representation = batch_representation(smiles_df, pmodel, device=device)

    # Determine return
    if dataset_split:
        return splitted_smiles_df, udl_representation
    else:
        return udl_representation
