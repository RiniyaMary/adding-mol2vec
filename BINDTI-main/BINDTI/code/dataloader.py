import torch.utils.data as data
import torch
import numpy as np
from functools import partial
import dgl
from dgllife.utils import smiles_to_bigraph, CanonicalAtomFeaturizer, CanonicalBondFeaturizer
from utils import integer_label_protein
from mol2vec.features import mol2alt_sentence
from gensim.models import Word2Vec
from rdkit import Chem
import os


class DTIDataset(data.Dataset):

    def __init__(self, list_IDs, df, max_drug_nodes=290,
                 mol2vec_model_path=None, max_smiles_length=100):
        self.list_IDs = list_IDs
        self.df = df
        self.max_drug_nodes = max_drug_nodes
        self.max_smiles_length = max_smiles_length
        self.mol2vec_model = None

        # Load Mol2Vec model if provided
        if mol2vec_model_path and os.path.exists(mol2vec_model_path):
            print(f"Loading Mol2Vec model from {mol2vec_model_path}")
            try:
                self.mol2vec_model = Word2Vec.load(mol2vec_model_path)
                print(f"✅ Mol2Vec model loaded: {len(self.mol2vec_model.wv.key_to_index)} tokens")
            except Exception as e:
                print(f"❌ Failed to load Mol2Vec model: {e}")
                self.mol2vec_model = None
        else:
            print("⚠️  No Mol2Vec model provided, using zero embeddings")

        self.atom_featurizer = CanonicalAtomFeaturizer()
        self.bond_featurizer = CanonicalBondFeaturizer(self_loop=True)
        self.fc = partial(smiles_to_bigraph, add_self_loop=True)

    def __len__(self):
        return len(self.list_IDs)

    def __getitem__(self, index):
        index = self.list_IDs[index]
        smiles_str = self.df.iloc[index]['SMILES']

        # -------- Drug graph processing --------
        v_d = self.fc(
            smiles=smiles_str,
            node_featurizer=self.atom_featurizer,
            edge_featurizer=self.bond_featurizer
        )

        actual_node_feats = v_d.ndata.pop('h')
        num_actual_nodes = actual_node_feats.shape[0]

        # ===== Truncate graph AND features together (SAFE) =====
        if num_actual_nodes > self.max_drug_nodes:
            actual_node_feats = actual_node_feats[:self.max_drug_nodes]
            v_d = dgl.node_subgraph(
                v_d,
                list(range(self.max_drug_nodes)),
                store_ids=False   # 🔥 KEY FIX (prevents _ID on nodes & edges)
            )
            num_actual_nodes = self.max_drug_nodes

        # -------- Padding logic --------
        num_virtual_nodes = self.max_drug_nodes - num_actual_nodes

        virtual_node_bit = torch.zeros((num_actual_nodes, 1))
        actual_node_feats = torch.cat((actual_node_feats, virtual_node_bit), dim=1)
        v_d.ndata['h'] = actual_node_feats

        if num_virtual_nodes > 0:
            virtual_node_feat = torch.cat(
                (
                    torch.zeros((num_virtual_nodes, 74)),
                    torch.ones((num_virtual_nodes, 1))
                ),
                dim=1
            )
            v_d.add_nodes(num_virtual_nodes, {'h': virtual_node_feat})

        v_d = v_d.add_self_loop()

        # -------- Mol2Vec SMILES embedding --------
        smiles_tensor = self._smiles_to_mol2vec_embedding(smiles_str)

        # -------- Protein & label --------
        v_p = self.df.iloc[index]['Protein']
        v_p = integer_label_protein(v_p)
        y = self.df.iloc[index]['Y']

        return v_d, v_p, smiles_tensor, y

    def _smiles_to_mol2vec_embedding(self, smiles_str):
        """Convert SMILES to Mol2Vec embedding sequence"""
        if self.mol2vec_model is None:
            return torch.zeros(self.max_smiles_length, 300)

        mol = Chem.MolFromSmiles(smiles_str)
        if mol is None:
            return torch.zeros(self.max_smiles_length, 300)

        sentence = mol2alt_sentence(mol, radius=1)

        embeddings = []
        for token in sentence[:self.max_smiles_length]:
            if token in self.mol2vec_model.wv:
                embeddings.append(self.mol2vec_model.wv[token])
            else:
                embeddings.append(np.zeros(300))

        while len(embeddings) < self.max_smiles_length:
            embeddings.append(np.zeros(300))

        return torch.from_numpy(np.array(embeddings)).float()


# -------- Collate function --------
def graph_collate_func_with_smiles(batch):
    d, p, s, y = zip(*batch)
    d = dgl.batch(d)
    p = torch.tensor(np.array(p))
    s = torch.stack(s)
    y = torch.tensor(y)
    return d, p, s, y
