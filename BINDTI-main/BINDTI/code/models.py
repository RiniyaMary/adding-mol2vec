import torch.nn as nn
import torch.nn.functional as F
import torch
from dgllife.model.gnn import GCN
from ACmix import ACmix
from Intention import BiIntention

def binary_cross_entropy(pred_output, labels):
    loss_fct = torch.nn.BCELoss()
    m = nn.Sigmoid()
    n = torch.squeeze(m(pred_output), 1)
    loss = loss_fct(n, labels)
    return n, loss

def cross_entropy_logits(linear_output, label, weights=None):
    class_output = F.log_softmax(linear_output, dim=1)
    n = F.softmax(linear_output, dim=1)[:, 1]
    max_class = class_output.max(1)
    y_hat = max_class[1]  # get the index of the max log-probability
    if weights is None:
        loss = nn.NLLLoss()(class_output, label.type_as(y_hat).view(label.size(0)))
    else:
        losses = nn.NLLLoss(reduction="none")(class_output, label.type_as(y_hat).view(label.size(0)))
        loss = torch.sum(weights * losses) / torch.sum(weights)
    return n, loss


def entropy_logits(linear_output):
    p = F.softmax(linear_output, dim=1)
    loss_ent = -torch.sum(p * (torch.log(p + 1e-5)), dim=1)
    return loss_ent


class BINDTI(nn.Module):
    def __init__(self, device='cuda', **config):
        super(BINDTI, self).__init__()
        drug_in_feats = config["DRUG"]["NODE_IN_FEATS"]
        drug_embedding = config["DRUG"]["NODE_IN_EMBEDDING"]
        drug_hidden_feats = config["DRUG"]["HIDDEN_LAYERS"]
        protein_emb_dim = config["PROTEIN"]["EMBEDDING_DIM"]
        num_filters = config["PROTEIN"]["NUM_FILTERS"]
        mlp_in_dim = config["DECODER"]["IN_DIM"]
        mlp_hidden_dim = config["DECODER"]["HIDDEN_DIM"]
        mlp_out_dim = config["DECODER"]["OUT_DIM"]
        drug_padding = config["DRUG"]["PADDING"]
        protein_padding = config["PROTEIN"]["PADDING"]
        out_binary = config["DECODER"]["BINARY"]
        protein_num_head = config['PROTEIN']['NUM_HEAD']
        cross_num_head = config['CROSSINTENTION']['NUM_HEAD']
        cross_emb_dim = config['CROSSINTENTION']['EMBEDDING_DIM']
        cross_layer = config['CROSSINTENTION']['LAYER']
        
        # NEW: Get Mol2Vec config
        smiles_embedding_dim = config["DRUG"].get("MOL2VEC_EMBEDDING_DIM", 300)
        use_mol2vec = config["DRUG"].get("USE_MOL2VEC", True)

        # UPDATED: Enhanced drug extractor with Mol2Vec
        self.drug_extractor = EnhancedMolecularGCN(
            in_feats=drug_in_feats, 
            dim_embedding=drug_embedding,
            padding=drug_padding,
            hidden_feats=drug_hidden_feats,
            smiles_embedding_dim=smiles_embedding_dim,
            use_mol2vec=use_mol2vec
        )
        
        self.protein_extractor = ProteinACmix(protein_emb_dim, num_filters, protein_num_head, protein_padding)
        self.cross_intention = BiIntention(embed_dim=cross_emb_dim, num_head=cross_num_head, layer=cross_layer, device=device)
        
        # UPDATED: MLP input dimension increased to account for fused features
        enhanced_mlp_in_dim = mlp_in_dim + (smiles_embedding_dim if use_mol2vec else 0)
        self.mlp_classifier = MLPDecoder(enhanced_mlp_in_dim, mlp_hidden_dim, mlp_out_dim, binary=out_binary)

    def forward(self, bg_d, v_p, smiles_emb=None, mode="train"):
        # UPDATED: Enhanced drug features with Mol2Vec
        v_d = self.drug_extractor(bg_d, smiles_emb)  # Pass Mol2Vec embeddings
        
        v_p = self.protein_extractor(v_p)
        f, v_d, v_p, att = self.cross_intention(drug=v_d, protein=v_p)
        
        # NEW: Fuse cross-intention features with Mol2Vec features
        if smiles_emb is not None and self.drug_extractor.use_mol2vec:
            # Get global Mol2Vec representation (mean pooling)
            mol2vec_global = torch.mean(smiles_emb, dim=1)  # [batch, 300]
            # Concatenate with cross-intention features
            f = torch.cat([f, mol2vec_global], dim=1)  # [batch, 256 + 300]
        
        score = self.mlp_classifier(f)
        if mode == "train":
            return v_d, v_p, f, score
        elif mode == "eval":
            return v_d, v_p, score, att


class EnhancedMolecularGCN(nn.Module):
    def __init__(self, in_feats, dim_embedding=128, padding=True, 
                 hidden_feats=None, activation=None, smiles_embedding_dim=300, use_mol2vec=True):
        super(EnhancedMolecularGCN, self).__init__()
        
        self.use_mol2vec = use_mol2vec
        self.smiles_embedding_dim = smiles_embedding_dim
        
        # Original GCN components
        self.init_transform = nn.Linear(in_feats, dim_embedding, bias=False)
        if padding:
            with torch.no_grad():
                self.init_transform.weight[-1].fill_(0)
        self.gnn = GCN(in_feats=dim_embedding, hidden_feats=hidden_feats, activation=activation)
        self.output_feats = hidden_feats[-1]
        
        # NEW: Mol2Vec processing components
        if self.use_mol2vec:
            self.smiles_conv1d = nn.Conv1d(smiles_embedding_dim, 128, kernel_size=3, padding=1)
            self.smiles_bn1 = nn.BatchNorm1d(128)
            self.smiles_pool = nn.AdaptiveMaxPool1d(1)
            self.smiles_projection = nn.Linear(128, self.output_feats)
            
            # Fusion layer to combine GCN and Mol2Vec features
            self.fusion_layer = nn.Linear(self.output_feats * 2, self.output_feats)
            self.fusion_bn = nn.BatchNorm1d(self.output_feats)
            self.dropout = nn.Dropout(0.1)

    def forward(self, batch_graph, smiles_embeddings=None):
        # Original GCN processing
        node_feats = batch_graph.ndata.pop('h')
        node_feats = self.init_transform(node_feats)
        node_feats = self.gnn(batch_graph, node_feats)
        batch_size = batch_graph.batch_size
        gcn_features = node_feats.view(batch_size, -1, self.output_feats)  # [batch, nodes, features]
        
        # NEW: Mol2Vec sequence processing (if provided and enabled)
        if self.use_mol2vec and smiles_embeddings is not None and smiles_embeddings.shape[1] > 0:
            # Process SMILES embeddings with 1D CNN + pooling
            smiles_emb = smiles_embeddings.transpose(1, 2)  # [batch, 300, seq_len]
            smiles_features = F.relu(self.smiles_bn1(self.smiles_conv1d(smiles_emb)))
            smiles_features = self.smiles_pool(smiles_features).squeeze(-1)  # [batch, 128]
            smiles_features = self.smiles_projection(smiles_features)  # [batch, output_feats]
            smiles_features = smiles_features.unsqueeze(1).expand(-1, gcn_features.size(1), -1)  # [batch, nodes, output_feats]
            
            # Fuse GCN and Mol2Vec features
            combined_features = torch.cat([gcn_features, smiles_features], dim=2)
            fused_features = self.fusion_layer(combined_features)
            fused_features = self.fusion_bn(fused_features.transpose(1, 2)).transpose(1, 2)
            fused_features = self.dropout(F.relu(fused_features))
            return fused_features
        
        return gcn_features


class MolecularGCN(nn.Module):
    def __init__(self, in_feats, dim_embedding=128, padding=True, hidden_feats=None, activation=None):
        super(MolecularGCN, self).__init__()
        self.init_transform = nn.Linear(in_feats, dim_embedding, bias=False)
        if padding:
            with torch.no_grad():
                self.init_transform.weight[-1].fill_(0)
        self.gnn = GCN(in_feats=dim_embedding, hidden_feats=hidden_feats, activation=activation)
        self.output_feats = hidden_feats[-1]

    def forward(self, batch_graph):
        node_feats = batch_graph.ndata.pop('h')
        node_feats = self.init_transform(node_feats)
        node_feats = self.gnn(batch_graph, node_feats)
        batch_size = batch_graph.batch_size
        node_feats = node_feats.view(batch_size, -1, self.output_feats)
        return node_feats


class ProteinACmix(nn.Module):
    def __init__(self, embedding_dim, num_filters, num_head, padding=True):
        super(ProteinACmix, self).__init__()
        if padding:
            self.embedding = nn.Embedding(26, embedding_dim, padding_idx=0)
        else:
            self.embedding = nn.Embedding(26, embedding_dim)
        in_ch = [embedding_dim] + num_filters
        self.in_ch = in_ch[-1]

        self.acmix1 = ACmix(in_planes=in_ch[0], out_planes=in_ch[1], head=num_head)
        self.bn1 = nn.BatchNorm1d(in_ch[1])
        self.acmix2 = ACmix(in_planes=in_ch[1], out_planes=in_ch[2], head=num_head)
        self.bn2 = nn.BatchNorm1d(in_ch[2])

        self.acmix3 = ACmix(in_planes=in_ch[2], out_planes=in_ch[3], head=num_head)
        self.bn3 = nn.BatchNorm1d(in_ch[3])

    def forward(self, v):
        v = self.embedding(v.long())
        v = v.transpose(2, 1)#64*128*1200

        v = self.bn1(F.relu(self.acmix1(v.unsqueeze(-2))).squeeze(-2))
        v = self.bn2(F.relu(self.acmix2(v.unsqueeze(-2))).squeeze(-2))

        v = self.bn3(F.relu(self.acmix3(v.unsqueeze(-2))).squeeze(-2))

        v = v.view(v.size(0), v.size(2), -1)
        return v


class MLPDecoder(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, binary=1):
        super(MLPDecoder, self).__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, out_dim)
        self.bn3 = nn.BatchNorm1d(out_dim)
        self.fc4 = nn.Linear(out_dim, binary)

    def forward(self, x):
        x = self.bn1(F.relu(self.fc1(x)))
        x = self.bn2(F.relu(self.fc2(x)))
        x = self.bn3(F.relu(self.fc3(x)))
        x = self.fc4(x)
        return x


# NEW: Alternative fusion approach - Choose one method
class SimpleFusionBINDTI(nn.Module):
    """Simplified version that only fuses at the MLP level"""
    def __init__(self, device='cuda', **config):
        super(SimpleFusionBINDTI, self).__init__()
        # Keep original components
        drug_in_feats = config["DRUG"]["NODE_IN_FEATS"]
        drug_embedding = config["DRUG"]["NODE_IN_EMBEDDING"]
        drug_hidden_feats = config["DRUG"]["HIDDEN_LAYERS"]
        protein_emb_dim = config["PROTEIN"]["EMBEDDING_DIM"]
        num_filters = config["PROTEIN"]["NUM_FILTERS"]
        mlp_in_dim = config["DECODER"]["IN_DIM"]
        mlp_hidden_dim = config["DECODER"]["HIDDEN_DIM"]
        mlp_out_dim = config["DECODER"]["OUT_DIM"]
        protein_num_head = config['PROTEIN']['NUM_HEAD']
        cross_num_head = config['CROSSINTENTION']['NUM_HEAD']
        cross_emb_dim = config['CROSSINTENTION']['EMBEDDING_DIM']
        cross_layer = config['CROSSINTENTION']['LAYER']
        
        # Use original GCN (no fusion at feature level)
        self.drug_extractor = MolecularGCN(
            in_feats=drug_in_feats, 
            dim_embedding=drug_embedding,
            padding=config["DRUG"]["PADDING"],
            hidden_feats=drug_hidden_feats
        )
        
        self.protein_extractor = ProteinACmix(protein_emb_dim, num_filters, protein_num_head, config["PROTEIN"]["PADDING"])
        self.cross_intention = BiIntention(embed_dim=cross_emb_dim, num_head=cross_num_head, layer=cross_layer, device=device)
        
        # Enhanced MLP that can handle optional Mol2Vec features
        self.mlp_in_dim = mlp_in_dim
        self.mlp_classifier = MLPDecoder(mlp_in_dim, mlp_hidden_dim, mlp_out_dim, binary=config["DECODER"]["BINARY"])

    def forward(self, bg_d, v_p, smiles_emb=None, mode="train"):
        v_d = self.drug_extractor(bg_d)
        v_p = self.protein_extractor(v_p)
        f, v_d, v_p, att = self.cross_intention(drug=v_d, protein=v_p)
        
        # Simple fusion: only use cross-intention features
        # Mol2Vec features are handled separately if needed
        score = self.mlp_classifier(f)
        
        if mode == "train":
            return v_d, v_p, f, score
        elif mode == "eval":
            return v_d, v_p, score, att