from models import BINDTI
from time import time
from utils import set_seed, mkdir, graph_collate_func_with_smiles  # UPDATED: New collate function
from configs import get_cfg_defaults
from dataloader import DTIDataset
from torch.utils.data import DataLoader
from trainer import Trainer
import torch
import argparse
import warnings, os
import pandas as pd
from datetime import datetime

cuda_id = 0
device = torch.device(f'cuda:{cuda_id}' if torch.cuda.is_available() else 'cpu')
#device = 'cpu'

# ========== CORRECTED PARSER SECTION ==========
parser = argparse.ArgumentParser(description="BINDTI for DTI prediction")
parser.add_argument('--data', type=str, metavar='TASK', help='dataset', default='sample')
parser.add_argument('--split', default='random', type=str, metavar='S', help="split task", choices=['random', 'random1', 'random2', 'random3', 'random4'])

# NEW: Add resume and checkpoint arguments
parser.add_argument('--resume', action='store_true', help='resume from checkpoint')
parser.add_argument('--checkpoint', type=str, default=None, help='path to checkpoint file')

# NEW: Add Mol2Vec toggle
parser.add_argument('--no-mol2vec', action='store_true', help='disable Mol2Vec features')

args = parser.parse_args()
# ========== END CORRECTED SECTION ==========


def main():
    torch.cuda.empty_cache()
    warnings.filterwarnings("ignore", message="invalid value encountered in divide")
    cfg = get_cfg_defaults()
    set_seed(cfg.SOLVER.SEED)
    mkdir(cfg.RESULT.OUTPUT_DIR + f'{args.data}/{args.split}')

    # NEW: Override Mol2Vec setting if disabled via command line
    if args.no_mol2vec:
        cfg.DRUG.USE_MOL2VEC = False
        print("⚠️  Mol2Vec features disabled via command line")

    print("start...")
    print(f"dataset:{args.data}")
    print(f"Hyperparameters: {dict(cfg)}")
    print(f"Running on: {device}", end="\n\n")
    
    # NEW: Print Mol2Vec status
    if cfg.DRUG.USE_MOL2VEC:
        print("🧪 Mol2Vec features: ENABLED")
        print(f"   Model path: {cfg.DRUG.MOL2VEC_MODEL_PATH}")
        print(f"   Embedding dim: {cfg.DRUG.MOL2VEC_EMBEDDING_DIM}")
    else:
        print("🧪 Mol2Vec features: DISABLED")

    dataFolder = f'../datasets/{args.data}'
    dataFolder = os.path.join(dataFolder, str(args.split))

    train_path = os.path.join(dataFolder, 'train_with_seq_cleaned.csv')
    val_path = os.path.join(dataFolder, "val_with_seq_cleaned.csv")
    test_path = os.path.join(dataFolder, "test_with_seq_cleaned.csv")
    df_train = pd.read_csv(train_path)
    df_val = pd.read_csv(val_path)
    df_test = pd.read_csv(test_path)

    # UPDATED: Dataset creation with Mol2Vec parameters
    mol2vec_model_path = cfg.DRUG.MOL2VEC_MODEL_PATH if cfg.DRUG.USE_MOL2VEC else None
    max_smiles_length = cfg.DRUG.MAX_SMILES_LENGTH

    train_dataset = DTIDataset(
        df_train.index.values, 
        df_train, 
        max_drug_nodes=cfg.DRUG.MAX_NODES,
        mol2vec_model_path=mol2vec_model_path,
        max_smiles_length=max_smiles_length
    )
    print(f'train_dataset:{len(train_dataset)}')
    
    val_dataset = DTIDataset(
        df_val.index.values, 
        df_val,
        max_drug_nodes=cfg.DRUG.MAX_NODES,
        mol2vec_model_path=mol2vec_model_path,
        max_smiles_length=max_smiles_length
    )
    
    test_dataset = DTIDataset(
        df_test.index.values, 
        df_test,
        max_drug_nodes=cfg.DRUG.MAX_NODES,
        mol2vec_model_path=mol2vec_model_path,
        max_smiles_length=max_smiles_length
    )

    # UPDATED: Use new collate function that handles SMILES embeddings
    params = {
        'batch_size': cfg.SOLVER.BATCH_SIZE, 
        'shuffle': True, 
        'num_workers': cfg.SOLVER.NUM_WORKERS,
        'drop_last': True, 
        'collate_fn': graph_collate_func_with_smiles  # UPDATED: New collate function
    }

    training_generator = DataLoader(train_dataset, **params)
    params['shuffle'] = False
    params['drop_last'] = False
    val_generator = DataLoader(val_dataset, **params)
    test_generator = DataLoader(test_dataset, **params)

    # NEW: Print data loader info
    print(f"\n📊 Data Loader Info:")
    print(f"   Batch size: {cfg.SOLVER.BATCH_SIZE}")
    print(f"   Using collate function: graph_collate_func_with_smiles")
    
    # Test one batch to verify Mol2Vec integration
    if cfg.DRUG.USE_MOL2VEC:
        print("   Testing Mol2Vec batch...")
        sample_batch = next(iter(training_generator))
        drug_graph, protein_seq, smiles_emb, labels = sample_batch
        print(f"   Batch shapes - Drug: {drug_graph.batch_size} graphs, Protein: {protein_seq.shape}, SMILES: {smiles_emb.shape}, Labels: {labels.shape}")

    model = BINDTI(device=device, **cfg).to(device=device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.SOLVER.LR, weight_decay=cfg.SOLVER.WEIGHT_DECAY)
    torch.backends.cudnn.benchmark = True

    # NEW: Print model info
    print(f"\n🤖 Model Info:")
    print(f"   Using enhanced BINDTI with Mol2Vec: {cfg.DRUG.USE_MOL2VEC}")
    if cfg.DRUG.USE_MOL2VEC:
        print(f"   MLP input dimension: {cfg.DECODER.IN_DIM + cfg.DRUG.MOL2VEC_EMBEDDING_DIM}")

    trainer = Trainer(model, opt, device, training_generator, val_generator, test_generator, args.data, args.split, **cfg)
    result = trainer.train(resume=args.resume, checkpoint_path=args.checkpoint)

    with open(os.path.join(cfg.RESULT.OUTPUT_DIR, f"{args.data}/{args.split}/model_architecture.txt"), "w") as wf:
        wf.write(str(model))
    with open(os.path.join(cfg.RESULT.OUTPUT_DIR, f"{args.data}/{args.split}/config.txt"), "w") as wf:
        wf.write(str(dict(cfg)))

    print(f"\nDirectory for saving result: {cfg.RESULT.OUTPUT_DIR}{args.data}")
    print(f'\nend...')

    return result


if __name__ == '__main__':
    print(f"start time: {datetime.now()}")
    s = time()
    result = main()
    e = time()
    print(f"end time: {datetime.now()}")
    print(f"Total running time: {round(e - s, 2)}s")
    
    # NEW: Final summary
    if 'auroc' in result:
        print(f"\n🎯 Final Results:")
        print(f"   AUROC: {result['auroc']:.4f}")
        print(f"   AUPRC: {result['auprc']:.4f}")
        print(f"   Best Epoch: {result['best_epoch']}")