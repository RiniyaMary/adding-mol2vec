# train_with_mol2vec.py
from mol2vec.features import mol2alt_sentence
from gensim.models import Word2Vec
from rdkit import Chem
import pandas as pd
import os
import numpy as np

def train_with_mol2vec():
    print("🚀 TRAINING WITH MOL2VEC FUNCTIONS")
    print("==================================")
    
    # Load your data
    data_folder = "../datasets/sample/random/"
    train_path = os.path.join(data_folder, 'train_with_seq_cleaned.csv')
    
    if not os.path.exists(train_path):
        print("❌ Training data not found")
        return
    
    df_train = pd.read_csv(train_path)
    print(f"📊 Loaded {len(df_train)} training samples")
    
    # Use mol2vec's professional function to convert molecules
    print("🔬 Converting molecules to Mol2Vec sentences...")
    sentences = []
    
    for i, smiles in enumerate(df_train['SMILES']):
        mol = Chem.MolFromSmiles(str(smiles))
        if mol:
            # This is the official mol2vec conversion!
            sentence = mol2alt_sentence(mol, radius=1)
            sentences.append(sentence)
        
        if (i + 1) % 100 == 0:
            print(f"   Processed {i + 1}/{len(df_train)}")
    
    print(f"✅ Generated {len(sentences)} molecular sentences")
    
    # Train using mol2vec methodology
    print("🏋️ Training Word2Vec model...")
    model = Word2Vec(
        sentences=sentences,
        vector_size=300,
        window=10,
        min_count=1,
        workers=4,
        epochs=30,
        sg=1
    )
    
    # Save the model
    os.makedirs("../models", exist_ok=True)
    model_path = "../models/mol2vec_trained.model"
    model.save(model_path)
    
    print(f"✅ Model saved to: {model_path}")
    print(f"📊 Vocabulary size: {len(model.wv.key_to_index)}")
    print(f"🔢 Embedding dimension: {model.vector_size}")
    
    # Test the model
    print("\n🧪 TESTING THE TRAINED MODEL:")
    test_smiles = "CCO"
    mol = Chem.MolFromSmiles(test_smiles)
    if mol:
        test_sentence = mol2alt_sentence(mol, radius=1)
        print(f"SMILES: {test_smiles}")
        print(f"Sentence: {test_sentence}")
        
        embeddings = []
        for token in test_sentence:
            if token in model.wv.key_to_index:
                embedding = model.wv[token]
                embeddings.append(embedding)
                print(f"   ✅ '{token}': embedded")
        
        if embeddings:
            avg_embedding = np.mean(embeddings, axis=0)
            print(f"📊 Molecule embedding shape: {avg_embedding.shape}")
            print("🎉 MODEL IS READY FOR BINDTI!")
    
    return model_path

if __name__ == "__main__":
    train_with_mol2vec()