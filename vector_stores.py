import os
import json
import faiss
import numpy as np
from typing import List, Dict, Union, Set
from sentence_transformers import SentenceTransformer
from dataclasses import dataclass
from tqdm import tqdm

# Import constants from data.py
from data import (
    DATA_DIR,
    FINAL_PATHS
)

# Additional constants for vector stores
VECTOR_STORES_DIR = os.path.join(DATA_DIR, 'vector_stores')
VECTOR_STORE_PATHS = {
    'problem_solutions': os.path.join(VECTOR_STORES_DIR, 'problem_solutions.faiss'),
    'karpavichus_solutions': os.path.join(VECTOR_STORES_DIR, 'karpavichus_solutions.faiss'),
    'cbt_solutions': os.path.join(VECTOR_STORES_DIR, 'cbt_solutions.faiss')
}

os.makedirs(VECTOR_STORES_DIR, exist_ok=True)

@dataclass
class BaseVectorStore:
    """Base class for all vector stores"""

    def __init__(self, load_from_disk: bool = True):
        self.encoder = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        self.dimension = 384  # dimension of the model's embeddings

        # Load data
        with open(self.data_path, 'r') as f:
            self.data = json.load(f)

        self.problems = list(self.data.keys())

        if load_from_disk and os.path.exists(self.store_path):
            self.index = faiss.read_index(self.store_path)
        else:
            self._build_index()

    def _build_index(self):
        """Build FAISS index from problems"""
        self.index = faiss.IndexFlatL2(self.dimension)

        embeddings = self.encoder.encode(self.problems, show_progress_bar=True)
        self.index.add(np.array(embeddings).astype('float32'))

        # Save index
        faiss.write_index(self.index, self.store_path)

    def search(self, query_problems: List[str], top_k: int = 5) -> List[Dict]:
        """
        Search for similar problems across all query problems and return top_k unique matches
        """
        # Get embeddings for all query problems
        query_embeddings = self.encoder.encode(query_problems)

        all_scores = []
        all_indices = []

        # Search for each query problem
        for emb in query_embeddings:
            D, I = self.index.search(np.array([emb]).astype('float32'), top_k * 2)  # Get more matches initially
            all_scores.extend(D[0])
            all_indices.extend(I[0])

        # Create (score, index) pairs and sort
        pairs = sorted(zip(all_scores, all_indices))

        # Get unique matches preserving order
        seen_meta = set()
        results = []

        for score, idx in pairs:
            problem = self.problems[idx]
            meta = self.data[problem]['meta']

            if meta not in seen_meta and len(results) < top_k:
                seen_meta.add(meta)
                results.append({
                    'problem': problem,
                    **self.data[problem],
                    'score': float(score)
                })

        return results

class ProblemSolutionsStore(BaseVectorStore):
    """Vector store for general problem-solution pairs"""

    def __init__(self, load_from_disk: bool = True):
        self.data_path = FINAL_PATHS['problem_solutions']
        self.store_path = VECTOR_STORE_PATHS['problem_solutions']
        super().__init__(load_from_disk)

class KarapavichusSolutionsStore(BaseVectorStore):
    """Vector store for Karpavichus-derived solutions"""

    def __init__(self, load_from_disk: bool = True):
        self.data_path = FINAL_PATHS['karpavichus_solutions']
        self.store_path = VECTOR_STORE_PATHS['karpavichus_solutions']
        super().__init__(load_from_disk)

class CBTSolutionsStore(BaseVectorStore):
    """Vector store for CBT solutions"""

    def __init__(self, load_from_disk: bool = True):
        self.data_path = FINAL_PATHS['cbt_solutions']
        self.store_path = VECTOR_STORE_PATHS['cbt_solutions']
        super().__init__(load_from_disk)
