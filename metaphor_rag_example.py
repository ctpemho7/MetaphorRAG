"""
Psychological Support RAG System
------------------------------
A bilingual (English-Russian) RAG system for psychological support,
featuring hybrid search, hypothetical document embeddings, and efficient models.

Key components:
- Translation (using M2M100)
- Embedding (using BGE-small)
- LLM (using Phi-2)
- Hybrid search (dense + sparse retrieval)
- Cross-encoder reranking
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from optimum.bettertransformer import BetterTransformer
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.retrievers import BM25Retriever
from sentence_transformers import CrossEncoder
from typing import List, Dict, Tuple
from functools import lru_cache
import langdetect
import numpy as np


class BilingualTranslator:
    """
    Handles efficient translation between English and Russian using M2M100.
    Features:
    - Caching for repeated translations
    - Automatic language detection
    - Optimized for GPU when available
    """
    
    def __init__(self, device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        # Initialize M2M100 model (smaller than NLLB while maintaining quality)
        self.model_name = "facebook/m2m100_418M"
        
        model_kwargs = {
            "device_map": "auto",
            "torch_dtype": torch.float16 if device == "cuda" else torch.float32,
        }
        
        self.translation_pipeline = pipeline(
            "translation",
            model=self.model_name,
            **model_kwargs
        )

    @lru_cache(maxsize=1000)
    def to_english(self, text: str) -> str:
        """Translate text to English if it's in Russian."""
        if not text.strip():
            return text
            
        try:
            source_lang = langdetect.detect(text)
            if source_lang != 'ru':
                return text
        except:
            return text  # Return original if language detection fails
            
        return self.translation_pipeline(
            text,
            src_lang="ru",
            tgt_lang="en",
            max_length=512
        )[0]['translation_text']

    @lru_cache(maxsize=1000)
    def to_russian(self, text: str) -> str:
        """Translate English text to Russian."""
        return self.translation_pipeline(
            text,
            src_lang="en",
            tgt_lang="ru",
            max_length=512
        )[0]['translation_text']


class EfficientModelLoader:
    """
    Handles loading and optimization of ML models.
    Features:
    - 4-bit quantization
    - Better Transformer optimization
    - Automatic device placement
    """
    
    @staticmethod
    def load_llm(
        model_name: str = "microsoft/phi-2",  # Efficient 2.7B parameter model
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        load_in_4bit: bool = True,
    ):
        """Load and optimize the main language model."""
        model_kwargs = {
            "device_map": "auto",
            "torch_dtype": torch.bfloat16,
        }
        
        # Add quantization configuration if requested
        if load_in_4bit:
            from transformers import BitsAndBytesConfig
            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
            )
        
        # Load and optimize model
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            **model_kwargs
        )
        # model = BetterTransformer.transform(model)
        # ValueError: This model already uses BetterTransformer optimizations from Transformers (torch.nn.functional.scaled_dot_product_attention). As such, there is no need to use `model.to_bettertransformers()` or `BetterTransformer.transform(model)` from the Optimum library. Details: https://huggingface.co/docs/transformers/perf_infer_gpu_one#pytorch-scaled-dot-product-attention.
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        return pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=512,
            temperature=0.7,
            top_p=0.95,
            batch_size=1
        )


class ModernSearchEngine:
    """
    Implements advanced search functionality.
    Features:
    - Hybrid search (dense + sparse)
    - Hypothetical document embeddings
    - Cross-encoder reranking
    """
    
    def __init__(self, embeddings, device="cuda" if torch.cuda.is_available() else "cpu"):
        self.embeddings = embeddings
        self.device = device
        # Using TinyBERT for efficient reranking
        self.cross_encoder = CrossEncoder('cross-encoder/ms-marco-TinyBERT-L-2', device=device)
        
    def create_vectors_and_index(self, documents: List, category: str) -> Tuple[FAISS, BM25Retriever]:
        """Create both vector store and BM25 index for hybrid search."""
        vector_store = FAISS.from_documents(documents, self.embeddings)
        bm25_retriever = BM25Retriever.from_documents(documents)
        return vector_store, bm25_retriever

    def generate_hypothetical_docs(self, llm, query: str, n_docs: int = 3) -> List[str]:
        """
        Generate hypothetical perfect documents for improved retrieval.
        This implements the Hypothetical Document Embeddings (HyDE) technique.
        """
        prompt = f"""Given this psychological support request, write {n_docs} short but detailed 
        passages that would be highly relevant and helpful. Each passage should be different in focus:
        1. Professional therapeutic perspective
        2. Similar experience sharing
        3. Empathetic support response
        
        Request: {query}
        
        Write each passage separated by [DOC]."""
        
        result = llm(prompt)[0]['generated_text']
        return result.split('[DOC]')

    def hybrid_search(self, 
                     query: str,
                     vector_store: FAISS,
                     bm25_retriever: BM25Retriever,
                     top_k: int = 5,
                     vector_weight: float = 0.7) -> List:
        """
        Perform hybrid search combining dense and sparse retrieval.
        
        Args:
            query: Search query
            vector_store: FAISS vector store
            bm25_retriever: BM25 retriever
            top_k: Number of results to return
            vector_weight: Weight for vector search scores (0-1)
        """
        # Get results from both retrievers
        vector_results = vector_store.similarity_search_with_score(query, k=top_k*2)
        bm25_results = bm25_retriever.get_relevant_documents(query)[:top_k*2]
        
        # Normalize vector scores
        vector_scores = [score for _, score in vector_results]
        min_score, max_score = min(vector_scores), max(vector_scores)
        normalized_vector_scores = [
            1 - (score - min_score) / (max_score - min_score) 
            for score in vector_scores
        ]
        
        # Combine results with weighted scores
        all_docs = {}
        for doc, score in zip([doc for doc, _ in vector_results], normalized_vector_scores):
            all_docs[doc.page_content] = score * vector_weight
            
        for doc in bm25_results:
            if doc.page_content in all_docs:
                all_docs[doc.page_content] += (1 - vector_weight)
            else:
                all_docs[doc.page_content] = (1 - vector_weight)
        
        return [(content, score) for content, score in all_docs.items()]

    def rerank(self, query: str, results: List[Tuple[str, float]], top_k: int = 5) -> List:
        """
        Rerank results using cross-encoder.
        Combines initial retrieval scores with cross-encoder scores.
        """
        if not results:
            return []
            
        # Prepare pairs for cross-encoder
        pairs = [[query, doc] for doc, _ in results]
        
        # Get cross-encoder scores
        cross_scores = self.cross_encoder.predict(pairs)
        
        # Combine with hybrid scores (70% cross-encoder, 30% hybrid)
        hybrid_scores = [score for _, score in results]
        final_scores = [
            0.7 * cross_score + 0.3 * hybrid_score
            for cross_score, hybrid_score in zip(cross_scores, hybrid_scores)
        ]
        
        # Sort and return top_k
        scored_results = list(zip([doc for doc, _ in results], final_scores))
        scored_results.sort(key=lambda x: x[1], reverse=True)
        
        return [doc for doc, _ in scored_results[:top_k]]


class PsychologicalRAG:
    """
    Main RAG system for psychological support.
    Features:
    - Bilingual support (EN/RU)
    - Advanced retrieval (hybrid search + HyDE)
    - Efficient models and optimizations
    """
    
    def __init__(self, 
                 device: str = "cuda" if torch.cuda.is_available() else "cpu",
                 load_in_4bit: bool = True):
        
        self.device = device
        # Initialize components
        self.translator = BilingualTranslator(device=device)
        self.embeddings = HuggingFaceEmbeddings(
            model_name="BAAI/bge-small-en-v1.5",
            model_kwargs={'device': device}
        )
        self.search_engine = ModernSearchEngine(self.embeddings, device)
        
        # Initialize stores for different document categories
        self.stores = {category: {'vector': None, 'bm25': None} 
                      for category in ['grounded', 'diverse', 'empathy']}
        
        # Set up text splitting
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
            separators=["\n\n", "\n", ".", "!", "?", ",", " "]
        )
        
        # Initialize language model
        self.llm = EfficientModelLoader.load_llm(
            load_in_4bit=load_in_4bit,
            device=device
        )
        
        # Set up response template
        self.template = """Task: Generate a helpful response for this psychological support request.

                            Context:
                            Request: {request}

                            Professional Insights:
                            {grounded_docs}

                            Community Experiences:
                            {diverse_docs}

                            Empathetic Responses:
                            {empathy_docs}

                            Requirements:
                            1. Show understanding
                            2. Provide professional guidance
                            3. Share relevant experiences
                            4. Maintain empathy

                            Response:"""

    def process_chunks(self, text: str, metadata: dict) -> List:
        """Process and chunk text with translation if needed."""
        english_text = self.translator.to_english(text)
        
        try:
            original_lang = langdetect.detect(text)
        except:
            original_lang = 'en'
        
        return self.text_splitter.create_documents(
            texts=[english_text],
            metadatas=[{**metadata, 'original_lang': original_lang}]
        )

    def index_documents(self, documents: Dict[str, List[Dict]]):
        """Index documents with both vector and sparse indices."""
        for category, docs in documents.items():
            processed_docs = []
            for doc in docs:
                chunks = self.process_chunks(doc['text'], doc['metadata'])
                processed_docs.extend(chunks)
            
            # Create both vector and BM25 indices
            vector_store, bm25_retriever = self.search_engine.create_vectors_and_index(
                processed_docs, category
            )
            
            self.stores[category]['vector'] = vector_store
            self.stores[category]['bm25'] = bm25_retriever
            
            print(f"Indexed {len(processed_docs)} chunks for {category}")

    def retrieve(self, request: str, top_k: int = 5) -> Dict[str, List]:
        """
        Enhanced retrieval combining:
        - Hybrid search
        - Hypothetical document embeddings
        - Cross-encoder reranking
        """
        try:
            request_lang = langdetect.detect(request)
        except:
            request_lang = 'en'
        
        english_request = self.translator.to_english(request)
        
        # Generate hypothetical documents
        hyde_docs = self.search_engine.generate_hypothetical_docs(self.llm, english_request)
        
        results = {}
        for category in self.stores:
            # Perform hybrid search for both original query and hypothetical docs
            all_results = []
            
            # Search with original query
            hybrid_results = self.search_engine.hybrid_search(
                english_request,
                self.stores[category]['vector'],
                self.stores[category]['bm25'],
                top_k=top_k
            )
            all_results.extend(hybrid_results)
            
            # Search with hypothetical documents
            for hyde_doc in hyde_docs:
                hyde_results = self.search_engine.hybrid_search(
                    hyde_doc,
                    self.stores[category]['vector'],
                    self.stores[category]['bm25'],
                    top_k=top_k//2
                )
                all_results.extend(hyde_results)
            
            # Rerank combined results
            reranked_results = self.search_engine.rerank(
                english_request,
                all_results,
                top_k=top_k
            )
            
            # Translate if needed
            if request_lang == 'ru':
                reranked_results = [
                    self.translator.to_russian(doc) 
                    for doc in reranked_results
                ]
            
            results[category] = reranked_results
        
        return results

    def generate_response(self, request: str, retrieved_docs: Dict[str, List]) -> str:
        """Generate final response with translation if needed."""
        try:
            request_lang = langdetect.detect(request)
        except:
            request_lang = 'en'
        
        english_request = self.translator.to_english(request)
        
        response = self.llm(
            self.template.format(
                request=english_request,
                grounded_docs="\n".join(retrieved_docs['grounded']),
                diverse_docs="\n".join(retrieved_docs['diverse']),
                empathy_docs="\n".join(retrieved_docs['empathy'])
            )
        )[0]['generated_text']
        
        if request_lang == 'ru':
            response = self.translator.to_russian(response)
        
        return response

    def process_request(self, request: str) -> str:
        """Main processing pipeline."""
        retrieved_docs = self.retrieve(request)
        return self.generate_response(request, retrieved_docs)
    

def main():
    # Example usage
    rag = PsychologicalRAG(
        device="cuda" if torch.cuda.is_available() else "cpu",
        load_in_4bit=True
    )
    
    documents = {
        'grounded': [
            {
                'text': 'Professional article about anxiety management...',
                'metadata': {'source': 'journal', 'type': 'article'}
            },
            {
                'text': 'Техники управления тревожностью...',
                'metadata': {'source': 'journal', 'type': 'article'}
            }
        ],
        'diverse': [
            {
                'text': 'Reddit post about coping with work stress...',
                'metadata': {'source': 'reddit', 'type': 'post'}
            },
            {
                'text': 'Пост на Reddit о том, как справляться со стрессом...',
                'metadata': {'source': 'reddit', 'type': 'post'}
            }
        ],
        'empathy': [
            {
                'text': 'Supportive conversation about dealing with anxiety...',
                'metadata': {'source': 'dialogue', 'type': 'conversation'}
            },
            {
                'text': 'Диалог поддержки о том, как справляться с тревогой...',
                'metadata': {'source': 'dialogue', 'type': 'conversation'}
            }
        ]
    }
    
    # Index documents
    print("Indexing documents...")
    rag.index_documents(documents)
    
    # Process requests in both languages
    english_request = "I've been feeling anxious about work lately..."
    russian_request = "В последнее время я испытываю тревогу по поводу работы..."
    
    print("\nProcessing English request:")
    english_response = rag.process_request(english_request)
    print(f"Response: {english_response}")
    
    print("\nProcessing Russian request:")
    russian_response = rag.process_request(russian_request)
    print(f"Response: {russian_response}")


if __name__ == "__main__":
    main()