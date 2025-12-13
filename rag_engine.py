
import json
import networkx as nx
import numpy as np
import os
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from duckduckgo_search import DDGS

try:
    import chromadb
    from chromadb.config import Settings
    CHROMA_AVAILABLE = True
except ImportError:
    CHROMA_AVAILABLE = False
    print("⚠ ChromaDB not found. Falling back to in-memory vector search.")

# Config
DATA_PATH = "data/mental_health_knowledge.json"
MODEL_NAME = 'all-MiniLM-L6-v2'
CHROMA_HOST = os.getenv("CHROMA_HOST", "localhost")
CHROMA_PORT = int(os.getenv("CHROMA_PORT", 8000))

class RAGEngine:
    def __init__(self):
        print("Initializing RAG Engine (ChromaDB + Graph + Web)...")
        self.encoder = SentenceTransformer(MODEL_NAME)
        
        # Initialize ChromaDB Client
        try:
            if CHROMA_AVAILABLE:
                if os.getenv("CHROMA_HOST"):
                    self.chroma_client = chromadb.HttpClient(host=CHROMA_HOST, port=CHROMA_PORT)
                else:
                    self.chroma_client = chromadb.PersistentClient(path="./chroma_db") # Persistent local storage
                
                self.collection = self.chroma_client.get_or_create_collection(name="mental_health_docs")
                self.use_chroma = True
                print(f"✓ Connected to ChromaDB (Persistent)")
            else:
                self.use_chroma = False
        except Exception as e:
            print(f"⚠ ChromaDB connection failed ({e}), falling back to in-memory vector search.")
            self.use_chroma = False

        self.load_knowledge_base()
        
    def load_knowledge_base(self):
        with open(DATA_PATH, 'r') as f:
            data = json.load(f)
            
        self.documents = data['documents']
        self.nodes = data['nodes']
        self.edges = data['edges']
        
        # Populate Vector DB if using Chroma
        if self.use_chroma:
            try:
                if self.collection.count() == 0:
                    print("Populating ChromaDB...")
                    ids = [d['id'] for d in self.documents]
                    docs = [d['content'] for d in self.documents]
                    metadatas = [{'title': d['title']} for d in self.documents]
                    embeddings = self.encoder.encode(docs).tolist()
                    
                    self.collection.add(
                        embeddings=embeddings,
                        documents=docs,
                        metadatas=metadatas,
                        ids=ids
                    )
            except Exception as e:
                print(f"Error populating Chroma: {e}")
        else:
            # Fallback
            self.doc_texts = [d['content'] for d in self.documents]
            self.doc_embeddings = self.encoder.encode(self.doc_texts)
        
        # Build Knowledge Graph
        self.graph = nx.DiGraph()
        for node in self.nodes:
            self.graph.add_node(node['id'], **node)
        for edge in self.edges:
            self.graph.add_edge(edge['source'], edge['target'], relation=edge['relation'])
            
    def vector_search(self, query, top_k=2):
        if self.use_chroma:
            try:
                query_vec = self.encoder.encode([query]).tolist()
                results = self.collection.query(
                    query_embeddings=query_vec,
                    n_results=top_k
                )
                
                output = []
                # Chroma returns list of lists
                if results['documents']:
                    for i, doc in enumerate(results['documents'][0]):
                        meta = results['metadatas'][0][i]
                        output.append({
                            'source': 'Local Knowledge (Vector)',
                            'title': meta['title'],
                            'content': doc,
                            'score': 0.9 # Chroma doesn't always return normalized score easily
                        })
                return output
            except Exception as e:
                print(f"Chroma search error: {e}")
                return []
        else:
            # Fallback implementation
            query_vec = self.encoder.encode([query])
            sims = cosine_similarity(query_vec, self.doc_embeddings)[0]
            top_indices = np.argsort(sims)[-top_k:][::-1]
            results = []
            for idx in top_indices:
                if sims[idx] > 0.3:
                    results.append({
                        'source': 'Local Knowledge',
                        'title': self.documents[idx]['title'],
                        'content': self.documents[idx]['content'],
                        'score': float(sims[idx])
                    })
            return results

    def graph_search(self, query):
        """Find related concepts in graph"""
        related_nodes = []
        query_lower = query.lower()
        start_nodes = [n for n in self.graph.nodes if n in query_lower or query_lower in n]
        
        for start_node in start_nodes:
            neighbors = list(self.graph.neighbors(start_node))
            related_nodes.extend(neighbors)
            related_nodes.append(start_node)
            
        results = []
        unique_nodes = set(related_nodes)
        for node_id in unique_nodes:
            node_data = self.graph.nodes[node_id]
            desc = node_data.get('description', '')
            results.append({
                'source': 'Knowledge Graph',
                'title': f"Concept: {node_id.title()}",
                'content': f"{node_id.title()} ({node_data.get('type')}): {desc}",
                'score': 1.0 
            })
        return results

    def web_search(self, query, max_results=2):
        print(f"Searching web for: {query}")
        results = []
        try:
            with DDGS() as ddgs:
                context_query = f"{query} mental health research"
                search_results = list(ddgs.text(context_query, max_results=max_results))
                for r in search_results:
                    results.append({
                        'source': 'Web Search (DuckDuckGo)',
                        'title': r['title'],
                        'content': r['body'],
                        'url': r['href'],
                        'score': 0.8
                    })
        except Exception as e:
            print(f"Web search failed: {e}")
            
        return results

        return results

    def save_memory(self, text, metadata=None):
        if self.use_chroma:
            try:
                import uuid
                doc_id = str(uuid.uuid4())
                embedding = self.encoder.encode([text]).tolist()
                
                self.collection.add(
                    embeddings=embedding,
                    documents=[text],
                    metadatas=[metadata or {"source": "user_memory"}],
                    ids=[doc_id]
                )
                print(f"saved memory {doc_id}")
            except Exception as e:
                print(f"Error saving memory: {e}")

    def query(self, user_query):
        vector_results = self.vector_search(user_query)
        graph_results = self.graph_search(user_query)
        web_results = self.web_search(user_query)
        
        all_results = vector_results + graph_results + web_results
        
        seen = set()
        final_results = []
        for r in all_results:
            if r['content'] not in seen:
                seen.add(r['content'])
                final_results.append(r)
                
        return final_results
