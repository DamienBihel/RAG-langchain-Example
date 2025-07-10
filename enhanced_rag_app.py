"""
Application RAG Améliorée avec Support Multi-Modes

Cette version étend l'application RAG originale pour supporter
différentes stratégies RAG de manière modulaire.

Fonctionnalités ajoutées :
- Support de multiples stratégies RAG
- Changement dynamique de stratégie
- Comparaison entre stratégies
- Métriques détaillées
- Interface unifiée
"""

import os
import getpass
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from pathlib import Path
import json
import time

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain_core.documents import Document
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

# Import des stratégies RAG
from rag_strategies import (
    RAGStrategy, RAGResult, RAGStrategyFactory,
    NaiveRAG, HybridSearchRAG, CorrectiveRAG, SelfReflectiveRAG
)


@dataclass
class RAGComparison:
    """Résultat de comparaison entre différentes stratégies RAG."""
    query: str
    results: Dict[str, RAGResult]
    best_strategy: str
    comparison_metrics: Dict[str, Any]


class EnhancedRAGApplication:
    """
    Application RAG améliorée avec support multi-stratégies.
    
    Cette classe étend les fonctionnalités de base avec :
    - Support de multiples stratégies RAG
    - Changement dynamique de stratégie
    - Comparaison de performances
    - Métriques détaillées
    """
    
    def __init__(self, default_strategy: str = 'naive'):
        """
        Initialise l'application RAG améliorée.
        
        Args:
            default_strategy: Stratégie par défaut à utiliser
        """
        # Composants de base
        self.llm = None
        self.embeddings = None
        self.vector_store = None
        self.client = None
        self.collection_name = "enhanced_documents"
        
        # Gestion des stratégies
        self.current_strategy = None
        self.available_strategies = {}
        self.default_strategy_name = default_strategy
        
        # Métriques et historique
        self.query_history = []
        self.performance_metrics = {}
        
        # Configuration
        self.sources = set()
        self.documents_count = 0
    
    def setup_environment(self):
        """Configure l'environnement et les clés API."""
        print("🔧 Configuration de l'environnement...")
        
        load_dotenv()
        
        if not os.environ.get("OPENAI_API_KEY"):
            os.environ["OPENAI_API_KEY"] = getpass.getpass("Entrez votre clé API OpenAI: ")
        
        print("✅ Environnement configuré")
    
    def initialize_models(self):
        """Initialise les modèles LLM et embeddings."""
        print("🔧 Initialisation des modèles...")
        
        # Modèle de chat
        self.llm = init_chat_model(
            "gpt-4o-mini",
            model_provider="openai",
            temperature=0.1
        )
        
        # Modèle d'embeddings
        self.embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        
        print("✅ Modèles initialisés")
    
    def setup_vector_store(self):
        """Configure la base de données vectorielle."""
        print("🗄️ Configuration de la base vectorielle...")
        
        # Client Qdrant
        self.client = QdrantClient(":memory:")
        
        # Création de la collection
        try:
            self.client.get_collection(self.collection_name)
            print(f"Collection '{self.collection_name}' trouvée")
        except ValueError:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=1536,
                    distance=Distance.COSINE
                )
            )
            print(f"Collection '{self.collection_name}' créée")
        
        # Interface vector store
        self.vector_store = QdrantVectorStore(
            client=self.client,
            collection_name=self.collection_name,
            embedding=self.embeddings,
        )
        
        print("✅ Base vectorielle configurée")
    
    def initialize_strategies(self):
        """Initialise toutes les stratégies RAG disponibles."""
        print("🎯 Initialisation des stratégies RAG...")
        
        # Configuration des stratégies
        strategy_configs = {
            'naive': {},
            'hybrid': {'vector_weight': 0.7, 'bm25_weight': 0.3},
            'corrective': {'relevance_threshold': 0.5, 'max_corrections': 2},
            'reflective': {'max_iterations': 2, 'quality_threshold': 0.8}
        }
        
        # Création des stratégies
        for strategy_name, config in strategy_configs.items():
            try:
                strategy = RAGStrategyFactory.create_strategy(
                    strategy_name,
                    llm=self.llm,
                    embeddings=self.embeddings,
                    vector_store=self.vector_store,
                    **config
                )
                
                self.available_strategies[strategy_name] = strategy
                print(f"✅ Stratégie '{strategy_name}' initialisée")
                
            except Exception as e:
                print(f"⚠️  Erreur lors de l'initialisation de '{strategy_name}': {e}")
        
        # Définition de la stratégie par défaut
        self.set_strategy(self.default_strategy_name)
        
        print(f"🎯 {len(self.available_strategies)} stratégies disponibles")
    
    def set_strategy(self, strategy_name: str):
        """
        Change la stratégie RAG active.
        
        Args:
            strategy_name: Nom de la stratégie à utiliser
        """
        if strategy_name not in self.available_strategies:
            available = list(self.available_strategies.keys())
            raise ValueError(f"Stratégie '{strategy_name}' non disponible. Disponibles: {available}")
        
        self.current_strategy = self.available_strategies[strategy_name]
        print(f"🔄 Stratégie active: {strategy_name}")
        
        # Configuration spécifique si nécessaire
        if strategy_name == 'hybrid' and hasattr(self.current_strategy, 'setup_bm25_retriever'):
            self._setup_hybrid_strategy()
        elif strategy_name == 'corrective' and hasattr(self.current_strategy, 'setup_evaluator'):
            self.current_strategy.setup_evaluator()
    
    def _setup_hybrid_strategy(self):
        """Configure la stratégie hybride avec BM25."""
        if self.documents_count > 0:
            print("🔧 Configuration de la stratégie hybride...")
            # Récupération des documents pour BM25
            try:
                docs = self.get_all_documents(limit=1000)
                documents = []
                for doc_info in docs:
                    doc = Document(
                        page_content=doc_info['content'],
                        metadata=doc_info['metadata']
                    )
                    documents.append(doc)
                
                if documents:
                    self.current_strategy.setup_bm25_retriever(documents)
            except Exception as e:
                print(f"⚠️  Erreur configuration BM25: {e}")
    
    def get_current_strategy(self) -> str:
        """Retourne le nom de la stratégie actuellement active."""
        return self.current_strategy.name if self.current_strategy else "Aucune"
    
    def list_strategies(self) -> List[str]:
        """Retourne la liste des stratégies disponibles."""
        return list(self.available_strategies.keys())
    
    def load_and_process_documents(self, source: str, file_type: str = "auto") -> List[Document]:
        """
        Charge et traite les documents (même logique que l'original).
        
        Args:
            source: Source du document
            file_type: Type de fichier
            
        Returns:
            Liste des documents traités
        """
        print(f"📄 Chargement du document depuis {source}...")
        
        # Détection du type
        if file_type == "auto":
            if source.startswith("http"):
                file_type = "web"
            elif source.endswith(".pdf"):
                file_type = "pdf"
            elif source.endswith(".md"):
                file_type = "markdown"
            else:
                file_type = "text"
        
        # Chargement selon le type
        if file_type == "web":
            import bs4
            from langchain_community.document_loaders import WebBaseLoader
            
            bs4_strainer = bs4.SoupStrainer(class_=("post-title", "post-header", "post-content"))
            loader = WebBaseLoader(
                web_paths=(source,),
                bs_kwargs={"parse_only": bs4_strainer},
            )
        elif file_type == "pdf":
            from langchain_community.document_loaders import PyPDFLoader
            loader = PyPDFLoader(source)
        elif file_type == "markdown":
            from langchain_community.document_loaders import UnstructuredMarkdownLoader
            loader = UnstructuredMarkdownLoader(source)
        else:
            from langchain_community.document_loaders import TextLoader
            loader = TextLoader(source)
        
        # Chargement et traitement
        docs = loader.load()
        print(f"Document chargé: {len(docs[0].page_content)} caractères")
        
        # Découpage
        from langchain_text_splitters import RecursiveCharacterTextSplitter
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            add_start_index=True,
        )
        
        splits = text_splitter.split_documents(docs)
        
        # Enrichissement des métadonnées
        for i, split in enumerate(splits):
            split.metadata["source"] = source
            split.metadata["file_type"] = file_type
            split.metadata["chunk_index"] = i
            split.metadata["total_chunks"] = len(splits)
        
        # Ajout à l'ensemble des sources
        self.sources.add(source)
        
        print(f"Document divisé en {len(splits)} chunks")
        return splits
    
    def populate_vector_store(self, documents: List[Document]):
        """
        Ajoute les documents dans la base vectorielle.
        
        Args:
            documents: Liste des documents à ajouter
        """
        print("🔄 Ajout des documents dans la base vectorielle...")
        
        self.vector_store.add_documents(documents)
        self.documents_count += len(documents)
        
        # Reconfiguration de la stratégie hybride si nécessaire
        if (self.current_strategy and 
            self.current_strategy.name == 'HybridSearchRAG' and 
            hasattr(self.current_strategy, 'setup_bm25_retriever')):
            self._setup_hybrid_strategy()
        
        # Vérification
        collection_info = self.client.get_collection(self.collection_name)
        print(f"✅ {collection_info.points_count} documents au total")
    
    def ask_question(self, query: str, k: int = 5) -> RAGResult:
        """
        Pose une question avec la stratégie active.
        
        Args:
            query: Question à poser
            k: Nombre de documents à récupérer
            
        Returns:
            Résultat RAG avec métadonnées
        """
        if not self.current_strategy:
            raise ValueError("Aucune stratégie active. Utilisez set_strategy() d'abord.")
        
        print(f"\n🤔 Question: {query}")
        print(f"🎯 Stratégie utilisée: {self.current_strategy.name}")
        
        # Exécution de la stratégie
        result = self.current_strategy.retrieve_and_generate(query, k)
        
        # Ajout à l'historique
        self.query_history.append({
            'query': query,
            'strategy': self.current_strategy.name,
            'result': result,
            'timestamp': time.time()
        })
        
        # Affichage des résultats
        self._display_rag_result(result)
        
        return result
    
    def compare_strategies(self, query: str, strategies: List[str] = None, k: int = 5) -> RAGComparison:
        """
        Compare plusieurs stratégies RAG sur la même question.
        
        Args:
            query: Question à tester
            strategies: Liste des stratégies à comparer (toutes si None)
            k: Nombre de documents à récupérer
            
        Returns:
            Résultat de la comparaison
        """
        if strategies is None:
            strategies = list(self.available_strategies.keys())
        
        print(f"\n🆚 Comparaison de {len(strategies)} stratégies")
        print(f"❓ Question: {query}")
        print("=" * 80)
        
        results = {}
        
        for strategy_name in strategies:
            if strategy_name not in self.available_strategies:
                print(f"⚠️  Stratégie '{strategy_name}' non disponible")
                continue
            
            print(f"\n🎯 Test avec {strategy_name}...")
            
            # Changement temporaire de stratégie
            original_strategy = self.current_strategy
            self.current_strategy = self.available_strategies[strategy_name]
            
            # Configuration spécifique si nécessaire
            if strategy_name == 'hybrid' and hasattr(self.current_strategy, 'setup_bm25_retriever'):
                self._setup_hybrid_strategy()
            elif strategy_name == 'corrective' and hasattr(self.current_strategy, 'setup_evaluator'):
                self.current_strategy.setup_evaluator()
            
            # Exécution
            try:
                result = self.current_strategy.retrieve_and_generate(query, k)
                results[strategy_name] = result
                print(f"✅ {strategy_name}: {result.processing_time:.2f}s")
            except Exception as e:
                print(f"❌ {strategy_name}: Erreur - {e}")
                continue
            
            # Restauration
            self.current_strategy = original_strategy
        
        # Analyse des résultats
        best_strategy = self._analyze_comparison_results(results)
        
        # Métriques de comparaison
        comparison_metrics = {
            'strategies_tested': len(results),
            'fastest_strategy': min(results.keys(), key=lambda x: results[x].processing_time),
            'processing_times': {k: v.processing_time for k, v in results.items()},
            'document_counts': {k: len(v.sources) for k, v in results.items()}
        }
        
        comparison = RAGComparison(
            query=query,
            results=results,
            best_strategy=best_strategy,
            comparison_metrics=comparison_metrics
        )
        
        # Affichage des résultats
        self._display_comparison_results(comparison)
        
        return comparison
    
    def _analyze_comparison_results(self, results: Dict[str, RAGResult]) -> str:
        """
        Analyse les résultats de comparaison pour déterminer la meilleure stratégie.
        
        Args:
            results: Résultats de chaque stratégie
            
        Returns:
            Nom de la meilleure stratégie
        """
        if not results:
            return "Aucune"
        
        # Analyse simple basée sur le temps de traitement et le nombre de sources
        scores = {}
        
        for strategy_name, result in results.items():
            # Score basé sur rapidité et nombre de sources
            time_score = 1.0 / (result.processing_time + 0.1)  # Plus rapide = meilleur
            sources_score = len(result.sources) / 10.0  # Plus de sources = meilleur
            
            scores[strategy_name] = time_score + sources_score
        
        return max(scores.keys(), key=lambda x: scores[x])
    
    def _display_rag_result(self, result: RAGResult):
        """Affiche les résultats d'une requête RAG."""
        print(f"\n💬 Réponse ({result.processing_time:.2f}s):")
        print("-" * 60)
        print(result.response)
        
        print(f"\n📚 Sources utilisées ({len(result.sources)} documents):")
        for i, doc in enumerate(result.sources, 1):
            source = doc.metadata.get('source', 'Unknown')
            relevance = doc.metadata.get('relevance_score', 'N/A')
            print(f"   {i}. {source} (Score: {relevance})")
    
    def _display_comparison_results(self, comparison: RAGComparison):
        """Affiche les résultats d'une comparaison de stratégies."""
        print(f"\n📊 RÉSULTATS DE LA COMPARAISON")
        print("=" * 80)
        
        # Tableau de comparaison
        print(f"{'Stratégie':<15} {'Temps (s)':<10} {'Sources':<8} {'Longueur':<10}")
        print("-" * 50)
        
        for strategy_name, result in comparison.results.items():
            print(f"{strategy_name:<15} {result.processing_time:<10.2f} {len(result.sources):<8} {len(result.response):<10}")
        
        print(f"\n🏆 Meilleure stratégie: {comparison.best_strategy}")
        print(f"⚡ Stratégie la plus rapide: {comparison.comparison_metrics['fastest_strategy']}")
    
    def get_all_documents(self, limit: int = 100, with_vectors: bool = False) -> List[dict]:
        """
        Récupère tous les documents stockés (même logique que l'original).
        
        Args:
            limit: Nombre maximum de documents
            with_vectors: Inclure les vecteurs
            
        Returns:
            Liste des documents avec métadonnées
        """
        try:
            points = self.client.scroll(
                collection_name=self.collection_name,
                limit=limit,
                with_vectors=with_vectors,
                with_payload=True
            )
            
            documents = []
            for point in points[0]:
                doc_info = {
                    "id": point.id,
                    "content": point.payload.get("page_content", ""),
                    "metadata": point.payload.get("metadata", {}),
                }
                if with_vectors:
                    doc_info["vector"] = point.vector
                documents.append(doc_info)
            
            return documents
            
        except Exception as e:
            print(f"❌ Erreur lors de la récupération des documents: {e}")
            return []
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """
        Retourne un résumé des performances des différentes stratégies.
        
        Returns:
            Dict avec statistiques de performance
        """
        if not self.query_history:
            return {"message": "Aucune requête dans l'historique"}
        
        # Analyse de l'historique
        strategy_stats = {}
        
        for entry in self.query_history:
            strategy = entry['strategy']
            result = entry['result']
            
            if strategy not in strategy_stats:
                strategy_stats[strategy] = {
                    'count': 0,
                    'total_time': 0,
                    'total_sources': 0,
                    'total_response_length': 0
                }
            
            stats = strategy_stats[strategy]
            stats['count'] += 1
            stats['total_time'] += result.processing_time
            stats['total_sources'] += len(result.sources)
            stats['total_response_length'] += len(result.response)
        
        # Calcul des moyennes
        summary = {}
        for strategy, stats in strategy_stats.items():
            summary[strategy] = {
                'queries_count': stats['count'],
                'avg_processing_time': stats['total_time'] / stats['count'],
                'avg_sources_count': stats['total_sources'] / stats['count'],
                'avg_response_length': stats['total_response_length'] / stats['count']
            }
        
        return {
            'total_queries': len(self.query_history),
            'strategies_used': list(strategy_stats.keys()),
            'detailed_stats': summary
        }
    
    def display_performance_summary(self):
        """Affiche un résumé des performances."""
        summary = self.get_performance_summary()
        
        if "message" in summary:
            print(summary["message"])
            return
        
        print(f"\n📈 RÉSUMÉ DES PERFORMANCES")
        print("=" * 80)
        print(f"Total des requêtes: {summary['total_queries']}")
        print(f"Stratégies utilisées: {', '.join(summary['strategies_used'])}")
        
        print(f"\n{'Stratégie':<15} {'Requêtes':<10} {'Temps moy.':<12} {'Sources moy.':<12} {'Longueur moy.':<12}")
        print("-" * 75)
        
        for strategy, stats in summary['detailed_stats'].items():
            print(f"{strategy:<15} {stats['queries_count']:<10} {stats['avg_processing_time']:<12.2f} {stats['avg_sources_count']:<12.1f} {stats['avg_response_length']:<12.0f}")
    
    def run_enhanced_demo(self):
        """Démonstration des capacités améliorées."""
        print("🚀 Démonstration RAG Améliorée")
        print("=" * 60)
        
        try:
            # Initialisation complète
            self.setup_environment()
            self.initialize_models()
            self.setup_vector_store()
            self.initialize_strategies()
            
            # Chargement des documents
            print("\n📥 Chargement des documents...")
            url = "https://lilianweng.github.io/posts/2023-06-23-agent/"
            documents = self.load_and_process_documents(url)
            self.populate_vector_store(documents)
            
            # Test des différentes stratégies
            test_questions = [
                "What is an AI agent?",
                "How does memory work in LLM agents?",
                "What are the challenges in agent planning?"
            ]
            
            for question in test_questions:
                print(f"\n{'='*80}")
                print(f"Test de comparaison: {question}")
                print('='*80)
                
                comparison = self.compare_strategies(question)
                
                # Pause pour lecture
                input("\nAppuyez sur Entrée pour continuer...")
            
            # Résumé des performances
            print(f"\n{'='*80}")
            print("RÉSUMÉ FINAL DES PERFORMANCES")
            print('='*80)
            self.display_performance_summary()
            
        except Exception as e:
            print(f"❌ Erreur: {e}")
            import traceback
            traceback.print_exc()


def main():
    """Fonction principale pour tester l'application améliorée."""
    app = EnhancedRAGApplication()
    app.run_enhanced_demo()


if __name__ == "__main__":
    main()