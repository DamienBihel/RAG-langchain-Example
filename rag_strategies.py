"""
Stratégies RAG Avancées - Architecture Modulaire

Ce module implémente différentes stratégies RAG utilisant le pattern Strategy
pour permettre le changement dynamique de mode selon les besoins.

Stratégies disponibles :
1. NaiveRAG - Implémentation basique (actuelle)
2. HybridSearchRAG - Combine BM25 + recherche vectorielle
3. CorrectiveRAG - Évalue et corrige les documents récupérés
4. SelfReflectiveRAG - Auto-évalue et raffine les réponses
5. AgentBasedRAG - Routage intelligent des requêtes
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Tuple
import time
from dataclasses import dataclass

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import OpenAIEmbeddings
from langchain.chat_models import init_chat_model


@dataclass
class RAGResult:
    """Résultat d'une opération RAG avec métadonnées détaillées."""
    response: str
    sources: List[Document]
    metadata: Dict[str, Any]
    processing_time: float
    confidence_score: float = 0.0
    strategy_used: str = ""


class RAGStrategy(ABC):
    """
    Classe abstraite définissant l'interface pour les stratégies RAG.
    
    Utilise le pattern Strategy pour permettre le changement dynamique
    de comportement selon les besoins spécifiques.
    """
    
    def __init__(self, llm=None, embeddings=None, vector_store=None):
        """
        Initialise la stratégie RAG avec les composants nécessaires.
        
        Args:
            llm: Modèle de langage pour la génération
            embeddings: Modèle d'embeddings pour la vectorisation
            vector_store: Base vectorielle pour le stockage
        """
        self.llm = llm
        self.embeddings = embeddings
        self.vector_store = vector_store
        self.name = self.__class__.__name__
    
    @abstractmethod
    def retrieve(self, query: str, k: int = 5) -> List[Document]:
        """
        Récupère les documents pertinents pour une requête.
        
        Args:
            query: Requête utilisateur
            k: Nombre de documents à récupérer
            
        Returns:
            Liste des documents pertinents
        """
        pass
    
    @abstractmethod
    def generate(self, query: str, documents: List[Document]) -> str:
        """
        Génère une réponse basée sur les documents récupérés.
        
        Args:
            query: Requête utilisateur
            documents: Documents de contexte
            
        Returns:
            Réponse générée
        """
        pass
    
    def retrieve_and_generate(self, query: str, k: int = 5) -> RAGResult:
        """
        Pipeline RAG complet : récupération + génération.
        
        Args:
            query: Requête utilisateur
            k: Nombre de documents à récupérer
            
        Returns:
            RAGResult avec réponse et métadonnées
        """
        start_time = time.time()
        
        # Récupération des documents
        documents = self.retrieve(query, k)
        
        # Génération de la réponse
        response = self.generate(query, documents)
        
        # Calcul des métadonnées
        processing_time = time.time() - start_time
        metadata = {
            "query": query,
            "documents_count": len(documents),
            "strategy": self.name,
            "retrieval_time": processing_time
        }
        
        return RAGResult(
            response=response,
            sources=documents,
            metadata=metadata,
            processing_time=processing_time,
            strategy_used=self.name
        )
    
    def evaluate_documents(self, query: str, documents: List[Document]) -> List[float]:
        """
        Évalue la pertinence des documents pour une requête.
        
        Args:
            query: Requête utilisateur
            documents: Documents à évaluer
            
        Returns:
            Liste des scores de pertinence (0-1)
        """
        # Implémentation basique - peut être overridée
        return [1.0] * len(documents)


class NaiveRAG(RAGStrategy):
    """
    Stratégie RAG basique - Implémentation actuelle.
    
    Utilise uniquement la recherche vectorielle par similarité cosinus
    sans validation ni optimisation supplémentaire.
    """
    
    def retrieve(self, query: str, k: int = 5) -> List[Document]:
        """Recherche vectorielle basique."""
        if not self.vector_store:
            raise ValueError("Vector store not configured")
        
        return self.vector_store.similarity_search(query, k=k)
    
    def generate(self, query: str, documents: List[Document]) -> str:
        """Génération basique avec template simple."""
        if not self.llm:
            raise ValueError("LLM not configured")
        
        # Préparation du contexte
        context = "\n\n".join([doc.page_content for doc in documents])
        
        # Template de prompt basique
        prompt = ChatPromptTemplate.from_template("""
Vous êtes un assistant IA. Répondez à la question en vous basant sur le contexte fourni.

Contexte:
{context}

Question: {question}

Réponse:""")
        
        # Génération
        messages = prompt.format_messages(context=context, question=query)
        response = self.llm.invoke(messages)
        
        return response.content


class HybridSearchRAG(RAGStrategy):
    """
    Stratégie RAG hybride combinant recherche vectorielle et BM25.
    
    Utilise un EnsembleRetriever pour combiner :
    - Recherche sémantique (embeddings)
    - Recherche lexicale (BM25)
    """
    
    def __init__(self, llm=None, embeddings=None, vector_store=None, 
                 vector_weight: float = 0.7, bm25_weight: float = 0.3):
        """
        Initialise la stratégie hybride.
        
        Args:
            vector_weight: Poids pour la recherche vectorielle
            bm25_weight: Poids pour la recherche BM25
        """
        super().__init__(llm, embeddings, vector_store)
        self.vector_weight = vector_weight
        self.bm25_weight = bm25_weight
        self.bm25_retriever = None
        self.ensemble_retriever = None
    
    def setup_bm25_retriever(self, documents: List[Document]):
        """
        Configure le retrieveur BM25 avec les documents.
        
        Args:
            documents: Corpus de documents pour BM25
        """
        try:
            from langchain_community.retrievers import BM25Retriever
            from langchain.retrievers import EnsembleRetriever
            
            # Configuration BM25
            self.bm25_retriever = BM25Retriever.from_documents(documents)
            self.bm25_retriever.k = 5
            
            # Configuration ensemble
            if self.vector_store:
                vector_retriever = self.vector_store.as_retriever(search_kwargs={"k": 5})
                
                self.ensemble_retriever = EnsembleRetriever(
                    retrievers=[vector_retriever, self.bm25_retriever],
                    weights=[self.vector_weight, self.bm25_weight]
                )
                
                print(f"✅ Hybrid Search configuré (Vector: {self.vector_weight}, BM25: {self.bm25_weight})")
            
        except ImportError:
            print("❌ BM25Retriever non disponible. Utilisez: pip install rank-bm25")
            raise
    
    def retrieve(self, query: str, k: int = 5) -> List[Document]:
        """Recherche hybride BM25 + vectorielle."""
        if not self.ensemble_retriever:
            # Fallback sur recherche vectorielle
            print("⚠️  Ensemble retriever non configuré, utilisation vectorielle seule")
            return self.vector_store.similarity_search(query, k=k)
        
        # Recherche hybride
        results = self.ensemble_retriever.get_relevant_documents(query)
        return results[:k]
    
    def generate(self, query: str, documents: List[Document]) -> str:
        """Génération optimisée avec template amélioré."""
        if not self.llm:
            raise ValueError("LLM not configured")
        
        # Préparation du contexte avec scores
        context_parts = []
        for i, doc in enumerate(documents):
            source_info = doc.metadata.get('source', 'Unknown')
            context_parts.append(f"Document {i+1} (Source: {source_info}):\n{doc.page_content}")
        
        context = "\n\n".join(context_parts)
        
        # Template amélioré
        prompt = ChatPromptTemplate.from_template("""
Vous êtes un assistant IA expert. Analysez les documents fournis et répondez à la question.

INSTRUCTIONS:
- Basez-vous uniquement sur les documents fournis
- Citez les sources quand c'est pertinent
- Si l'information n'est pas dans les documents, indiquez-le clairement

DOCUMENTS:
{context}

QUESTION: {question}

RÉPONSE:""")
        
        messages = prompt.format_messages(context=context, question=query)
        response = self.llm.invoke(messages)
        
        return response.content


class CorrectiveRAG(RAGStrategy):
    """
    Stratégie RAG corrective qui évalue et corrige les documents récupérés.
    
    Processus :
    1. Récupération initiale des documents
    2. Évaluation de la pertinence
    3. Correction si nécessaire (recherche web, etc.)
    4. Génération avec documents corrigés
    """
    
    def __init__(self, llm=None, embeddings=None, vector_store=None,
                 relevance_threshold: float = 0.5, max_corrections: int = 2):
        """
        Initialise la stratégie corrective.
        
        Args:
            relevance_threshold: Seuil de pertinence pour correction
            max_corrections: Nombre maximum de corrections
        """
        super().__init__(llm, embeddings, vector_store)
        self.relevance_threshold = relevance_threshold
        self.max_corrections = max_corrections
        self.evaluator = None
    
    def setup_evaluator(self):
        """Configure l'évaluateur de pertinence."""
        try:
            from sentence_transformers import CrossEncoder
            
            # Modèle d'évaluation cross-encoder
            self.evaluator = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-12-v2')
            print("✅ Évaluateur de pertinence configuré")
            
        except ImportError:
            print("❌ CrossEncoder non disponible. Utilisez: pip install sentence-transformers")
            self.evaluator = None
    
    def evaluate_documents(self, query: str, documents: List[Document]) -> List[float]:
        """
        Évalue la pertinence des documents avec un modèle spécialisé.
        
        Args:
            query: Requête utilisateur
            documents: Documents à évaluer
            
        Returns:
            Liste des scores de pertinence (0-1)
        """
        if not self.evaluator:
            # Fallback sur évaluation basique
            return [1.0] * len(documents)
        
        # Évaluation avec cross-encoder
        pairs = [(query, doc.page_content[:512]) for doc in documents]  # Limite à 512 chars
        scores = self.evaluator.predict(pairs)
        
        # Normalisation des scores (0-1)
        normalized_scores = [(score + 1) / 2 for score in scores]
        
        return normalized_scores
    
    def retrieve(self, query: str, k: int = 5) -> List[Document]:
        """Récupération avec évaluation et correction."""
        if not self.vector_store:
            raise ValueError("Vector store not configured")
        
        # Récupération initiale
        documents = self.vector_store.similarity_search(query, k=k*2)  # Plus de documents
        
        # Évaluation des documents
        scores = self.evaluate_documents(query, documents)
        
        # Filtrage par pertinence
        relevant_docs = []
        for doc, score in zip(documents, scores):
            doc.metadata['relevance_score'] = score
            if score >= self.relevance_threshold:
                relevant_docs.append(doc)
        
        # Correction si pas assez de documents pertinents
        if len(relevant_docs) < k // 2:
            print(f"⚠️  Seulement {len(relevant_docs)} documents pertinents trouvés")
            print("🔄 Tentative de correction avec recherche web...")
            
            # Recherche web de secours
            web_docs = self.web_search_fallback(query, k - len(relevant_docs))
            relevant_docs.extend(web_docs)
        
        return relevant_docs[:k]
    
    def web_search_fallback(self, query: str, k: int) -> List[Document]:
        """
        Recherche web de secours quand les documents locaux sont insuffisants.
        
        Args:
            query: Requête de recherche
            k: Nombre de résultats souhaités
            
        Returns:
            Liste de documents depuis la recherche web
        """
        try:
            from duckduckgo_search import DDGS
            
            # Recherche DuckDuckGo
            with DDGS() as ddgs:
                results = list(ddgs.text(query, max_results=k))
            
            # Conversion en documents
            web_docs = []
            for result in results:
                doc = Document(
                    page_content=result.get('body', ''),
                    metadata={
                        'source': result.get('href', ''),
                        'title': result.get('title', ''),
                        'type': 'web_search',
                        'relevance_score': 0.8  # Score par défaut pour recherche web
                    }
                )
                web_docs.append(doc)
            
            print(f"✅ {len(web_docs)} documents récupérés via recherche web")
            return web_docs
            
        except ImportError:
            print("❌ DuckDuckGo search non disponible. Utilisez: pip install duckduckgo-search")
            return []
        except Exception as e:
            print(f"❌ Erreur recherche web: {e}")
            return []
    
    def generate(self, query: str, documents: List[Document]) -> str:
        """Génération avec prise en compte des scores de pertinence."""
        if not self.llm:
            raise ValueError("LLM not configured")
        
        # Tri des documents par pertinence
        documents_sorted = sorted(documents, 
                                key=lambda d: d.metadata.get('relevance_score', 0), 
                                reverse=True)
        
        # Préparation du contexte avec scores
        context_parts = []
        for i, doc in enumerate(documents_sorted):
            score = doc.metadata.get('relevance_score', 0)
            source = doc.metadata.get('source', 'Unknown')
            doc_type = doc.metadata.get('type', 'local')
            
            context_parts.append(
                f"Document {i+1} [Pertinence: {score:.2f}] [Type: {doc_type}] [Source: {source}]:\n"
                f"{doc.page_content[:800]}..."
            )
        
        context = "\n\n".join(context_parts)
        
        # Template avec gestion de la pertinence
        prompt = ChatPromptTemplate.from_template("""
Vous êtes un assistant IA expert en analyse de documents. 

INSTRUCTIONS:
- Analysez les documents fournis (triés par pertinence)
- Privilégiez les documents avec un score de pertinence élevé
- Indiquez si certaines informations proviennent de sources externes
- Soyez transparent sur la fiabilité des sources

DOCUMENTS (triés par pertinence):
{context}

QUESTION: {question}

RÉPONSE DÉTAILLÉE:""")
        
        messages = prompt.format_messages(context=context, question=query)
        response = self.llm.invoke(messages)
        
        return response.content


class SelfReflectiveRAG(RAGStrategy):
    """
    Stratégie RAG auto-réflexive qui évalue et raffine ses propres réponses.
    
    Processus :
    1. Génération initiale de la réponse
    2. Auto-évaluation de la réponse
    3. Raffinement si nécessaire
    4. Répétition jusqu'à satisfaction ou limite atteinte
    """
    
    def __init__(self, llm=None, embeddings=None, vector_store=None,
                 max_iterations: int = 3, quality_threshold: float = 0.8):
        """
        Initialise la stratégie auto-réflexive.
        
        Args:
            max_iterations: Nombre maximum d'itérations de raffinement
            quality_threshold: Seuil de qualité pour arrêter le raffinement
        """
        super().__init__(llm, embeddings, vector_store)
        self.max_iterations = max_iterations
        self.quality_threshold = quality_threshold
    
    def retrieve(self, query: str, k: int = 5) -> List[Document]:
        """Récupération vectorielle standard."""
        if not self.vector_store:
            raise ValueError("Vector store not configured")
        
        return self.vector_store.similarity_search(query, k=k)
    
    def evaluate_response(self, query: str, response: str, documents: List[Document]) -> Dict[str, float]:
        """
        Auto-évalue la qualité de la réponse générée.
        
        Args:
            query: Requête originale
            response: Réponse générée
            documents: Documents utilisés
            
        Returns:
            Dict avec scores de qualité
        """
        if not self.llm:
            raise ValueError("LLM not configured")
        
        # Template d'auto-évaluation
        evaluation_prompt = ChatPromptTemplate.from_template("""
Vous êtes un évaluateur expert. Analysez la réponse suivante et donnez des scores de 0 à 10.

QUESTION ORIGINALE: {question}

RÉPONSE À ÉVALUER: {response}

CONTEXTE DISPONIBLE: {context}

Évaluez selon ces critères (répondez UNIQUEMENT par les scores numériques):
1. PERTINENCE (0-10): La réponse répond-elle à la question?
2. PRÉCISION (0-10): Les informations sont-elles exactes?
3. COMPLÉTUDE (0-10): La réponse est-elle complète?
4. COHÉRENCE (0-10): La réponse est-elle logique et bien structurée?
5. UTILISATION_SOURCES (0-10): Les sources sont-elles bien utilisées?

Format de réponse:
PERTINENCE: X
PRÉCISION: X
COMPLÉTUDE: X
COHÉRENCE: X
UTILISATION_SOURCES: X""")
        
        # Préparation du contexte
        context = "\n".join([doc.page_content[:200] + "..." for doc in documents])
        
        # Évaluation
        messages = evaluation_prompt.format_messages(
            question=query,
            response=response,
            context=context
        )
        
        evaluation_response = self.llm.invoke(messages)
        
        # Parsing des scores
        scores = {}
        try:
            for line in evaluation_response.content.split('\n'):
                if ':' in line:
                    criterion, score_str = line.split(':', 1)
                    score = float(score_str.strip())
                    scores[criterion.strip()] = score / 10.0  # Normalisation 0-1
        except:
            # Fallback si parsing échoue
            scores = {
                'PERTINENCE': 0.7,
                'PRÉCISION': 0.7,
                'COMPLÉTUDE': 0.7,
                'COHÉRENCE': 0.7,
                'UTILISATION_SOURCES': 0.7
            }
        
        return scores
    
    def refine_response(self, query: str, initial_response: str, 
                       documents: List[Document], evaluation_scores: Dict[str, float]) -> str:
        """
        Raffine la réponse basée sur l'évaluation.
        
        Args:
            query: Requête originale
            initial_response: Réponse initiale
            documents: Documents utilisés
            evaluation_scores: Scores d'évaluation
            
        Returns:
            Réponse raffinée
        """
        if not self.llm:
            raise ValueError("LLM not configured")
        
        # Identification des points faibles
        weak_points = [k for k, v in evaluation_scores.items() if v < 0.6]
        
        if not weak_points:
            return initial_response
        
        # Template de raffinement
        refinement_prompt = ChatPromptTemplate.from_template("""
Vous devez améliorer la réponse suivante en vous concentrant sur les aspects faibles identifiés.

QUESTION ORIGINALE: {question}

RÉPONSE INITIALE: {initial_response}

CONTEXTE DISPONIBLE: {context}

ASPECTS À AMÉLIORER: {weak_points}

INSTRUCTIONS:
- Gardez les bonnes parties de la réponse initiale
- Concentrez-vous sur l'amélioration des aspects faibles
- Utilisez mieux le contexte disponible
- Soyez plus précis et complet

RÉPONSE AMÉLIORÉE:""")
        
        # Préparation du contexte
        context = "\n\n".join([doc.page_content for doc in documents])
        weak_points_str = ", ".join(weak_points)
        
        # Raffinement
        messages = refinement_prompt.format_messages(
            question=query,
            initial_response=initial_response,
            context=context,
            weak_points=weak_points_str
        )
        
        refined_response = self.llm.invoke(messages)
        
        return refined_response.content
    
    def generate(self, query: str, documents: List[Document]) -> str:
        """
        Génération avec auto-évaluation et raffinement itératif.
        
        Args:
            query: Requête utilisateur
            documents: Documents de contexte
            
        Returns:
            Réponse raffinée
        """
        if not self.llm:
            raise ValueError("LLM not configured")
        
        # Génération initiale
        initial_response = self._generate_initial_response(query, documents)
        current_response = initial_response
        
        print(f"🤔 Génération initiale terminée")
        
        # Boucle de raffinement
        for iteration in range(self.max_iterations):
            print(f"🔄 Itération de raffinement {iteration + 1}/{self.max_iterations}")
            
            # Auto-évaluation
            scores = self.evaluate_response(query, current_response, documents)
            
            # Calcul du score global
            overall_score = sum(scores.values()) / len(scores)
            print(f"📊 Score global: {overall_score:.2f}")
            
            # Arrêt si qualité suffisante
            if overall_score >= self.quality_threshold:
                print(f"✅ Qualité suffisante atteinte (seuil: {self.quality_threshold})")
                break
            
            # Raffinement
            current_response = self.refine_response(query, current_response, documents, scores)
            print(f"🔧 Réponse raffinée")
        
        return current_response
    
    def _generate_initial_response(self, query: str, documents: List[Document]) -> str:
        """Génère la réponse initiale."""
        context = "\n\n".join([doc.page_content for doc in documents])
        
        prompt = ChatPromptTemplate.from_template("""
Vous êtes un assistant IA expert. Répondez à la question de manière détaillée et précise.

CONTEXTE:
{context}

QUESTION: {question}

RÉPONSE:""")
        
        messages = prompt.format_messages(context=context, question=query)
        response = self.llm.invoke(messages)
        
        return response.content


# Factory pour créer les stratégies
class RAGStrategyFactory:
    """Factory pour créer les différentes stratégies RAG."""
    
    @staticmethod
    def create_strategy(strategy_type: str, **kwargs) -> RAGStrategy:
        """
        Crée une stratégie RAG selon le type spécifié.
        
        Args:
            strategy_type: Type de stratégie ('naive', 'hybrid', 'corrective', 'reflective')
            **kwargs: Arguments pour la stratégie
            
        Returns:
            Instance de la stratégie RAG
        """
        strategies = {
            'naive': NaiveRAG,
            'hybrid': HybridSearchRAG,
            'corrective': CorrectiveRAG,
            'reflective': SelfReflectiveRAG
        }
        
        if strategy_type not in strategies:
            raise ValueError(f"Stratégie inconnue: {strategy_type}. Disponibles: {list(strategies.keys())}")
        
        return strategies[strategy_type](**kwargs)
    
    @staticmethod
    def list_strategies() -> List[str]:
        """Retourne la liste des stratégies disponibles."""
        return ['naive', 'hybrid', 'corrective', 'reflective']