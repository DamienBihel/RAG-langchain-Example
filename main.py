"""
Application RAG optimisée avec LangChain et Qdrant
Architecture modulaire avec gestion d'erreurs robuste et RAG complet.

Cette application implémente un système RAG (Retrieval-Augmented Generation) complet :
1. Chargement et traitement de documents web
2. Découpage en chunks avec chevauchement
3. Génération d'embeddings vectoriels
4. Stockage dans base vectorielle Qdrant
5. Recherche sémantique et génération de réponses
"""

import getpass
import os
import bs4
from typing import List, Optional

# ============================================================================
# IMPORTS CONSOLIDÉS
# ============================================================================

# Gestion des variables d'environnement
from dotenv import load_dotenv

# Modèles LangChain
from langchain.chat_models import init_chat_model
from langchain_openai import OpenAIEmbeddings

# Base vectorielle et stockage
from langchain_qdrant import QdrantVectorStore
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Templates et prompts
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document

# Client Qdrant et gestion des erreurs
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
from qdrant_client.http.exceptions import UnexpectedResponse


class RAGApplication:
    """
    Application RAG modulaire avec toutes les fonctionnalités.
    
    Cette classe encapsule toute la logique RAG :
    - Configuration de l'environnement
    - Initialisation des modèles (LLM + Embeddings)
    - Gestion de la base vectorielle Qdrant
    - Chargement et traitement de documents
    - Recherche sémantique et génération de réponses
    """
    
    def __init__(self):
        """Initialise l'application RAG avec des attributs par défaut."""
        self.llm = None                    # Modèle de langage (GPT-4o-mini)
        self.embeddings = None             # Modèle d'embeddings (text-embedding-3-small)
        self.vector_store = None           # Interface vers la base vectorielle Qdrant
        self.client = None                 # Client Qdrant pour la gestion des collections
        self.collection_name = "documents" # Nom de la collection dans Qdrant
        
    def setup_environment(self):
        """
        Configure l'environnement et les clés API.
        
        Charge les variables d'environnement depuis le fichier .env
        et demande la clé API OpenAI si elle n'est pas trouvée.
        """
        load_dotenv()
        
        # Vérification et demande de la clé API si nécessaire
        if not os.environ.get("OPENAI_API_KEY"):
            os.environ["OPENAI_API_KEY"] = getpass.getpass("Entrez votre clé API OpenAI: ")
    
    def initialize_models(self):
        """
        Initialise les modèles LLM et embeddings.
        
        Configure :
        - Modèle de chat GPT-4o-mini avec température basse (0.1) pour des réponses cohérentes
        - Modèle d'embeddings text-embedding-3-small (optimisé coût/performance)
        """
        print("🔧 Initialisation des modèles...")
        
        # Modèle de chat optimisé avec température basse pour des réponses cohérentes
        self.llm = init_chat_model(
            "gpt-4o-mini", 
            model_provider="openai",
            temperature=0.1  # Réponses plus déterministes
        )
        
        # Embeddings optimisés (moins coûteux que text-embedding-3-large)
        # Taille des vecteurs : 1536 dimensions
        self.embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        
        print("✅ Modèles initialisés avec succès")
    
    def setup_vector_store(self):
        """
        Configure la base de données vectorielle Qdrant.
        
        Crée ou récupère une collection Qdrant avec les paramètres optimaux :
        - Distance COSINE pour la similarité sémantique
        - Taille des vecteurs adaptée au modèle d'embedding
        - Client en mémoire pour le développement (peut être persisté en production)
        """
        print("🗄️ Configuration de la base vectorielle...")
        
        # Client Qdrant en mémoire RAM (pour le développement/test)
        # En production, utilisez une instance persistante : QdrantClient("localhost", port=6333)
        self.client = QdrantClient(":memory:")
        
        # Gestion robuste de la collection : création si elle n'existe pas
        try:
            self.client.get_collection(self.collection_name)
            print(f"Collection '{self.collection_name}' trouvée")
        except ValueError:
            # Crée la collection avec les paramètres optimaux
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=1536,  # Taille pour text-embedding-3-small
                    distance=Distance.COSINE  # Métrique de similarité sémantique
                )
            )
            print(f"Collection '{self.collection_name}' créée")
        
        # Initialise l'interface vector store pour LangChain
        self.vector_store = QdrantVectorStore(
            client=self.client,
            collection_name=self.collection_name,
            embedding=self.embeddings,
        )
        
        print("✅ Base vectorielle configurée")
    
    def load_and_process_documents(self, url: str) -> List[Document]:
        """
        Charge et traite les documents depuis une URL.
        
        Args:
            url (str): URL du document web à charger
            
        Returns:
            List[Document]: Liste des chunks de documents traités
            
        Processus :
        1. Chargement du document web avec BeautifulSoup
        2. Extraction du contenu pertinent (titre, en-tête, contenu)
        3. Découpage en chunks avec chevauchement
        4. Retour des chunks prêts pour l'embedding
        """
        print(f"📄 Chargement du document depuis {url}...")
        
        # Configuration du loader web avec BeautifulSoup
        # Extraction sélective : seulement titre, en-tête et contenu principal
        bs4_strainer = bs4.SoupStrainer(class_=("post-title", "post-header", "post-content"))
        loader = WebBaseLoader(
            web_paths=(url,),
            bs_kwargs={"parse_only": bs4_strainer},  # Optimisation : parse seulement les éléments nécessaires
        )
        
        # Chargement du document
        docs = loader.load()
        print(f"Document chargé: {len(docs[0].page_content)} caractères")
        
        # Découpage en chunks optimisé pour RAG
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,      # Taille optimale pour les embeddings
            chunk_overlap=200,     # Chevauchement pour maintenir le contexte
            add_start_index=True,  # Suivi de la position dans le document original
        )
        
        splits = text_splitter.split_documents(docs)
        print(f"Document divisé en {len(splits)} chunks")
        
        return splits
    
    def populate_vector_store(self, documents: List[Document]):
        """
        Ajoute les documents dans la base vectorielle.
        
        Args:
            documents (List[Document]): Liste des chunks à ajouter
            
        Processus :
        1. Génération automatique des embeddings pour chaque chunk
        2. Stockage dans la collection Qdrant
        3. Vérification du nombre de documents ajoutés
        """
        print("🔄 Ajout des documents dans la base vectorielle...")
        
        # Ajout automatique avec génération d'embeddings
        self.vector_store.add_documents(documents)
        
        # Vérification du stockage
        collection_info = self.client.get_collection(self.collection_name)
        print(f"✅ {collection_info.points_count} documents ajoutés")
    
    def search_documents(self, query: str, k: int = 3) -> List[Document]:
        """
        Recherche les documents pertinents pour une requête.
        
        Args:
            query (str): Requête de recherche
            k (int): Nombre de documents à retourner (défaut: 3)
            
        Returns:
            List[Document]: Documents les plus pertinents
            
        Utilise la recherche par similarité cosinus dans l'espace vectoriel.
        """
        return self.vector_store.similarity_search(query, k=k)
    
    def generate_response(self, query: str, context_docs: List[Document]) -> str:
        """
        Génère une réponse basée sur les documents trouvés.
        
        Args:
            query (str): Question de l'utilisateur
            context_docs (List[Document]): Documents de contexte
            
        Returns:
            str: Réponse générée par le LLM
            
        Processus :
        1. Préparation du contexte à partir des documents
        2. Génération du prompt avec template optimisé
        3. Appel au LLM pour générer la réponse
        """
        # Prépare le contexte en concaténant les documents
        context = "\n\n".join([doc.page_content for doc in context_docs])
        
        # Template de prompt optimisé pour RAG
        prompt = ChatPromptTemplate.from_template("""
Vous êtes un assistant IA spécialisé dans l'analyse de documents.
Répondez à la question en vous basant uniquement sur le contexte fourni.

Contexte:
{context}

Question: {question}

Réponse:""")
        
        # Génération de la réponse via le LLM
        messages = prompt.format_messages(context=context, question=query)
        response = self.llm.invoke(messages)
        
        return response.content
    
    def ask_question(self, query: str) -> tuple[str, List[Document]]:
        """
        Interface complète RAG: recherche + génération.
        
        Args:
            query (str): Question de l'utilisateur
            
        Returns:
            tuple[str, List[Document]]: (réponse générée, documents sources)
            
        Implémente le pipeline RAG complet :
        1. Recherche sémantique des documents pertinents
        2. Génération de réponse basée sur le contexte
        """
        print(f"\n🤔 Question: {query}")
        
        # Étape 1: Recherche des documents pertinents
        relevant_docs = self.search_documents(query)
        
        # Étape 2: Génération de la réponse basée sur le contexte
        response = self.generate_response(query, relevant_docs)
        
        return response, relevant_docs
    
    def run_demo(self):
        """
        Démonstration complète de l'application RAG.
        
        Exécute le pipeline RAG complet :
        1. Configuration de l'environnement
        2. Initialisation des modèles
        3. Configuration de la base vectorielle
        4. Chargement et traitement des documents
        5. Tests avec différentes questions
        """
        try:
            # Configuration complète de l'application
            self.setup_environment()
            self.initialize_models()
            self.setup_vector_store()
            
            # Chargement et traitement des documents
            url = "https://lilianweng.github.io/posts/2023-06-23-agent/"
            documents = self.load_and_process_documents(url)
            self.populate_vector_store(documents)
            
            # Tests RAG avec différentes questions
            questions = [
                "What is the main topic of this article?",
                "What are the key components of autonomous agents?",
                "How does planning work in AI agents?"
            ]
            
            print("\n" + "="*60)
            print("🎯 DÉMONSTRATION RAG COMPLÈTE")
            print("="*60)
            
            # Exécution des tests
            for question in questions:
                response, sources = self.ask_question(question)
                
                print(f"\n💬 Réponse:\n{response}")
                print(f"\n📚 Sources utilisées: {len(sources)} documents")
                print("-" * 40)
            
            print("\n🎉 Application RAG complète et fonctionnelle !")
            
        except Exception as e:
            print(f"❌ Erreur: {e}")
            import traceback
            traceback.print_exc()


def main():
    """
    Point d'entrée principal de l'application.
    
    Crée une instance de RAGApplication et lance la démonstration.
    """
    print("🚀 Démarrage de l'application RAG")
    print("=" * 50)
    
    app = RAGApplication()
    app.run_demo()


if __name__ == "__main__":
    main()