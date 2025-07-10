"""
Application RAG optimisée avec LangChain et Qdrant
Architecture modulaire avec gestion d'erreurs robuste et RAG complet.

Ce projet pédagogique implémente un système RAG (Retrieval-Augmented Generation) complet :
1. Chargement et traitement de documents web
2. Découpage en chunks avec chevauchement
3. Génération d'embeddings vectoriels
4. Stockage dans base vectorielle Qdrant
5. Recherche sémantique et génération de réponses

CONCEPTS PÉDAGOGIQUES EXPLIQUÉS :
- RAG (Retrieval-Augmented Generation) : Technique qui combine recherche et génération
- Embeddings : Représentations vectorielles du texte pour la similarité sémantique
- Vector Store : Base de données spécialisée pour stocker et rechercher des vecteurs
- Chunking : Découpage des documents en morceaux pour optimiser le traitement
- Similarity Search : Recherche par similarité cosinus dans l'espace vectoriel
- Prompt Engineering : Conception de prompts pour guider le LLM
"""

import getpass
import os
import bs4
from typing import List, Optional

# ============================================================================
# IMPORTS CONSOLIDÉS - EXPLICATION DES BIBLIOTHÈQUES
# ============================================================================

# Gestion des variables d'environnement (sécurité des clés API)
from dotenv import load_dotenv

# Modèles LangChain - Framework pour applications LLM
from langchain.chat_models import init_chat_model
from langchain_openai import OpenAIEmbeddings

# Base vectorielle et stockage - Gestion des embeddings
from langchain_qdrant import QdrantVectorStore
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Templates et prompts - Conception de prompts pour le LLM
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document

# Client Qdrant et gestion des erreurs - Base de données vectorielle
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
from qdrant_client.http.exceptions import UnexpectedResponse


class RAGApplication:
    """
    Application RAG modulaire avec toutes les fonctionnalités.
    
    Cette classe encapsule toute la logique RAG et sert d'exemple pédagogique
    pour comprendre l'architecture d'un système RAG complet.
    
    CONCEPTS PÉDAGOGIQUES :
    - Architecture modulaire : Séparation des responsabilités
    - Pipeline RAG : Flux de données du document à la réponse
    - Gestion d'erreurs robuste : Traitement des cas d'échec
    - Configuration flexible : Adaptation aux différents besoins
    
    Cette classe implémente :
    - Configuration de l'environnement
    - Initialisation des modèles (LLM + Embeddings)
    - Gestion de la base vectorielle Qdrant
    - Chargement et traitement de documents
    - Recherche sémantique et génération de réponses
    """
    
    def __init__(self):
        """
        Initialise l'application RAG avec des attributs par défaut.
        
        CONCEPTS PÉDAGOGIQUES :
        - Encapsulation : Regroupement des données et méthodes
        - État de l'application : Variables d'instance pour maintenir l'état
        - Configuration par défaut : Valeurs initiales raisonnables
        """
        # Modèle de langage (LLM) - Cerveau du système RAG
        self.llm = None                    # Modèle de langage (GPT-4o-mini)
        
        # Modèle d'embeddings - Conversion texte → vecteurs
        self.embeddings = None             # Modèle d'embeddings (text-embedding-3-small)
        
        # Interface vers la base vectorielle - Stockage des embeddings
        self.vector_store = None           # Interface vers la base vectorielle Qdrant
        
        # Client Qdrant - Gestion directe de la base de données
        self.client = None                 # Client Qdrant pour la gestion des collections
        
        # Nom de la collection - Organisation des données
        self.collection_name = "documents" # Nom de la collection dans Qdrant
        
    def setup_environment(self):
        """
        Configure l'environnement et les clés API.
        
        CONCEPTS PÉDAGOGIQUES :
        - Sécurité des clés API : Protection des informations sensibles
        - Variables d'environnement : Configuration externalisée
        - Gestion d'erreurs : Vérification de la présence des clés
        
        Charge les variables d'environnement depuis le fichier .env
        et demande la clé API OpenAI si elle n'est pas trouvée.
        """
        # Chargement des variables d'environnement depuis .env
        load_dotenv()
        
        # Vérification et demande de la clé API si nécessaire
        # BONNE PRATIQUE : Ne jamais hardcoder les clés API dans le code
        if not os.environ.get("OPENAI_API_KEY"):
            os.environ["OPENAI_API_KEY"] = getpass.getpass("Entrez votre clé API OpenAI: ")
    
    def initialize_models(self):
        """
        Initialise les modèles LLM et embeddings.
        
        CONCEPTS PÉDAGOGIQUES :
        - Modèles de langage : Compréhension et génération de texte
        - Embeddings : Représentations vectorielles pour la similarité
        - Température : Contrôle de la créativité vs cohérence
        - Optimisation coût/performance : Choix de modèles équilibrés
        
        Configure :
        - Modèle de chat GPT-4o-mini avec température basse (0.1) pour des réponses cohérentes
        - Modèle d'embeddings text-embedding-3-small (optimisé coût/performance)
        """
        print("🔧 Initialisation des modèles...")
        
        # Modèle de chat optimisé avec température basse pour des réponses cohérentes
        # CONCEPT : La température contrôle la "créativité" du modèle
        # - 0.0 = très déterministe, 1.0 = très créatif
        self.llm = init_chat_model(
            "gpt-4o-mini", 
            model_provider="openai",
            temperature=0.1  # Réponses plus déterministes
        )
        
        # Embeddings optimisés (moins coûteux que text-embedding-3-large)
        # CONCEPT : Les embeddings convertissent le texte en vecteurs numériques
        # - Permettent la recherche sémantique (similarité de sens)
        # - Taille des vecteurs : 1536 dimensions pour text-embedding-3-small
        self.embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        
        print("✅ Modèles initialisés avec succès")
    
    def setup_vector_store(self):
        """
        Configure la base de données vectorielle Qdrant.
        
        CONCEPTS PÉDAGOGIQUES :
        - Base de données vectorielle : Stockage optimisé pour les embeddings
        - Distance cosinus : Métrique de similarité sémantique
        - Collection : Organisation des données par projet/domaine
        - Client en mémoire vs persistant : Choix selon l'usage
        
        Crée ou récupère une collection Qdrant avec les paramètres optimaux :
        - Distance COSINE pour la similarité sémantique
        - Taille des vecteurs adaptée au modèle d'embedding
        - Client en mémoire pour le développement (peut être persisté en production)
        """
        print("🗄️ Configuration de la base vectorielle...")
        
        # Client Qdrant en mémoire (pour le développement/test)
        # CONCEPT : Base de données vectorielle spécialisée
        # - Optimisée pour la recherche de similarité
        # - En production, utilisez une instance persistante : QdrantClient("localhost", port=6333)
        self.client = QdrantClient(":memory:")
        
        # Gestion robuste de la collection : création si elle n'existe pas
        # CONCEPT : Gestion d'erreurs pour la robustesse
        try:
            self.client.get_collection(self.collection_name)
            print(f"Collection '{self.collection_name}' trouvée")
        except ValueError:
            # Crée la collection avec les paramètres optimaux
            # CONCEPT : Configuration de la base vectorielle
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=1536,  # Taille pour text-embedding-3-small
                    distance=Distance.COSINE  # Métrique de similarité sémantique
                )
            )
            print(f"Collection '{self.collection_name}' créée")
        
        # Initialise l'interface vector store pour LangChain
        # CONCEPT : Abstraction pour simplifier l'utilisation
        self.vector_store = QdrantVectorStore(
            client=self.client,
            collection_name=self.collection_name,
            embedding=self.embeddings,
        )
        
        print("✅ Base vectorielle configurée")
    
    def load_and_process_documents(self, source: str, file_type: str = "auto") -> List[Document]:
        """
        Charge et traite les documents depuis une source.
        
        CONCEPTS PÉDAGOGIQUES :
        - Document Loaders : Chargement de différents formats
        - Text Splitting : Découpage intelligent des documents
        - Chunking : Optimisation pour les embeddings
        - Métadonnées : Informations contextuelles sur les chunks
        
        Args:
            source (str): Chemin vers le document (URL ou fichier)
            file_type (str): Type de fichier ("web", "pdf", "markdown", "auto")
            
        Returns:
            List[Document]: Liste des chunks de documents traités
            
        Processus :
        1. Chargement du document selon son type
        2. Découpage en chunks avec chevauchement
        3. Enrichissement des métadonnées
        4. Retour des chunks prêts pour l'embedding
        """
        print(f"📄 Chargement du document depuis {source}...")
        
        # Détection automatique du type de fichier si non spécifié
        if file_type == "auto":
            if source.startswith("http"):
                file_type = "web"
            elif source.endswith(".pdf"):
                file_type = "pdf"
            elif source.endswith(".md"):
                file_type = "markdown"
            else:
                file_type = "text"
        
        # Chargement selon le type de document
        # CONCEPT : Adapter le loader au type de document
        if file_type == "web":
            # Configuration du loader web avec BeautifulSoup
            # CONCEPT : Parsing HTML pour extraire le contenu pertinent
            bs4_strainer = bs4.SoupStrainer(class_=("post-title", "post-header", "post-content"))
            loader = WebBaseLoader(
                web_paths=(source,),
                bs_kwargs={"parse_only": bs4_strainer},  # Optimisation : parse seulement les éléments nécessaires
            )
        elif file_type == "pdf":
            # CONCEPT : Chargement de PDF avec extraction de texte
            from langchain_community.document_loaders import PyPDFLoader
            loader = PyPDFLoader(source)
        elif file_type == "markdown":
            # CONCEPT : Chargement de Markdown avec préservation de la structure
            from langchain_community.document_loaders import UnstructuredMarkdownLoader
            loader = UnstructuredMarkdownLoader(source)
        else:
            # CONCEPT : Chargement de texte brut
            from langchain_community.document_loaders import TextLoader
            loader = TextLoader(source)
        
        # Chargement du document
        docs = loader.load()
        print(f"Document chargé: {len(docs[0].page_content)} caractères")
        
        # Découpage en chunks optimisé pour RAG
        # CONCEPT : Text Splitting pour optimiser les embeddings
        # - Chunks trop petits = perte de contexte
        # - Chunks trop grands = embeddings moins précis
        # - Chevauchement = maintien du contexte entre chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,      # Taille optimale pour les embeddings
            chunk_overlap=200,     # Chevauchement pour maintenir le contexte
            add_start_index=True,  # Suivi de la position dans le document original
        )
        
        splits = text_splitter.split_documents(docs)
        
        # Enrichir les métadonnées des chunks
        # CONCEPT : Métadonnées pour la traçabilité et le contexte
        for i, split in enumerate(splits):
            split.metadata["source"] = source
            split.metadata["file_type"] = file_type
            split.metadata["chunk_index"] = i
            split.metadata["total_chunks"] = len(splits)
        
        print(f"Document divisé en {len(splits)} chunks")
        
        return splits
    
    def populate_vector_store(self, documents: List[Document]):
        """
        Ajoute les documents dans la base vectorielle.
        
        CONCEPTS PÉDAGOGIQUES :
        - Génération d'embeddings : Conversion automatique texte → vecteurs
        - Stockage vectoriel : Organisation optimisée pour la recherche
        - Vérification : Contrôle de qualité du stockage
        
        Args:
            documents (List[Document]): Liste des chunks à ajouter
            
        Processus :
        1. Génération automatique des embeddings pour chaque chunk
        2. Stockage dans la collection Qdrant
        3. Vérification du nombre de documents ajoutés
        """
        print("🔄 Ajout des documents dans la base vectorielle...")
        
        # Ajout automatique avec génération d'embeddings
        # CONCEPT : Pipeline automatique de traitement
        # - Chaque chunk est converti en embedding
        # - Les embeddings sont stockés avec leurs métadonnées
        self.vector_store.add_documents(documents)
        
        # Vérification du stockage
        # CONCEPT : Contrôle de qualité et monitoring
        collection_info = self.client.get_collection(self.collection_name)
        print(f"✅ {collection_info.points_count} documents ajoutés")
    
    def search_documents(self, query: str, k: int = 3) -> List[Document]:
        """
        Recherche les documents pertinents pour une requête.
        
        CONCEPTS PÉDAGOGIQUES :
        - Recherche sémantique : Trouver des documents par similarité de sens
        - Similarité cosinus : Métrique pour mesurer la similarité entre vecteurs
        - Paramètre k : Nombre de documents à retourner (trade-off précision/recouvrement)
        
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
        
        CONCEPTS PÉDAGOGIQUES :
        - Prompt Engineering : Conception de prompts pour guider le LLM
        - Contexte : Information fournie au LLM pour la génération
        - Template : Structure réutilisable pour les prompts
        
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
        # CONCEPT : Augmentation du prompt avec le contexte
        context = "\n\n".join([doc.page_content for doc in context_docs])
        
        # Template de prompt optimisé pour RAG
        # CONCEPT : Prompt Engineering pour des réponses de qualité
        # - Instructions claires pour le LLM
        # - Séparation contexte/question
        # - Guidage pour la réponse
        prompt = ChatPromptTemplate.from_template("""
Vous êtes un assistant IA spécialisé dans l'analyse de documents.
Répondez à la question en vous basant uniquement sur le contexte fourni.

Contexte:
{context}

Question: {question}

Réponse:""")
        
        # Génération de la réponse via le LLM
        # CONCEPT : Appel au modèle de langage avec le prompt formaté
        messages = prompt.format_messages(context=context, question=query)
        response = self.llm.invoke(messages)
        
        return response.content
    
    def ask_question(self, query: str) -> tuple[str, List[Document]]:
        """
        Interface complète RAG: recherche + génération.
        
        CONCEPTS PÉDAGOGIQUES :
        - Pipeline RAG : Flux complet de traitement
        - Recherche + Génération : Deux étapes distinctes
        - Transparence : Affichage des sources utilisées
        
        Args:
            query (str): Question de l'utilisateur
            
        Returns:
            tuple[str, List[Document]]: (réponse générée, documents sources)
            
        Implémente le pipeline RAG complet :
        1. Recherche sémantique des documents pertinents
        2. Génération de réponse basée sur le contexte
        3. Affichage détaillé des documents sources utilisés
        """
        print(f"\n🤔 Question: {query}")
        
        # Étape 1: Recherche des documents pertinents
        # CONCEPT : Retrieval - Trouver les informations pertinentes
        relevant_docs = self.search_documents(query)
        
        # Étape 2: Génération de la réponse basée sur le contexte
        # CONCEPT : Generation - Créer une réponse basée sur le contexte
        response = self.generate_response(query, relevant_docs)
        
        # Étape 3: Affichage détaillé des documents sources
        # CONCEPT : Transparence et traçabilité
        print(f"\n📚 DOCUMENTS UTILISÉS DANS LE RAG ({len(relevant_docs)} documents):")
        print("=" * 80)
        
        for i, doc in enumerate(relevant_docs, 1):
            print(f"\n📄 Document #{i}")
            print("-" * 40)
            
            # Affichage des métadonnées
            # CONCEPT : Métadonnées pour comprendre la source
            metadata = doc.metadata
            if metadata:
                print("📌 Métadonnées:")
                for key, value in metadata.items():
                    print(f"   • {key}: {value}")
            
            # Affichage du contenu (premiers 300 caractères)
            # CONCEPT : Aperçu du contenu utilisé
            content = doc.page_content
            print(f"📝 Contenu ({len(content)} caractères):")
            print(f"   {content[:300]}...")
            if len(content) > 300:
                print(f"   ... (tronqué, {len(content) - 300} caractères supplémentaires)")
        
        print("=" * 80)
        
        return response, relevant_docs
    
    def get_all_documents(self, limit: int = 100, with_vectors: bool = False) -> List[dict]:
        """
        Récupère tous les documents stockés dans la base vectorielle.
        
        Args:
            limit (int): Nombre maximum de documents à récupérer (défaut: 100)
            with_vectors (bool): Inclure les vecteurs dans la réponse (défaut: False)
            
        Returns:
            List[dict]: Liste des documents avec leurs métadonnées
        """
        print(f"📋 Récupération des documents depuis la collection '{self.collection_name}'...")
        
        try:
            # Récupération des points depuis Qdrant
            points = self.client.scroll(
                collection_name=self.collection_name,
                limit=limit,
                with_vectors=with_vectors,
                with_payload=True
            )
            
            documents = []
            for point in points[0]:  # points[0] contient la liste des points
                doc_info = {
                    "id": point.id,
                    "content": point.payload.get("page_content", ""),
                    "metadata": point.payload.get("metadata", {}),
                }
                if with_vectors:
                    doc_info["vector"] = point.vector
                documents.append(doc_info)
            
            print(f"✅ {len(documents)} documents trouvés")
            return documents
            
        except Exception as e:
            print(f"❌ Erreur lors de la récupération des documents: {e}")
            return []
    
    def display_documents(self, limit: int = 10, show_content: bool = True, max_content_length: int = 500):
        """
        Affiche les documents stockés dans la base vectorielle de façon formatée.
        
        Args:
            limit (int): Nombre de documents à afficher (défaut: 10)
            show_content (bool): Afficher le contenu des documents (défaut: True)
            max_content_length (int): Longueur maximale du contenu à afficher (défaut: 500)
        """
        documents = self.get_all_documents(limit=limit)
        
        if not documents:
            print("Aucun document trouvé dans la base vectorielle.")
            return
        
        print(f"\n{'='*80}")
        print(f"📚 DOCUMENTS DANS LA BASE VECTORIELLE ({len(documents)} documents)")
        print(f"{'='*80}")
        
        for i, doc in enumerate(documents, 1):
            print(f"\n{'─'*60}")
            print(f"📄 Document #{i} (ID: {doc['id']})")
            print(f"{'─'*60}")
            
            # Affichage des métadonnées
            metadata = doc.get('metadata', {})
            if metadata:
                print("📌 Métadonnées:")
                for key, value in metadata.items():
                    print(f"   • {key}: {value}")
            
            # Affichage du contenu
            if show_content and doc.get('content'):
                content = doc['content']
                if len(content) > max_content_length:
                    content = content[:max_content_length] + "..."
                print(f"\n📝 Contenu:")
                print(f"   {content}")
            
            print(f"\n   Taille du contenu: {len(doc.get('content', ''))} caractères")
    
    def search_and_display(self, query: str, k: int = 5):
        """
        Recherche et affiche les documents pertinents avec leurs scores de similarité.
        
        Args:
            query (str): Requête de recherche
            k (int): Nombre de documents à retourner (défaut: 5)
        """
        print(f"\n🔍 Recherche: '{query}'")
        print(f"{'='*60}")
        
        # Recherche avec scores
        results = self.vector_store.similarity_search_with_score(query, k=k)
        
        if not results:
            print("Aucun document pertinent trouvé.")
            return
        
        print(f"📊 {len(results)} documents trouvés (triés par pertinence)")
        print(f"{'='*60}")
        
        # Grouper par source
        sources_found = set()
        
        for i, (doc, score) in enumerate(results, 1):
            print(f"\n🏆 Résultat #{i}")
            print(f"   Score de similarité: {score:.4f}")
            
            # Métadonnées avec mise en évidence de la source
            if doc.metadata:
                source = doc.metadata.get('source', 'Unknown')
                sources_found.add(source)
                print(f"   📍 Source: {source}")
                print("   📌 Autres métadonnées:")
                for key, value in doc.metadata.items():
                    if key != 'source':
                        print(f"      • {key}: {value}")
            
            # Contenu (extrait)
            content = doc.page_content[:300] + "..." if len(doc.page_content) > 300 else doc.page_content
            print(f"   📝 Extrait:")
            print(f"      {content}")
            print(f"   📏 Taille totale: {len(doc.page_content)} caractères")
        
        # Résumé des sources
        if sources_found:
            print(f"\n📚 Sources utilisées: {', '.join(sources_found)}")
    
    def get_sources_summary(self) -> dict:
        """
        Obtient un résumé des sources chargées dans la base vectorielle.
        
        Returns:
            dict: Résumé avec sources, nombre de documents par source, etc.
        """
        documents = self.get_all_documents(limit=10000)
        
        sources_info = {}
        total_size = 0
        
        for doc in documents:
            source = doc['metadata'].get('source', 'Unknown')
            file_type = doc['metadata'].get('file_type', 'Unknown')
            
            if source not in sources_info:
                sources_info[source] = {
                    'file_type': file_type,
                    'chunks_count': 0,
                    'total_size': 0,
                    'chunk_indices': []
                }
            
            sources_info[source]['chunks_count'] += 1
            sources_info[source]['total_size'] += len(doc['content'])
            sources_info[source]['chunk_indices'].append(doc['metadata'].get('chunk_index', -1))
            total_size += len(doc['content'])
        
        return {
            'sources': sources_info,
            'total_sources': len(sources_info),
            'total_chunks': len(documents),
            'total_size': total_size
        }
    
    def display_sources(self):
        """Affiche un résumé formaté des sources chargées."""
        summary = self.get_sources_summary()
        
        print(f"\n{'='*80}")
        print(f"📚 RÉSUMÉ DES SOURCES CHARGÉES")
        print(f"{'='*80}")
        
        print(f"\n📊 Statistiques globales:")
        print(f"   • Nombre de sources: {summary['total_sources']}")
        print(f"   • Nombre total de chunks: {summary['total_chunks']}")
        print(f"   • Taille totale: {summary['total_size']:,} caractères")
        
        if summary['sources']:
            print(f"\n📑 Détail par source:")
            print(f"{'─'*80}")
            
            for source, info in summary['sources'].items():
                print(f"\n📄 {source}")
                print(f"   • Type: {info['file_type']}")
                print(f"   • Chunks: {info['chunks_count']}")
                print(f"   • Taille: {info['total_size']:,} caractères")
                print(f"   • Taille moyenne par chunk: {info['total_size'] // info['chunks_count']:,} caractères")
    
    def load_all_documents_from_folder(self, folder_path: str = "./documents") -> List[Document]:
        """
        Charge tous les documents du dossier spécifié.
        
        Args:
            folder_path (str): Chemin vers le dossier contenant les documents
            
        Returns:
            List[Document]: Liste de tous les documents chargés
            
        Processus :
        1. Parcours récursif du dossier
        2. Détection automatique du type de fichier
        3. Chargement et traitement de chaque document
        4. Retour de tous les documents combinés
        """
        print(f"📁 Chargement de tous les documents depuis {folder_path}...")
        
        all_documents = []
        
        try:
            import os
            import glob
            
            # Parcours récursif du dossier
            for root, dirs, files in os.walk(folder_path):
                for file in files:
                    # Ignorer les fichiers système et temporaires
                    if file.startswith('.') or file.endswith(('.tmp', '.temp')):
                        continue
                    
                    file_path = os.path.join(root, file)
                    relative_path = os.path.relpath(file_path, folder_path)
                    
                    print(f"📄 Traitement de {relative_path}...")
                    
                    try:
                        # Détection du type de fichier
                        if file.endswith('.md'):
                            docs = self.load_and_process_documents(file_path, file_type="markdown")
                        elif file.endswith('.pdf'):
                            docs = self.load_and_process_documents(file_path, file_type="pdf")
                        elif file.endswith(('.html', '.htm')):
                            docs = self.load_and_process_documents(file_path, file_type="web")
                        else:
                            print(f"⚠️ Type de fichier non supporté: {file}")
                            continue
                        
                        # Ajouter les métadonnées de source
                        for doc in docs:
                            doc.metadata["source_file"] = relative_path
                            doc.metadata["source_folder"] = folder_path
                        
                        all_documents.extend(docs)
                        print(f"✅ {len(docs)} chunks ajoutés depuis {relative_path}")
                        
                    except Exception as e:
                        print(f"❌ Erreur lors du traitement de {file}: {e}")
                        continue
            
            print(f"📊 Total: {len(all_documents)} chunks chargés depuis {folder_path}")
            return all_documents
            
        except Exception as e:
            print(f"❌ Erreur lors du chargement du dossier {folder_path}: {e}")
            return []
    
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
            
            # Chargement et traitement des documents depuis le dossier 'documents'
            documents = self.load_all_documents_from_folder("./documents")
            
            if not documents:
                print("❌ Aucun document trouvé dans le dossier 'documents'")
                return
            
            self.populate_vector_store(documents)
            
            # Tests RAG avec des questions adaptées au contenu des documents
            questions = [
                "Qu'est-ce que le RAG (Retrieval Augmented Generation) ?",
                "Comment éviter les biais dans un système d'IA ?",
                "Quel est le rôle d'un Centre d'Excellence en IA ?"
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


# ============================================================================
# FONCTION PRINCIPALE - DÉMONSTRATION COMPLÈTE DU RAG
# ============================================================================

def main():
    """
    Fonction principale démontrant l'utilisation complète du système RAG.
    
    CONCEPTS PÉDAGOGIQUES DÉMONTRÉS :
    - Initialisation complète : Configuration de tous les composants
    - Pipeline RAG : Flux complet de traitement des documents
    - Gestion d'erreurs : Traitement robuste des cas d'échec
    - Interface utilisateur : Interaction claire et informative
    
    Cette fonction illustre :
    1. Configuration de l'environnement et des modèles
    2. Chargement et traitement de documents
    3. Stockage dans la base vectorielle
    4. Recherche sémantique et génération de réponses
    5. Affichage détaillé des résultats
    """
    
    # ========================================================================
    # ÉTAPE 1: INITIALISATION DU SYSTÈME RAG
    # ========================================================================
    print("🚀 Initialisation du système RAG...")
    
    # Création de l'instance de l'application RAG
    # CONCEPT : Instanciation de la classe principale
    rag_app = RAGApplication()
    
    # Configuration de l'environnement (clés API, etc.)
    # CONCEPT : Sécurité et configuration externalisée
    rag_app.setup_environment()
    
    # Initialisation des modèles (LLM + Embeddings)
    # CONCEPT : Modèles de langage et embeddings pour le traitement
    rag_app.initialize_models()
    
    # Configuration de la base vectorielle Qdrant
    # CONCEPT : Base de données spécialisée pour les vecteurs
    rag_app.setup_vector_store()
    
    print("✅ Système RAG initialisé avec succès!\n")
    
    # ========================================================================
    # ÉTAPE 2: CHARGEMENT ET TRAITEMENT DES DOCUMENTS
    # ========================================================================
    print("📚 Chargement des documents...")
    
    # Sources de documents à traiter
    # CONCEPT : Diversité des sources (web, fichiers locaux)
    documents_to_load = [
        # Document web (article technique)
        {
            "source": "https://lilianweng.github.io/posts/2023-06-23-agent/",
            "type": "web",
            "description": "Article sur les agents LLM"
        },
        # Document local Markdown (si disponible)
        {
            "source": "documents/rag_tutorial.md",
            "type": "markdown", 
            "description": "Tutoriel RAG local"
        },
        # Document local PDF (si disponible)
        {
            "source": "documents/ai_research.pdf",
            "type": "pdf",
            "description": "Recherche IA locale"
        }
    ]
    
    # Traitement de chaque document
    # CONCEPT : Pipeline de traitement pour chaque source
    for doc_info in documents_to_load:
        try:
            # Chargement et traitement du document
            # CONCEPT : Text splitting et préparation pour embeddings
            documents = rag_app.load_and_process_documents(
                source=doc_info["source"],
                file_type=doc_info["type"]
            )
            
            # Ajout dans la base vectorielle
            # CONCEPT : Génération d'embeddings et stockage
            rag_app.populate_vector_store(documents)
            
            print(f"✅ Document '{doc_info['description']}' traité avec succès")
            
        except Exception as e:
            # Gestion d'erreurs robuste
            # CONCEPT : Traitement des cas d'échec sans arrêter le processus
            print(f"⚠️ Erreur lors du traitement de '{doc_info['description']}': {e}")
            print("   Le document sera ignoré, le système continue...")
            continue
    
    print(f"\n📊 Documents chargés dans la base vectorielle")
    
    # ========================================================================
    # ÉTAPE 3: DÉMONSTRATION DES CAPACITÉS RAG
    # ========================================================================
    print("\n" + "="*80)
    print("🎯 DÉMONSTRATION DES CAPACITÉS RAG")
    print("="*80)
    
    # Questions de démonstration
    # CONCEPT : Diversité des requêtes pour tester le système
    demo_questions = [
        {
            "question": "Qu'est-ce qu'un agent LLM et comment fonctionne-t-il?",
            "description": "Question technique sur les agents LLM"
        },
        {
            "question": "Expliquez les concepts de RAG et d'embeddings vectoriels",
            "description": "Question pédagogique sur les concepts RAG"
        },
        {
            "question": "Quels sont les avantages et inconvénients des systèmes RAG?",
            "description": "Question d'analyse comparative"
        }
    ]
    
    # Traitement de chaque question
    # CONCEPT : Pipeline RAG complet pour chaque requête
    for i, q_info in enumerate(demo_questions, 1):
        print(f"\n🔍 Question #{i}: {q_info['description']}")
        print("-" * 60)
        
        try:
            # Appel du pipeline RAG complet
            # CONCEPT : Recherche + Génération en une seule méthode
            response, sources = rag_app.ask_question(q_info["question"])
            
            # Affichage de la réponse
            # CONCEPT : Interface utilisateur claire
            print(f"\n💡 RÉPONSE:")
            print(response)
            
        except Exception as e:
            # Gestion d'erreurs pour chaque question
            # CONCEPT : Robustesse du système
            print(f"❌ Erreur lors du traitement de la question: {e}")
            continue
    
    print("\n" + "="*80)
    print("✅ Démonstration RAG terminée!")
    print("="*80)


# ============================================================================
# POINT D'ENTRÉE ET GESTION D'ERREURS
# ============================================================================

if __name__ == "__main__":
    """
    Point d'entrée principal avec gestion d'erreurs complète.
    
    CONCEPTS PÉDAGOGIQUES :
    - Point d'entrée : Démarrage contrôlé de l'application
    - Gestion d'erreurs globale : Protection contre les crashs
    - Messages informatifs : Feedback utilisateur clair
    - Structure main : Bonne pratique Python
    
    Cette section assure :
    - Démarrage sécurisé de l'application
    - Gestion des erreurs non prévues
    - Messages d'erreur informatifs
    - Sortie propre en cas de problème
    """
    
    try:
        # Exécution de la fonction principale
        # CONCEPT : Point d'entrée standard Python
        main()
        
    except KeyboardInterrupt:
        # Gestion de l'interruption utilisateur (Ctrl+C)
        # CONCEPT : Sortie propre lors de l'interruption
        print("\n\n⚠️ Interruption utilisateur détectée")
        print("Arrêt propre de l'application...")
        
    except Exception as e:
        # Gestion des erreurs non prévues
        # CONCEPT : Robustesse et debugging
        print(f"\n❌ Erreur inattendue: {e}")
        print("Vérifiez votre configuration et vos clés API")
        print("Assurez-vous que tous les packages sont installés")
        
    finally:
        # Code de nettoyage (optionnel)
        # CONCEPT : Ressources et état final
        print("\n👋 Application RAG fermée")


# ============================================================================
# EXEMPLES D'UTILISATION ET BONNES PRATIQUES
# ============================================================================

"""
EXEMPLES D'UTILISATION PÉDAGOGIQUES
====================================

1. UTILISATION BASIQUE :
   ```python
   # Création et configuration
   rag_app = RAGApplication()
   rag_app.setup_environment()
   rag_app.initialize_models()
   rag_app.setup_vector_store()
   
   # Chargement d'un document
   docs = rag_app.load_and_process_documents("mon_document.pdf", "pdf")
   rag_app.populate_vector_store(docs)
   
   # Question-réponse
   response, sources = rag_app.ask_question("Ma question?")
   ```

2. UTILISATION AVANCÉE :
   ```python
   # Configuration personnalisée
   rag_app = RAGApplication()
   rag_app.collection_name = "mes_documents"  # Collection personnalisée
   
   # Chargement de multiples sources
   sources = [
       ("https://example.com/article", "web"),
       ("documents/rapport.pdf", "pdf"),
       ("data/notes.md", "markdown")
   ]
   
   for source, file_type in sources:
       docs = rag_app.load_and_process_documents(source, file_type)
       rag_app.populate_vector_store(docs)
   ```

3. RECHERCHE PERSONNALISÉE :
   ```python
   # Recherche avec paramètres personnalisés
   docs = rag_app.search_documents("ma requête", k=5)  # Plus de résultats
   
   # Génération de réponse personnalisée
   response = rag_app.generate_response("ma question", docs)
   ```

BONNES PRATIQUES PÉDAGOGIQUES
==============================

1. SÉCURITÉ :
   - ✅ Utilisez des variables d'environnement pour les clés API
   - ✅ Ne committez jamais de clés API dans le code
   - ✅ Utilisez getpass() pour les entrées sensibles

2. GESTION D'ERREURS :
   - ✅ Gérez les exceptions pour chaque opération critique
   - ✅ Fournissez des messages d'erreur informatifs
   - ✅ Continuez le traitement même si un document échoue

3. OPTIMISATION :
   - ✅ Choisissez la taille de chunk appropriée (1000 caractères)
   - ✅ Utilisez un chevauchement pour maintenir le contexte
   - ✅ Sélectionnez le bon modèle d'embedding selon vos besoins

4. MONITORING :
   - ✅ Affichez les métadonnées des documents utilisés
   - ✅ Suivez le nombre de documents dans la base
   - ✅ Vérifiez la qualité des réponses générées

5. ARCHITECTURE :
   - ✅ Utilisez une architecture modulaire (classe RAGApplication)
   - ✅ Séparez les responsabilités (chargement, recherche, génération)
   - ✅ Rendez le code réutilisable et extensible

CONCEPTS CLÉS À COMPRENDRE
==========================

1. RAG (Retrieval-Augmented Generation) :
   - Combine recherche sémantique et génération de texte
   - Améliore la précision des réponses des LLM
   - Permet l'utilisation de connaissances externes

2. Embeddings Vectoriels :
   - Représentations numériques du texte
   - Permettent la recherche par similarité sémantique
   - Base de la recherche dans les systèmes RAG

3. Base de Données Vectorielle :
   - Stockage optimisé pour les embeddings
   - Recherche rapide par similarité cosinus
   - Qdrant est une solution performante et open-source

4. Text Splitting :
   - Découpage intelligent des documents
   - Optimisation pour les embeddings
   - Maintien du contexte avec chevauchement

5. Prompt Engineering :
   - Conception de prompts pour guider les LLM
   - Intégration du contexte dans les prompts
   - Optimisation pour des réponses de qualité

EXTENSIONS POSSIBLES
====================

1. Interface Web :
   - Ajout d'une interface Flask/FastAPI
   - Interface utilisateur avec Streamlit
   - API REST pour l'intégration

2. Fonctionnalités Avancées :
   - Support de multiples langues
   - Recherche hybride (sémantique + mots-clés)
   - Filtrage par métadonnées
   - Cache des embeddings

3. Monitoring et Analytics :
   - Logs détaillés des requêtes
   - Métriques de performance
   - Dashboard de monitoring

4. Intégrations :
   - Connexion à des bases de données existantes
   - Intégration avec des systèmes de fichiers
   - Webhooks pour les mises à jour automatiques

Ce projet sert de base solide pour comprendre et étendre les systèmes RAG.
Chaque composant est modulaire et peut être adapté selon les besoins spécifiques.
"""