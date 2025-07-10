"""
Script pour afficher les documents stockés dans le RAG.

Ce script montre comment utiliser les nouvelles méthodes pour :
- Afficher tous les documents stockés
- Rechercher et afficher des documents spécifiques
- Examiner les métadonnées et le contenu
"""

from main import RAGApplication


def demonstrate_document_display():
    """Démontre l'affichage des documents dans le RAG."""
    
    print("🚀 Démonstration de l'affichage des documents RAG")
    print("=" * 80)
    
    # Initialisation de l'application
    app = RAGApplication()
    app.setup_environment()
    app.initialize_models()
    app.setup_vector_store()
    
    # Chargement des documents d'exemple
    print("\n📥 Chargement des documents d'exemple...")
    url = "https://lilianweng.github.io/posts/2023-06-23-agent/"
    documents = app.load_and_process_documents(url)
    app.populate_vector_store(documents)
    
    # === AFFICHAGE DES SOURCES ===
    print("\n\n" + "="*80)
    print("LISTING DES SOURCES CHARGÉES")
    print("="*80)
    app.display_sources()
    
    # === DÉMONSTRATION 1: Afficher tous les documents ===
    print("\n\n" + "="*80)
    print("DÉMONSTRATION 1: Afficher tous les documents stockés")
    print("="*80)
    
    # Affichage avec contenu limité
    app.display_documents(limit=5, show_content=True, max_content_length=300)
    
    # === DÉMONSTRATION 2: Récupérer les documents sans affichage ===
    print("\n\n" + "="*80)
    print("DÉMONSTRATION 2: Récupérer les documents pour traitement")
    print("="*80)
    
    all_docs = app.get_all_documents(limit=20)
    print(f"\n📊 Statistiques des documents:")
    print(f"   • Nombre total de documents: {len(all_docs)}")
    
    # Analyse des tailles
    sizes = [len(doc['content']) for doc in all_docs]
    if sizes:
        print(f"   • Taille moyenne: {sum(sizes) / len(sizes):.0f} caractères")
        print(f"   • Taille min: {min(sizes)} caractères")
        print(f"   • Taille max: {max(sizes)} caractères")
    
    # === DÉMONSTRATION 3: Recherche et affichage ===
    print("\n\n" + "="*80)
    print("DÉMONSTRATION 3: Recherche sémantique avec scores")
    print("="*80)
    
    queries = [
        "What is an agent in AI?",
        "memory components",
        "planning and reasoning"
    ]
    
    for query in queries:
        app.search_and_display(query, k=3)
        print("\n" + "-"*60)
    
    # === DÉMONSTRATION 4: Affichage détaillé d'un document spécifique ===
    print("\n\n" + "="*80)
    print("DÉMONSTRATION 4: Examen détaillé des métadonnées")
    print("="*80)
    
    # Récupérer quelques documents avec leurs métadonnées complètes
    docs_with_metadata = app.get_all_documents(limit=3)
    
    for i, doc in enumerate(docs_with_metadata, 1):
        print(f"\n📄 Document {i} - Analyse détaillée:")
        print(f"   ID: {doc['id']}")
        print(f"   Métadonnées disponibles:")
        for key, value in doc['metadata'].items():
            print(f"      • {key}: {value}")


def interactive_exploration():
    """Mode interactif pour explorer les documents."""
    
    print("\n\n" + "="*80)
    print("🔍 MODE EXPLORATION INTERACTIVE")
    print("="*80)
    
    app = RAGApplication()
    app.setup_environment()
    app.initialize_models()
    app.setup_vector_store()
    
    # Charger des documents
    url = "https://lilianweng.github.io/posts/2023-06-23-agent/"
    documents = app.load_and_process_documents(url)
    app.populate_vector_store(documents)
    
    while True:
        print("\n\nOptions disponibles:")
        print("1. Afficher tous les documents")
        print("2. Rechercher des documents")
        print("3. Statistiques de la base")
        print("4. Afficher les sources")
        print("5. Charger un nouveau document")
        print("6. Quitter")
        
        choice = input("\nVotre choix (1-6): ")
        
        if choice == "1":
            limit = input("Nombre de documents à afficher (par défaut 10): ")
            limit = int(limit) if limit else 10
            app.display_documents(limit=limit)
            
        elif choice == "2":
            query = input("Entrez votre recherche: ")
            k = input("Nombre de résultats (par défaut 5): ")
            k = int(k) if k else 5
            app.search_and_display(query, k=k)
            
        elif choice == "3":
            docs = app.get_all_documents(limit=1000)
            print(f"\n📊 Statistiques de la base vectorielle:")
            print(f"   • Documents stockés: {len(docs)}")
            if docs:
                sizes = [len(doc['content']) for doc in docs]
                print(f"   • Taille totale: {sum(sizes):,} caractères")
                print(f"   • Taille moyenne: {sum(sizes) / len(sizes):.0f} caractères")
        
        elif choice == "4":
            app.display_sources()
            
        elif choice == "5":
            source = input("Entrez l'URL ou le chemin du fichier: ")
            try:
                documents = app.load_and_process_documents(source)
                app.populate_vector_store(documents)
                print(f"✅ Document chargé avec succès!")
            except Exception as e:
                print(f"❌ Erreur lors du chargement: {e}")
                
        elif choice == "6":
            print("\n👋 Au revoir!")
            break
        
        else:
            print("❌ Choix invalide")


if __name__ == "__main__":
    # Lancer la démonstration automatique
    demonstrate_document_display()
    
    # Optionnel: lancer le mode interactif
    response = input("\n\nVoulez-vous explorer les documents en mode interactif? (o/n): ")
    if response.lower() == 'o':
        interactive_exploration()