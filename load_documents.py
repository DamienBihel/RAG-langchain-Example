#!/usr/bin/env python3
"""
Script pour charger des documents dans le RAG depuis différentes sources.

Usage:
    python load_documents.py [source1] [source2] ...
    
Exemples:
    python load_documents.py https://example.com/doc.html
    python load_documents.py ./documents/guide.pdf ./docs/README.md
    python load_documents.py  # Mode interactif
"""

import sys
from pathlib import Path
from main import RAGApplication


def load_multiple_sources(app, sources):
    """Charge plusieurs sources dans l'application RAG."""
    
    print(f"\n📥 Chargement de {len(sources)} source(s)...")
    print("=" * 60)
    
    success_count = 0
    
    for source in sources:
        print(f"\n🔄 Traitement de: {source}")
        try:
            documents = app.load_and_process_documents(source)
            app.populate_vector_store(documents)
            success_count += 1
            print(f"✅ {source} chargé avec succès")
        except Exception as e:
            print(f"❌ Erreur pour {source}: {e}")
    
    print(f"\n📊 Résultat: {success_count}/{len(sources)} sources chargées")
    
    # Afficher le résumé des sources
    if success_count > 0:
        app.display_sources()


def interactive_mode(app):
    """Mode interactif pour charger des documents."""
    
    print("\n🎯 MODE CHARGEMENT INTERACTIF")
    print("=" * 60)
    print("Entrez les sources une par une (tapez 'done' pour terminer)")
    
    sources = []
    while True:
        source = input("\n📍 Source (URL ou fichier): ").strip()
        
        if source.lower() == 'done':
            break
        
        if source:
            sources.append(source)
            print(f"   → Ajouté: {source}")
    
    if sources:
        load_multiple_sources(app, sources)
    else:
        print("Aucune source fournie.")


def main():
    """Point d'entrée principal."""
    
    print("🚀 Chargeur de documents RAG")
    print("=" * 60)
    
    # Initialisation
    app = RAGApplication()
    app.setup_environment()
    app.initialize_models()
    app.setup_vector_store()
    
    # Récupérer les sources depuis les arguments
    sources = sys.argv[1:] if len(sys.argv) > 1 else []
    
    if sources:
        # Mode batch avec arguments
        load_multiple_sources(app, sources)
    else:
        # Mode interactif
        interactive_mode(app)
    
    # Options post-chargement
    while True:
        print("\n\n📋 OPTIONS POST-CHARGEMENT:")
        print("1. Afficher le résumé des sources")
        print("2. Rechercher dans les documents")
        print("3. Afficher des documents")
        print("4. Charger d'autres documents")
        print("5. Quitter")
        
        choice = input("\nVotre choix (1-5): ")
        
        if choice == "1":
            app.display_sources()
            
        elif choice == "2":
            query = input("Recherche: ")
            app.search_and_display(query)
            
        elif choice == "3":
            app.display_documents(limit=10)
            
        elif choice == "4":
            interactive_mode(app)
            
        elif choice == "5":
            print("\n👋 Au revoir!")
            break
        
        else:
            print("❌ Choix invalide")


if __name__ == "__main__":
    main()