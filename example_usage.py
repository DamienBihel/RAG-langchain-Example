"""
Exemples d'utilisation de l'application RAG avec différents formats de documents.

Ce script montre comment utiliser l'application RAG pour charger :
- Des pages web
- Des fichiers PDF
- Des fichiers Markdown
"""

from main import RAGApplication
from pathlib import Path


def test_web_document():
    """Test avec un document web."""
    print("\n" + "="*60)
    print("🌐 TEST AVEC DOCUMENT WEB")
    print("="*60)
    
    app = RAGApplication()
    app.setup_environment()
    app.initialize_models()
    app.setup_vector_store()
    
    # Chargement d'un document web
    url = "https://python.langchain.com/docs/tutorials/rag/"
    documents = app.load_and_process_documents(url, file_type="web")
    app.populate_vector_store(documents)
    
    # Test de questions
    response, sources = app.ask_question("What is RAG?")
    print(f"\n💬 Réponse: {response}")
    print(f"📚 Sources: {len(sources)} documents utilisés")


def test_pdf_document():
    """Test avec un document PDF."""
    print("\n" + "="*60)
    print("📄 TEST AVEC DOCUMENT PDF")
    print("="*60)
    
    app = RAGApplication()
    app.setup_environment()
    app.initialize_models()
    app.setup_vector_store()
    
    # Assurez-vous d'avoir un fichier PDF dans le dossier documents
    pdf_path = "./documents/example.pdf"
    
    if Path(pdf_path).exists():
        documents = app.load_and_process_documents(pdf_path, file_type="pdf")
        app.populate_vector_store(documents)
        
        # Test de questions
        response, sources = app.ask_question("What is the main topic of this document?")
        print(f"\n💬 Réponse: {response}")
        print(f"📚 Sources: {len(sources)} documents utilisés")
    else:
        print(f"⚠️  Fichier PDF non trouvé : {pdf_path}")
        print("   Créez un dossier 'documents' et ajoutez-y un fichier PDF pour tester")


def test_markdown_document():
    """Test avec un document Markdown."""
    print("\n" + "="*60)
    print("📝 TEST AVEC DOCUMENT MARKDOWN")
    print("="*60)
    
    app = RAGApplication()
    app.setup_environment()
    app.initialize_models()
    app.setup_vector_store()
    
    # Utilisation du README.md du projet (s'il existe)
    markdown_path = "./README.md"
    
    # Si README.md n'existe pas, utilisons CLAUDE.md
    if not Path(markdown_path).exists():
        markdown_path = "./CLAUDE.md"
    
    if Path(markdown_path).exists():
        documents = app.load_and_process_documents(markdown_path, file_type="markdown")
        app.populate_vector_store(documents)
        
        # Test de questions
        response, sources = app.ask_question("What is this project about?")
        print(f"\n💬 Réponse: {response}")
        print(f"📚 Sources: {len(sources)} documents utilisés")
    else:
        print(f"⚠️  Fichier Markdown non trouvé : {markdown_path}")


def test_mixed_documents():
    """Test avec plusieurs types de documents."""
    print("\n" + "="*60)
    print("🔀 TEST AVEC DOCUMENTS MIXTES")
    print("="*60)
    
    app = RAGApplication()
    app.setup_environment()
    app.initialize_models()
    app.setup_vector_store()
    
    # Chargement de différents types de documents
    sources = [
        ("https://python.langchain.com/docs/tutorials/rag/", "web"),
        ("./CLAUDE.md", "markdown"),
        # Ajoutez vos propres fichiers PDF ici
        # ("./documents/guide.pdf", "pdf"),
    ]
    
    all_documents = []
    for source, file_type in sources:
        if file_type != "pdf" or Path(source).exists():
            try:
                documents = app.load_and_process_documents(source, file_type=file_type)
                all_documents.extend(documents)
                print(f"✅ Chargé: {source}")
            except Exception as e:
                print(f"❌ Erreur lors du chargement de {source}: {e}")
    
    if all_documents:
        app.populate_vector_store(all_documents)
        
        # Test de questions sur l'ensemble des documents
        questions = [
            "Quelle approche technique ET organisationnelle pour déployer l'IA ?",
            "Comment éviter les biais dans un système RAG ?",
            "Quel lien entre Centre d'Excellence et implémentation technique ?"
        ]
        
        for question in questions:
            response, sources = app.ask_question(question)
            print(f"\n🤔 Question: {question}")
            print(f"💬 Réponse: {response}")
            print(f"📚 Sources: {len(sources)} documents utilisés")


def main():
    """Point d'entrée principal pour les tests."""
    print("🚀 Démonstration de l'application RAG multi-format")
    print("=" * 50)
    
    # Décommentez les tests que vous voulez exécuter
    
    # Test avec document web
    test_web_document()
    
    # Test avec document PDF (nécessite un fichier PDF)
    # test_pdf_document()
    
    # Test avec document Markdown
    test_markdown_document()
    
    # Test avec documents mixtes
    # test_mixed_documents()
    
    print("\n✅ Tests terminés !")


if __name__ == "__main__":
    main()