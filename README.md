# Application RAG avec LangChain et Qdrant

## 🚀 Description

Application RAG (Retrieval-Augmented Generation) complète et modulaire utilisant LangChain et Qdrant pour la recherche sémantique et la génération de réponses basées sur des documents.

## ✨ Fonctionnalités

- **Chargement de documents web** avec BeautifulSoup
- **Découpage intelligent** en chunks avec chevauchement
- **Génération d'embeddings** avec OpenAI text-embedding-3-small
- **Stockage vectoriel** dans Qdrant
- **Recherche sémantique** par similarité cosinus
- **Génération de réponses** avec GPT-4o-mini
- **Architecture modulaire** et extensible

## 🛠️ Technologies

- **LangChain** : Framework pour applications LLM
- **OpenAI** : Modèles GPT-4o-mini et text-embedding-3-small
- **Qdrant** : Base de données vectorielle
- **BeautifulSoup** : Parsing HTML
- **Python 3.10+** : Langage principal

## 📋 Prérequis

- Python 3.10 ou supérieur
- Clé API OpenAI
- Environnement virtuel Python

## 🔧 Installation

1. **Cloner le dépôt**
   ```bash
   git clone <url-du-depot>
   cd RAG-langchain-Exampke
   ```

2. **Créer l'environnement virtuel**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # Linux/Mac
   # ou
   .venv\Scripts\activate     # Windows
   ```

3. **Installer les dépendances**
   ```bash
   pip install -qU "langchain[openai]"
   pip install -qU langchain-qdrant
   pip install python-dotenv
   pip install beautifulsoup4
   pip install langchain-community
   ```

4. **Configurer les variables d'environnement**
   
   Créer un fichier `.env` à la racine du projet :
   ```env
   OPENAI_API_KEY=sk-votre-cle-api-openai
   ```

## 🚀 Utilisation

### Exécution simple

```bash
python main.py
```

### Utilisation programmatique

```python
from main import RAGApplication

# Créer l'application
app = RAGApplication()

# Configuration
app.setup_environment()
app.initialize_models()
app.setup_vector_store()

# Charger des documents
documents = app.load_and_process_documents("https://example.com/article")
app.populate_vector_store(documents)

# Poser une question
response, sources = app.ask_question("Quelle est la question ?")
print(response)
```

## 📁 Structure du projet

```
RAG-langchain-Exampke/
├── main.py              # Application RAG principale
├── .env                 # Variables d'environnement (non versionné)
├── .gitignore          # Fichiers ignorés par Git
├── README.md           # Documentation du projet
└── requirements.txt    # Dépendances Python
```

## 🔒 Sécurité

- Le fichier `.env` est ignoré par Git pour protéger les clés API
- Les clés API ne sont jamais affichées dans les logs
- Utilisation d'environnements virtuels pour l'isolation

## 🧪 Tests

L'application inclut une démonstration complète qui teste :

1. **Chargement de documents** depuis une URL
2. **Découpage en chunks** (63 chunks pour 43k caractères)
3. **Stockage vectoriel** dans Qdrant
4. **Recherche sémantique** avec différentes questions
5. **Génération de réponses** basées sur le contexte

## 📊 Performance

- **Embeddings** : text-embedding-3-small (1536 dimensions)
- **LLM** : GPT-4o-mini avec température 0.1
- **Base vectorielle** : Qdrant avec distance COSINE
- **Chunks** : 1000 caractères avec chevauchement de 200

## 🔧 Configuration avancée

### Modifier les paramètres de découpage

```python
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,      # Taille des chunks
    chunk_overlap=200,     # Chevauchement
    add_start_index=True,  # Suivi de position
)
```

### Changer le modèle d'embedding

```python
self.embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
```

### Utiliser Qdrant en production

```python
self.client = QdrantClient("localhost", port=6333)
```

## 🤝 Contribution

1. Fork le projet
2. Créer une branche feature (`git checkout -b feature/AmazingFeature`)
3. Commit les changements (`git commit -m 'Add AmazingFeature'`)
4. Push vers la branche (`git push origin feature/AmazingFeature`)
5. Ouvrir une Pull Request

## 📝 Licence

Ce projet est sous licence MIT. Voir le fichier `LICENSE` pour plus de détails.

## 🆘 Support

Pour toute question ou problème :

1. Vérifier la documentation
2. Consulter les issues existantes
3. Créer une nouvelle issue avec les détails du problème

## 🔄 Mise à jour

Pour mettre à jour les dépendances :

```bash
pip install --upgrade langchain langchain-openai langchain-qdrant
```

---

**Note** : Ce projet est configuré pour être privé par défaut. Assurez-vous de configurer la visibilité appropriée sur votre plateforme Git. 