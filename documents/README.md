# Organisation des Documents

Ce dossier contient les documents à traiter par l'application RAG.

## Structure des dossiers

```
documents/
├── pdf/          # Placez vos fichiers PDF ici
├── markdown/     # Placez vos fichiers Markdown (.md) ici
├── web/          # (Optionnel) Pour sauvegarder des pages web téléchargées
└── README.md     # Ce fichier
```

## Comment ajouter des documents

### Fichiers PDF
Placez vos fichiers PDF dans le dossier `pdf/` :
```bash
cp mon-document.pdf ./documents/pdf/
```

### Fichiers Markdown
Placez vos fichiers Markdown dans le dossier `markdown/` :
```bash
cp mon-document.md ./documents/markdown/
```

## Exemples d'utilisation

```python
from main import RAGApplication

app = RAGApplication()
app.setup_environment()
app.initialize_models()
app.setup_vector_store()

# Charger un PDF
documents = app.load_and_process_documents("./documents/pdf/guide.pdf")

# Charger un Markdown
documents = app.load_and_process_documents("./documents/markdown/notes.md")

# Charger tous les documents d'un dossier
import glob

# Tous les PDF
for pdf_file in glob.glob("./documents/pdf/*.pdf"):
    docs = app.load_and_process_documents(pdf_file)
    app.populate_vector_store(docs)
```

## Formats supportés

- **PDF** : `.pdf` - Documents PDF standard
- **Markdown** : `.md`, `.markdown` - Fichiers Markdown
- **Web** : URLs commençant par `http://` ou `https://`

## Limitations

- Taille maximale recommandée par fichier : 50 MB
- Les PDF protégés par mot de passe ne sont pas supportés
- Les PDF scannés (images) nécessitent OCR (non inclus)