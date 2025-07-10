# RAG-langchain-Example

**Statut du projet : PUBLIC**

---

## Présentation

Ce projet propose une application RAG (Retrieval-Augmented Generation) complète, modulaire et pédagogique, basée sur LangChain et Qdrant. Il permet la recherche sémantique et la génération de réponses à partir de documents web, PDF, Markdown, etc.

---

## Configuration rapide

### 1. Cloner le dépôt
```sh
git clone https://github.com/DamienBihel/RAG-langchain-Example.git
cd RAG-langchain-Example
```

### 2. Installer les dépendances
```sh
pip install -r requirements.txt
```

### 3. Configurer les variables d'environnement

- Copiez le fichier `.env.template` en `.env` :
  ```sh
  cp .env.template .env
  ```
- Ouvrez `.env` et renseignez votre clé OpenAI :
  ```env
  OPENAI_API_KEY=sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
  ```
- (Optionnel) Modifiez le `USER_AGENT` si besoin pour vos requêtes web.

**⚠️ Ne partagez jamais votre fichier `.env` !**

### 4. Lancer l'application principale
```sh
python main.py
```

---

## Sécurité

- Le fichier `.env` ne doit jamais être versionné ni partagé (il est dans le `.gitignore`).
- Utilisez le template `.env.template` pour partager la structure sans exposer de secrets.
- Les clés API sont obligatoires pour utiliser les modèles OpenAI (LLM et embeddings).

---

## Fonctionnalités principales
- Chargement de documents web, PDF, Markdown, texte
- Découpage intelligent (chunking) avec chevauchement
- Génération d'embeddings vectoriels (OpenAI)
- Stockage et recherche sémantique (Qdrant)
- Génération de réponses contextuelles (LLM)
- Affichage des sources utilisées pour la réponse
- Architecture modulaire et pédagogique

---

## Exemples d'utilisation
Voir le fichier `example_usage.py` et la section "Exemples d'utilisation" dans le code source.

---

## Pour aller plus loin
- Voir `TEST_REPORT.md` pour le rapport de tests
- Voir `QUICK_START_GUIDE.md` pour un guide pas à pas
- Voir `RAG_MODES_ANALYSIS.md` pour l'analyse des stratégies RAG

---

## Licence
Projet open-source, usage libre à des fins pédagogiques, de recherche ou de démonstration. 