# Agentic AI - TP3 RAG

Application RAG (Retrieval-Augmented Generation) avec interface Streamlit.

Le projet permet de charger un ou plusieurs PDF (ex: documentation ERP, structure de tables BDD), de créer un index vectoriel local, puis de poser des questions en langage naturel. Le LLM répond en se basant sur le contexte retrouvé dans les documents.

## Apercu

Dans votre cas d'usage, vous avez fourni un document décrivant la structure de la base ERP, puis le chatbot a pu proposer une requete SQL pertinente a partir de ce contexte.

## Capture d'ecran

![Interface Streamlit - Chat RAG](img/image.png)

## Fonctionnalites

- Import de plusieurs PDF depuis l'interface Streamlit.
- Extraction du texte via PyPDF2.
- Decoupage en chunks avec `RecursiveCharacterTextSplitter`.
- Vectorisation avec `OpenAIEmbeddings`.
- Stockage vectoriel local via Chroma.
- Recuperation des passages les plus pertinents (`retriever`) pour enrichir le prompt.
- Generation de reponses avec `ChatOpenAI` (`gpt-4o`).

## Architecture (resume)

1. Upload des documents PDF.
2. Extraction du texte.
3. Decoupage en morceaux.
4. Creation de l'index vectoriel.
5. Recherche des chunks pertinents selon la question.
6. Construction du prompt avec contexte.
7. Generation de la reponse par le LLM.

## Prerequis

- Python 3.11+
- Une cle API OpenAI (variable d'environnement `OPENAI_API_KEY`)
- Environnement virtuel recommande

## Installation

```bash
# Cloner le projet
# git clone <url-du-repo>
# cd TP3-AI-Agent-RAG

# Creer et activer un environnement virtuel (Windows)
python -m venv .venv
.venv\Scripts\activate

# Installer les dependances
pip install -e .
```

Alternative avec uv:

```bash
uv sync
```

## Configuration

Creer un fichier `.env` a la racine du projet:

```env
OPENAI_API_KEY=votre_cle_openai
```

## Lancement

```bash
streamlit run rag.py
```

Puis ouvrir l'URL fournie par Streamlit (souvent `http://localhost:8501`).

## Guide d'utilisation

1. Ouvrir l'application.
2. Dans la sidebar, charger un ou plusieurs PDF.
3. Cliquer sur `Submit` pour indexer les documents.
4. Poser une question dans le champ de chat.
5. Lire la reponse generee a partir du contexte recupere.

## Structure du projet

```text
TP3-AI-Agent-RAG/
|- rag.py
|- pyproject.toml
|- RAGV2.ipynb
|- img/
|  |- image.png
|- pdfs/
|- store/
```

## Limites actuelles

- Le bouton `Submit` doit etre execute avant la premiere question (sinon aucun retriever en session).
- La qualite des reponses depend de la qualite du texte extrait des PDF.
- Pas encore de memoire conversationnelle multi-turn avancee.

## Pistes d'amelioration

- Ajouter une gestion d'erreur explicite si aucun document n'a ete charge.
- Afficher les sources exactes (chunk/page) utilisees dans la reponse.
- Sauvegarder/recharger automatiquement l'index Chroma entre sessions.
- Ajouter un prompt specialise "Text-to-SQL" pour fiabiliser les requetes SQL ERP.

## Stack technique

- Streamlit
- LangChain
- ChromaDB
- OpenAI API (Embeddings + Chat)
- PyPDF2

## Auteur

RABIH Hamza - Agentic AI / RAG
