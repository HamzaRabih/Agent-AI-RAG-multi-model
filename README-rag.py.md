# README - rag.py

## Resume

`rag.py` est le module principal du projet. Il expose une interface Streamlit qui combine:

- RAG texte sur PDF (embeddings OpenAI + Chroma)
- Retrieval image (OpenCLIP + ChromaDB)
- Reponse LLM finale basee sur contexte texte et, si possible, sur les images retrouvees

## Fonctionnement

### 1) Ingestion PDF

- Upload de un ou plusieurs PDF dans la sidebar
- Extraction de texte avec `PyPDF2`
- Decoupage en chunks avec `RecursiveCharacterTextSplitter`
- Vectorisation avec `OpenAIEmbeddings`
- Indexation dans Chroma

### 2) Ingestion images

- Upload de fichiers image (`.jpg`, `.jpeg`, `.png`, `.webp`)
- Sauvegarde locale dans `images-store-vdb/uploaded_images/`
- Embedding image via `OpenCLIPEmbeddingFunction`
- Stockage dans la collection ChromaDB `images_collection`

### 3) Question utilisateur

- Retrieval texte via retriever LangChain
- Retrieval image via `image_collection.query(...)`
- Si 2 images sont retrouvees:
  - encodage base64
  - envoi a une chaine multimodale (`vision_chain`)
- Sinon:
  - fallback vers reponse texte classique

## Prerequis

- Python `>=3.11`
- `OPENAI_API_KEY` configure dans `.env`
- Dependances installees via `pip install -e .` ou `uv sync`

## Lancer

```powershell
streamlit run rag.py
```

## Utilisation

1. Ouvrir la page Streamlit.
2. Charger les PDF.
3. Charger les images (optionnel, mais recommande pour mode multimodal).
4. Cliquer sur `Submit`.
5. Poser une question.

## Variables de session Streamlit

- `st.session_state.retriever`: retriever texte
- `st.session_state.image_collection`: collection ChromaDB image

## Erreurs frequentes

### AttributeError sur retriever

Selon version LangChain, utiliser `invoke(...)` plutot que `get_relevant_documents(...)`.

### Aucune image retournee

Verifier:

- formats supportes
- presence d'images uploades
- indexation terminee apres `Submit`

### Reponse non pertinente

Verifier:

- qualite du texte PDF extrait
- taille des chunks
- formulation de la question

## Limites

- Le pipeline multimodal courant attend idealement 2 images.
- Pas de citation explicite des pages PDF dans la reponse.
- Pas de persistence fine du retriever texte entre redemarrages.

## Evolutions recommandees

1. Ajouter un store Chroma persistant pour la partie texte.
2. Ajouter la citation des sources (page/chunk).
3. Ajouter une strategy de reranking sur les chunks.
4. Ajouter un mode conversation multi-turn avec memoire.
