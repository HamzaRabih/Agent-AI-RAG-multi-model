# README - rag-multimodel

## Resume

`rag-multimodel.ipynb` est un notebook d'experimentation pour le retrieval image et le question answering multimodal.

Il demontre:

- creation d'une base vectorielle image avec ChromaDB
- embeddings OpenCLIP
- recherche d'images par requete texte
- utilisation d'un prompt vision avec images encodees en base64

## Pipeline du notebook

1. Charger les images depuis `vehicules/`.
2. Configurer ChromaDB (`images-store-vdb`).
3. Indexer les URIs image dans `images_collection`.
4. Lancer des requetes de similarite (ex: `white cars`, `red cars`).
5. Visualiser les images retrouvees.
6. Construire une chaine vision LangChain.
7. Poser une question et generer une reponse Markdown.

## Structure locale attendue

```text
rag-multimodel/
|-- rag-multimodel.ipynb
|-- vehicules/
|-- images-store-vdb/
|-- overview-a.svg
`-- overview-b.svg
```

## Dependances cles

- `chromadb`
- `open-clip-torch`
- `Pillow`
- `langchain-openai`
- `python-dotenv`

## Bonnes pratiques

- Vider ou versionner correctement `images-store-vdb` selon le besoin.
- Conserver des noms de fichiers image stables pour la reproductibilite.
- Reexecuter les cellules d'indexation en cas d'ajout de nouvelles images.

## Limitations

- Notebook centree demo, pas une application deployable.
- Peu de gestion d'erreurs (fichiers absents, formats invalides).
- Prompt vision non evalue quantitativement.

## Lien avec le projet principal

Les concepts de ce notebook sont integres dans `rag.py`, qui combine retrieval texte + retrieval image dans une interface Streamlit unique.
