# README - RAGV2

## Resume

`RAGV2.ipynb` est un notebook focalise sur le pipeline RAG texte (PDF -> chunks -> embeddings -> retrieval -> reponse LLM).

Ce notebook sert de base pour la partie textuelle de l'application `rag.py`.

## Objectifs pedagogiques

- Comprendre le flux RAG de bout en bout.
- Tester les parametres de chunking.
- Evaluer l'impact du retrieval sur la qualite des reponses.

## Pipeline type

1. Charger les documents PDF.
2. Extraire le texte.
3. Decouper en chunks.
4. Creer des embeddings texte.
5. Indexer dans Chroma.
6. Recuperer les passages pertinents.
7. Construire le prompt et appeler le LLM.

## Parametres importants a ajuster

- `chunk_size`
- `chunk_overlap`
- nombre de documents recuperes (`k`)
- prompt system/utilisateur

## Bonnes pratiques

- Verifier la qualite de l'extraction PDF avant indexation.
- Eviter des chunks trop grands (bruit) ou trop petits (perte de contexte).
- Comparer les reponses avec et sans retrieval pour mesurer la valeur RAG.

## Limitations

- Notebook d'exploration, pas un service de production.
- Pas de suivi de performance automatise.
- Pas de benchmark formel sur jeu de test.

## Lien avec le projet principal

Les mecanismes de ce notebook ont ete repris dans `rag.py` pour la partie retrieval texte, puis combines avec la partie image issue de `rag-multimodel.ipynb`.
