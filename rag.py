import streamlit as st
from PyPDF2 import PdfReader
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
# chromadb pour créer une base de données vectorielle
import chromadb
# OpenCLIPEmbeddingFunction pour générer les embeddings des images
from chromadb.utils.embedding_functions import OpenCLIPEmbeddingFunction
# ImageLoader pour charger et traiter les images dans ChromaDB
from chromadb.utils.data_loaders import ImageLoader
from pathlib import Path  # Gérer les chemins de fichiers de façon portable
from PIL import Image
import base64

load_dotenv(override=True)
from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
)

prompt_template = """
Answer the following question based only on the provided context:
<context>
    {context}
</context>
<question>
    {input}
</question>
"""

llm = ChatOpenAI(model="gpt-4o", temperature=0)
parser = StrOutputParser()

vision_system_message = """
Answer using the provided text context and images.
Reference visible elements from the images when relevant.
If context is insufficient, say what is missing.
Use Markdown in the answer.
"""

image_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "{system_message}"),
        (
            "user",
            [
                {"type": "text", "text": "Question: {user_question}\n\nContext: {text_context}"},
                {"type": "image_url", "image_url": "data:image/jpeg;base64,{image_data1}"},
                {"type": "image_url", "image_url": "data:image/jpeg;base64,{image_data2}"},
            ],
        ),
    ]
)
vision_chain = image_prompt | llm | parser


def main():
    st.set_page_config(page_title="RAG", layout="wide")
    st.subheader("Retrieval Augmented generation", divider="blue")

    with st.sidebar:
        st.sidebar.title("Data loader")
        #st.image("rag.png")
        pdf_docs = st.file_uploader(label="Load your pdfs", accept_multiple_files=True)
        images = st.file_uploader(label="Load your Images", accept_multiple_files=True)

        if st.button("Submit"):
            with st.spinner("Loading"):
                content = ""

                # Process PDF documents
                if not pdf_docs:
                    st.warning("Veuillez charger au moins un PDF.")
                    return

                for pdf in pdf_docs:
                    reader = PdfReader(pdf)
                    for page in reader.pages:
                        content += page.extract_text() or ""
   
                if not content.strip():
                    st.warning("Aucun texte exploitable n'a ete extrait des PDF charges.")
                    return

                splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
                    chunk_size=512, chunk_overlap=16
                )

                chunks = splitter.split_text(content)

                embedding_model = OpenAIEmbeddings()
                vector_store = Chroma.from_texts(
                    chunks,
                    embedding_model,
                    collection_name="data_collection",
                )
                retriever = vector_store.as_retriever(
                    kwargs={"k": 5},
                )

                #image processing
                st.session_state.image_collection = None
                if images:
                    chromadb_client = chromadb.PersistentClient(path="images-store-vdb")
                    upload_dir = Path("images-store-vdb") / "uploaded_images"
                    upload_dir.mkdir(parents=True, exist_ok=True)

                    # Créer un ImageLoader pour charger les images depuis le disque
                    image_loader = ImageLoader()

                    # Créer le modèle d'embedding OpenCLIP qui convertira les images en vecteurs
                    image_embedding_model = OpenCLIPEmbeddingFunction()

                    # Créer ou récupérer une collection ChromaDB nommée "images_collection"
                    # Cette collection stockera les embeddings des images et permettra les recherches par similarité
                    chroma_vdb = chromadb_client.get_or_create_collection(
                        name="images_collection",  # Nom de la collection
                        data_loader= image_loader,  # Chargeur d'images,
                        embedding_function=image_embedding_model # Fonction pour générer les embeddings
                    )

                    allowed_suffixes = {".jpg", ".jpeg", ".png", ".webp"}
                    image_ids = []
                    image_uris = []

                    for image in images:
                        image_path = upload_dir / image.name
                        image_path.write_bytes(image.getbuffer())
                        if image_path.suffix.lower() in allowed_suffixes:
                            image_ids.append(image.name)
                            image_uris.append(image_path.as_posix())

                    if image_ids:
                        chroma_vdb.add(
                            ids=image_ids,
                            uris=image_uris,
                        )
                        st.session_state.image_collection = chroma_vdb
                    else:
                        st.warning("Aucune image valide trouvee (formats supportes: jpg, jpeg, png, webp).")
                    
                st.session_state.retriever = retriever
    st.subheader("Chatbot")
    user_question = st.text_input("Ask Your Question")
    if user_question:
        if "retriever" not in st.session_state:
            st.warning("Chargez d'abord vos documents puis cliquez sur Submit.")
            return

        context_docs = st.session_state.retriever.invoke(user_question)
        context_list = [d.page_content for d in context_docs]
        context_text = ". ".join(context_list)

        image_collection = st.session_state.get("image_collection")
        if image_collection is not None:
            results = image_collection.query(
                query_texts=[user_question],
                n_results=2,
                include=["uris", "distances"],
            )
            uris = results.get("uris", [[]])
            image_paths = uris[0] if uris and len(uris) > 0 else []

            if len(image_paths) >= 2:
                image_paths1 = image_paths[0]
                image_paths2 = image_paths[1]

                st.image(Image.open(image_paths1), caption="Image similaire 1")
                st.image(Image.open(image_paths2), caption="Image similaire 2")

                with open(image_paths1, "rb") as image_file:
                    image_data1 = image_file.read()
                with open(image_paths2, "rb") as image_file:
                    image_data2 = image_file.read()

                prompt_inputs = {
                    "system_message": vision_system_message,
                    "user_question": user_question,
                    "text_context": context_text,
                    "image_data1": base64.b64encode(image_data1).decode("utf-8"),
                    "image_data2": base64.b64encode(image_data2).decode("utf-8"),
                }
                response = vision_chain.invoke(prompt_inputs)
                st.markdown(response)
                return

            st.info("Moins de 2 images retrouvees. Reponse basee uniquement sur le texte.")

        prompt = prompt_template.format(context=context_text, input=user_question)
        resp = llm.invoke(prompt)
        st.write(resp.content)


if __name__ == "__main__":
    main()