import base64
from pathlib import Path
from uuid import uuid4

import chromadb
import streamlit as st
from PyPDF2 import PdfReader
from chromadb.utils.data_loaders import ImageLoader
from chromadb.utils.embedding_functions import OpenCLIPEmbeddingFunction
from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv(override=True)

PDF_COLLECTION_NAME = "data_collection"
IMAGE_COLLECTION_NAME = "images_collection"
IMAGE_STORE_PATH = Path("images-store-vdb")
UPLOADED_IMAGE_DIR = IMAGE_STORE_PATH / "uploaded_images"
SUPPORTED_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}
TEXT_RETRIEVAL_K = 5
IMAGE_RETRIEVAL_K = 2

llm = ChatOpenAI(model="gpt-4o", temperature=0)
parser = StrOutputParser()

text_prompt_template = """
Answer the following question based only on the provided context:
<context>
{context}
</context>
<question>
{question}
</question>
"""

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
                {"type": "image_url", "image_url": "{image_data1}"},
                {"type": "image_url", "image_url": "{image_data2}"},
            ],
        ),
    ]
)
vision_chain = image_prompt | llm | parser


def guess_mime_type(image_path: Path) -> str:
    suffix = image_path.suffix.lower()
    if suffix in {".jpg", ".jpeg"}:
        return "image/jpeg"
    if suffix == ".png":
        return "image/png"
    if suffix == ".webp":
        return "image/webp"
    return "application/octet-stream"


def encode_image_as_data_uri(image_path: str | Path) -> str:
    path = Path(image_path)
    mime_type = guess_mime_type(path)
    encoded_image = base64.b64encode(path.read_bytes()).decode("utf-8")
    return f"data:{mime_type};base64,{encoded_image}"


def pdf_process(pdf_docs):
    if not pdf_docs:
        st.warning("Veuillez charger au moins un PDF.")
        return None

    text_parts = []
    for pdf in pdf_docs:
        reader = PdfReader(pdf)
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text_parts.append(page_text)

    content = "\n".join(text_parts).strip()
    if not content:
        st.warning("Aucun texte exploitable n'a ete extrait des PDF charges.")
        return None

    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=512,
        chunk_overlap=16,
    )
    chunks = splitter.split_text(content)
    if not chunks:
        st.warning("Le texte extrait est vide apres decoupage.")
        return None

    embedding_model = OpenAIEmbeddings()
    vector_store = Chroma.from_texts(
        chunks,
        embedding_model,
        collection_name=PDF_COLLECTION_NAME,
    )
    return vector_store.as_retriever(search_kwargs={"k": TEXT_RETRIEVAL_K})


def images_process(images):
    if not images:
        st.warning("Aucune image chargee.")
        return None

    chromadb_client = chromadb.PersistentClient(path=str(IMAGE_STORE_PATH))
    UPLOADED_IMAGE_DIR.mkdir(parents=True, exist_ok=True)

    image_loader = ImageLoader()
    image_embedding_model = OpenCLIPEmbeddingFunction()
    chroma_vdb = chromadb_client.get_or_create_collection(
        name=IMAGE_COLLECTION_NAME,
        data_loader=image_loader,
        embedding_function=image_embedding_model,
    )

    image_ids = []
    image_uris = []

    for image in images:
        image_path = UPLOADED_IMAGE_DIR / image.name
        if image_path.suffix.lower() not in SUPPORTED_IMAGE_SUFFIXES:
            continue

        image_path.write_bytes(image.getbuffer())
        image_ids.append(f"{image_path.stem}-{uuid4().hex}")
        image_uris.append(image_path.as_posix())

    if not image_ids:
        st.warning("Aucune image valide trouvee (formats supportes: jpg, jpeg, png, webp).")
        return None

    chroma_vdb.add(ids=image_ids, uris=image_uris)
    st.session_state.image_collection = chroma_vdb
    return chroma_vdb


def find_context_text(retriever, question):
    if retriever is None:
        st.warning("Aucun retriever disponible. Veuillez charger vos documents PDF et cliquer sur Submit.")
        return ""

    context_docs = retriever.invoke(question)
    return "\n\n".join(doc.page_content for doc in context_docs if doc.page_content)


def find_context_images(image_collection, question, max_results=IMAGE_RETRIEVAL_K):
    if image_collection is None:
        return []

    results = image_collection.query(
        query_texts=[question],
        n_results=max_results,
        include=["uris", "distances"],
    )
    uris = results.get("uris", [[]])
    if not uris or not uris[0]:
        return []
    return uris[0]


def build_vision_inputs(user_question, context_text, image_paths):
    return {
        "system_message": vision_system_message,
        "user_question": user_question,
        "text_context": context_text,
        "image_data1": encode_image_as_data_uri(image_paths[0]),
        "image_data2": encode_image_as_data_uri(image_paths[1]),
    }


def main():
    st.set_page_config(page_title="RAG", layout="wide")
    st.subheader("Retrieval Augmented generation", divider="blue")

    with st.sidebar:
        st.title("Data loader")
        pdf_docs = st.file_uploader(label="Load your pdfs", accept_multiple_files=True)
        images = st.file_uploader(label="Load your Images", accept_multiple_files=True)

        if st.button("Submit"):
            with st.spinner("Loading"):
                st.session_state.retriever = pdf_process(pdf_docs)
                st.session_state.image_collection = None

                if images:
                    image_collection = images_process(images)
                    if image_collection is not None:
                        st.success(f"{image_collection.count()} image(s) chargee(s) avec succes.")
                    else:
                        st.warning("Aucune image n'a ete chargee.")

    st.subheader("Chatbot")
    user_question = st.text_input("Ask Your Question")
    if not user_question:
        return

    retriever = st.session_state.get("retriever")
    if retriever is None:
        st.warning("Chargez d'abord vos documents puis cliquez sur Submit.")
        return

    context_text = find_context_text(retriever, user_question)
    image_collection = st.session_state.get("image_collection")
    image_paths = find_context_images(image_collection, user_question)

    if len(image_paths) >= 2:
        prompt_inputs = build_vision_inputs(user_question, context_text, image_paths[:2])
        response = vision_chain.invoke(prompt_inputs)
        st.markdown(response)
        return

    if image_paths:
        st.info("Moins de 2 images retrouvees. Reponse basee uniquement sur le texte.")

    prompt = text_prompt_template.format(context=context_text, question=user_question)
    resp = llm.invoke(prompt)
    st.write(resp.content)


if __name__ == "__main__":
    main()