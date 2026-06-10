import fitz  # PyMuPDF
import os
import logging
from typing import List, Tuple
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_anthropic import ChatAnthropic
from langchain.schema import Document
from app.core.config import settings

logger = logging.getLogger(__name__)

FDA_LABELS_DIR = os.path.join(os.path.dirname(__file__), "../../data/fda_labels")


class RAGService:
    def __init__(self):
        self.vectorstores: dict[str, FAISS] = {}
        self.llm = ChatAnthropic(
            model="claude-3-5-haiku-20241022",
            api_key=settings.ANTHROPIC_API_KEY,
            max_tokens=512
        )
        self._load_all_labels()

    def _load_all_labels(self):
        """Load all FDA label PDFs from the data directory on startup."""
        if not os.path.exists(FDA_LABELS_DIR):
            logger.warning(f"FDA labels directory not found: {FDA_LABELS_DIR}")
            return

        for filename in os.listdir(FDA_LABELS_DIR):
            if filename.endswith(".pdf"):
                drug_name = filename.replace(".pdf", "").lower()
                try:
                    self._index_label(drug_name, os.path.join(FDA_LABELS_DIR, filename))
                    logger.info(f"Indexed FDA label: {drug_name}")
                except Exception as e:
                    logger.error(f"Failed to index {drug_name}: {e}")

    def _index_label(self, drug_name: str, pdf_path: str):
        """Extract text from PDF and index into FAISS."""
        doc = fitz.open(pdf_path)
        pages = []
        for page_num, page in enumerate(doc):
            text = page.get_text()
            if text.strip():
                pages.append(Document(
                    page_content=text,
                    metadata={"drug": drug_name, "page": page_num + 1, "source": pdf_path}
                ))

        splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
        chunks = splitter.split_documents(pages)

        # Use simple embeddings via Anthropic — fall back to TF-IDF style for now
        from langchain_community.embeddings import HuggingFaceEmbeddings
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        self.vectorstores[drug_name] = FAISS.from_documents(chunks, embeddings)

    def retrieve_context(self, drug_name: str, query: str, k: int = 4) -> List[Tuple[str, str]]:
        """
        Retrieve top-k relevant passages from the drug's FDA label.
        Returns list of (passage, section_hint) tuples.
        """
        drug_key = drug_name.lower()
        if drug_key not in self.vectorstores:
            logger.warning(f"No FDA label indexed for drug: {drug_name}")
            return []

        results = self.vectorstores[drug_key].similarity_search(query, k=k)
        return [
            (doc.page_content, f"Page {doc.metadata.get('page', '?')}")
            for doc in results
        ]

    def is_drug_indexed(self, drug_name: str) -> bool:
        return drug_name.lower() in self.vectorstores

    def list_indexed_drugs(self) -> List[str]:
        return list(self.vectorstores.keys())


# Singleton
rag_service = RAGService()
