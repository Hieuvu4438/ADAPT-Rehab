"""
RAG Pipeline for Clinical Rehabilitation Knowledge.

Retrieval-Augmented Generation grounds LLM responses in
verified clinical guidelines, reducing hallucination risk.

Knowledge base contents:
- Exercise definitions and parameters
- Clinical guidelines (ACSM, APTA, WHO)
- Contraindications and safety rules
- Vietnamese rehabilitation context

Usage:
    rag = RAGPipeline()
    rag.initialize(api_key="sk-...")
    rag.load_documents("data/knowledge_base/")
    result = rag.retrieve("What exercises for shoulder pain?")
    print(result.context)
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict
from pathlib import Path


@dataclass
class RAGResult:
    """Result of RAG retrieval."""
    context: str = ""
    sources: List[str] = field(default_factory=list)
    is_valid: bool = False


class RAGPipeline:
    """
    RAG pipeline for rehabilitation knowledge base.

    Uses LangChain + FAISS for vector storage and retrieval.
    Embeds documents using OpenAI embeddings.

    Example:
        >>> rag = RAGPipeline()
        >>> rag.initialize(api_key="sk-...")
        >>> rag.load_documents("data/knowledge_base/")
        >>> result = rag.retrieve("Bài tập nào an toàn cho đau vai?")
        >>> print(result.context)
    """

    def __init__(self):
        self._embeddings = None
        self._vectorstore = None
        self._documents = []
        self._is_initialized = False

    def initialize(self, api_key: Optional[str] = None, **kwargs) -> bool:
        """
        Initialize RAG pipeline with embeddings model.

        Args:
            api_key: OpenAI API key for embeddings.
            **kwargs: Additional parameters.

        Returns:
            bool: True if initialization successful.
        """
        try:
            from langchain_openai import OpenAIEmbeddings
            self._embeddings = OpenAIEmbeddings(openai_api_key=api_key)
            self._is_initialized = True
            return True
        except ImportError:
            print("[RAG] Install: pip install langchain langchain-openai faiss-cpu")
            return False
        except Exception as e:
            print(f"[RAG] Init failed: {e}")
            return False

    def load_documents(self, knowledge_base_path: str) -> bool:
        """
        Load documents from knowledge base directory.

        Args:
            knowledge_base_path: Path to directory with .txt/.md files.

        Returns:
            bool: True if documents loaded successfully.
        """
        try:
            from langchain_community.document_loaders import DirectoryLoader, TextLoader
            from langchain.text_splitter import RecursiveCharacterTextSplitter
            from langchain_community.vectorstores import FAISS

            path = Path(knowledge_base_path)
            if not path.exists():
                print(f"[RAG] Knowledge base not found: {knowledge_base_path}")
                return False

            # Load all text files
            loader = DirectoryLoader(
                str(path),
                glob="**/*.md",
                loader_cls=TextLoader,
                loader_kwargs={"encoding": "utf-8"},
            )
            documents = loader.load()

            if not documents:
                print(f"[RAG] No documents found in {knowledge_base_path}")
                return False

            # Split into chunks
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=500,
                chunk_overlap=50,
                separators=["\n\n", "\n", ". ", " "],
            )
            chunks = splitter.split_documents(documents)

            # Create vector store
            self._vectorstore = FAISS.from_documents(chunks, self._embeddings)
            self._documents = chunks

            print(f"[RAG] Loaded {len(chunks)} chunks from {len(documents)} documents")
            return True

        except Exception as e:
            print(f"[RAG] Load failed: {e}")
            return False

    def retrieve(self, query: str, k: int = 3) -> RAGResult:
        """
        Retrieve relevant context for a query.

        Args:
            query: User query about rehabilitation.
            k: Number of documents to retrieve.

        Returns:
            RAGResult with retrieved context and sources.
        """
        if not self._is_initialized or not self._vectorstore:
            return RAGResult()

        try:
            docs = self._vectorstore.similarity_search(query, k=k)
            context = "\n\n".join([d.page_content for d in docs])
            sources = [d.metadata.get("source", "unknown") for d in docs]

            return RAGResult(
                context=context,
                sources=list(set(sources)),
                is_valid=True,
            )
        except Exception as e:
            return RAGResult()

    def add_document(self, text: str, source: str = "manual") -> bool:
        """
        Add a single document to the knowledge base.

        Args:
            text: Document text.
            source: Source identifier.

        Returns:
            bool: True if added successfully.
        """
        try:
            from langchain.schema import Document
            from langchain_community.vectorstores import FAISS

            doc = Document(page_content=text, metadata={"source": source})

            if self._vectorstore:
                self._vectorstore.add_documents([doc])
            else:
                self._vectorstore = FAISS.from_documents([doc], self._embeddings)

            return True
        except Exception as e:
            print(f"[RAG] Add document failed: {e}")
            return False

    def save(self, path: str) -> bool:
        """Save vector store to disk."""
        if self._vectorstore:
            self._vectorstore.save_local(path)
            return True
        return False

    def load(self, path: str) -> bool:
        """Load vector store from disk."""
        try:
            from langchain_community.vectorstores import FAISS
            self._vectorstore = FAISS.load_local(path, self._embeddings)
            return True
        except Exception:
            return False

    def close(self):
        """Release resources."""
        self._vectorstore = None
        self._embeddings = None
        self._is_initialized = False
