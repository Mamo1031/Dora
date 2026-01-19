"""Document processing module for loading and splitting PDF documents."""

from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


class DocumentProcessor:
    """Process documents for RAG: load PDFs and split text into chunks."""

    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
    ) -> None:
        """Initialize the document processor.

        Parameters
        ----------
        chunk_size : int, optional
            Size of each text chunk in characters, by default 1000
        chunk_overlap : int, optional
            Number of characters to overlap between chunks, by default 200
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
        )

    def load_pdf(self, file_path: str | Path) -> list:
        """Load a PDF file and return document chunks.

        Parameters
        ----------
        file_path : str | Path
            Path to the PDF file

        Returns
        -------
        list
            List of document chunks (LangChain Document objects)

        Raises
        ------
        FileNotFoundError
            If the PDF file does not exist
        ValueError
            If the file is not a PDF
        RuntimeError
            If PDF loading fails
        """
        file_path = Path(file_path)

        if not file_path.exists():
            msg = f"PDF file not found: {file_path}"
            raise FileNotFoundError(msg)

        if file_path.suffix.lower() != ".pdf":
            msg = f"File must be a PDF: {file_path}"
            raise ValueError(msg)

        try:
            loader = PyPDFLoader(str(file_path))
            documents = loader.load()

            # Split documents into chunks
            chunks = self.text_splitter.split_documents(documents)

            # Add metadata about source file
            for chunk in chunks:
                if "source" not in chunk.metadata:
                    chunk.metadata["source"] = str(file_path)
                chunk.metadata["file_name"] = file_path.name
        except Exception as e:
            msg = f"Failed to load PDF: {e}"
            raise RuntimeError(msg) from e
        else:
            return chunks
