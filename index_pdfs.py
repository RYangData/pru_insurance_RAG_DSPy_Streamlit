from dotenv import load_dotenv
import os
from llama_index.core.node_parser import MarkdownElementNodeParser
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core import VectorStoreIndex
from llama_index.core import Settings
from llama_index.core import StorageContext
from llama_index.vector_stores.qdrant import QdrantVectorStore
import qdrant_client
from qdrant_client import QdrantClient

# Load environment variables from .env file
load_dotenv()

# Import LlamaParse library for document parsing
from llama_parse import LlamaParse
import nest_asyncio; nest_asyncio.apply()  # Allows asyncio to run in a notebook or script

# Initialize the LlamaParse with specific settings for parsing insurance documents
parser = LlamaParse(
    result_type="markdown",  # Defines the output format of parsed data
    parsing_instruction="""
    This document is structured as an insurance product brochure.
  
    Identify the document title at the very beginning. The title usually serves as the product name.

    # General Formatting for Headers
    - Section titles should succinctly summarize the content that follows.
    - Only assign headers related to insurance products or key sections found in a product brochure.

    ## Sub-section Headers
    - Identify sub-sections related to the main sections.

    Parsing Tables
    - Handle tables with care to maintain consistency with the original format.

    Note: Maintain logical flow and structure for easy navigation and comprehension.
    """
)

# Start document parsing
print("Starting Document Parsing")
documents = parser.load_data("./data/pruhealth-critical-illness-extended-care-iii-product-brochure.pdf")
print("Successfully Parsed extended-care-iii-product-brochure.pdf")

documents_2 = parser.load_data("./data/pruhealth-critical-illness-first-protect-ii-product-brochure.pdf")
print("Successfully Parsed first-protect-ii-product-brochure.pdf")

# Initialize Qdrant client for vector storage
db_client = qdrant_client.QdrantClient(
    host="localhost",  # Localhost address of Qdrant service
    port=6333  # Port where Qdrant is listening
)

# Define settings for text embedding model used in LlamaIndex
Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small")

# Set up vector store and storage context for managing embeddings
vector_store = QdrantVectorStore(client=db_client, collection_name="llama_parse_prudential_health_both_pdfs")
storage_context = StorageContext.from_defaults(vector_store=vector_store)

# Create and populate the vector store index with parsed documents
index = VectorStoreIndex.from_documents(
    documents + documents_2,
    storage_context=storage_context,
)
print("Index creation and population complete.")
