import os
from dotenv import load_dotenv

CODE_DIR = os.path.dirname(os.path.realpath(__file__))
ROOT_DIR = os.path.dirname(CODE_DIR)
DATA_DIR = os.path.join(ROOT_DIR, 'data')

COLLECTION_NAME = "site"
SECTION_COLLECTION_NAME = "sections"
SNIPPET_COLLECTION_NAME = "snippet-search"

load_dotenv()

QDRANT_HOST = os.environ.get("QDRANT_HOST", "localhost")
QDRANT_PORT = os.environ.get("QDRANT_PORT", 6333)
QDRANT_API_KEY = os.environ.get("QDRANT_API_KEY")

NEURAL_ENCODER = 'sentence-transformers/all-MiniLM-L6-v2'
SNIPPET_ENCODER = 'mixedbread-ai/mxbai-embed-large-v1'

SEARCH_LIMIT = 5
SECTIONS_EXACT_LIMIT = int(os.environ.get("SECTIONS_EXACT_LIMIT", 100))
SECTIONS_SEARCH_LIMIT = int(os.environ.get("SECTIONS_SEARCH_LIMIT", 10))

SKILLS_COLLECTION_NAME = "skills"
SKILLS_EXACT_LIMIT = int(os.environ.get("SKILLS_EXACT_LIMIT", 100))
SKILLS_SEARCH_LIMIT = int(os.environ.get("SKILLS_SEARCH_LIMIT", 3))

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
