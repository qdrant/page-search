import os
from dotenv import load_dotenv

CODE_DIR = os.path.dirname(os.path.realpath(__file__))
ROOT_DIR = os.path.dirname(CODE_DIR)
DATA_DIR = os.path.join(ROOT_DIR, 'data')

COLLECTION_NAME = "site-cohere"

load_dotenv()

QDRANT_HOST = os.environ.get("QDRANT_HOST", "localhost")
QDRANT_PORT = os.environ.get("QDRANT_PORT", 6333)
QDRANT_API_KEY = os.environ.get("QDRANT_API_KEY")

COHERE_API_KEY = os.environ.get("COHERE_API_KEY")

NEURAL_ENCODER = 'all-MiniLM-L6-v2'

SEARCH_LIMIT = 5
