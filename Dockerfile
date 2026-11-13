# The indexers for the site, sections, skills, and snippet-search collections.
# The update-search-index workflow builds it and runs one module per job.
FROM python:3.10-slim

ENV PYTHONFAULTHANDLER=1 \
  PYTHONUNBUFFERED=1 \
  PYTHONHASHSEED=random \
  PIP_NO_CACHE_DIR=off \
  PIP_DISABLE_PIP_VERSION_CHECK=on \
  PIP_DEFAULT_TIMEOUT=100 \
  POETRY_VERSION=2.3.2

RUN pip install "poetry==$POETRY_VERSION"

# Copy only requirements to cache them in docker layer
WORKDIR /code
COPY poetry.lock pyproject.toml /code/

# Project initialization:
RUN poetry config virtualenvs.create false \
  && poetry install --without dev --no-interaction --no-ansi --no-root

# The model encode.py embeds the site collection with
RUN python -c 'from fastembed import TextEmbedding; TextEmbedding("sentence-transformers/all-MiniLM-L6-v2")'


# Creating folders, and files for a project:
COPY . /code
