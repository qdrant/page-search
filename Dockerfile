FROM node:current as builder

FROM python:3.10-slim

ENV PYTHONFAULTHANDLER=1 \
  PYTHONUNBUFFERED=1 \
  PYTHONHASHSEED=random \
  PIP_NO_CACHE_DIR=off \
  PIP_DISABLE_PIP_VERSION_CHECK=on \
  PIP_DEFAULT_TIMEOUT=100 \
  POETRY_VERSION=1.2.1

RUN pip install "poetry==$POETRY_VERSION"

# Copy only requirements to cache them in docker layer
WORKDIR /code
COPY poetry.lock pyproject.toml /code/

# Project initialization:
RUN poetry config virtualenvs.create false \
  && poetry install --no-dev --no-interaction --no-ansi

# Install pre-trained models here
# Example:
RUN python -c 'from sentence_transformers import SentenceTransformer; SentenceTransformer("all-MiniLM-L6-v2")'

# Creating folders, and files for a project:
COPY . /code

CMD uvicorn site_search.service:app --host 0.0.0.0 --port 8005 --workers ${WORKERS:-1}
