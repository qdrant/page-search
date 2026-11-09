# Indexers

The Python modules in `site_search` crawl the site and fill the Qdrant
collections. The service that answers searches is in `rust_search`.

## Setup

- `pip install virtualenv` - install venv manager
- `virtualenv venv` - create virtual env
- `source venv/bin/activate` - enter venv
- `pip install poetry` - install package manager
- `poetry install` - install all dependencies

They read `QDRANT_HOST`, `QDRANT_PORT` (6333 by default) and `QDRANT_API_KEY`
from the environment or a `.env` file.

## Run

### Site

Crawl the HTML of qdrant.tech into `data/abstracts.jsonl`, then embed it into
the `site` collection:

```
poetry run python -m site_search.crawl
poetry run python -m site_search.encode
```

### Sections

Read the Markdown of every page into the `sections` collection. Run the Rust
`index_pages` binary afterwards to rebuild `pages`.

```
poetry run python -m site_search.sections
```

### Skills

Read skills.qdrant.tech into the `skills` collection:

```
poetry run python -m site_search.skills
```

### Snippets

Read the code examples of every page into the `snippet-search` collection.
New examples get descriptions from OpenAI, so this needs `OPENAI_API_KEY`.

```
poetry run python -m site_search.snippets
```

## Docker

The update workflow builds the root `Dockerfile` and runs each module in it:

```
docker build -t page-search-indexer .
docker run --rm -e QDRANT_HOST -e QDRANT_API_KEY page-search-indexer python -m site_search.sections
```
