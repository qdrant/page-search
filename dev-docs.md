# Development documentation


## Setup

- `pip install virtualenv` - install venv manager
- `virtualenv venv` - create virtual env 
- `source venv/bin/activate` - enter venv
- `pip install poetry` - install package manager
- `poetry install` - install all dependencies

## Tests

The section-indexer regression tests run without a Qdrant server or model downloads:

```
python -m unittest discover -s tests -v
```

The indexer loads the embedding model and generates a sample vector before deleting
the existing sections collection. A failure during this preflight leaves the existing
index intact. Failures after deletion, such as crawling or upload errors, can still
leave an incomplete index.

With a local Qdrant instance running

```
poetry run pytest
```

or with a Docker container

```
./tests/integration-tests.sh
```

## Run

### Crawler

Crawl content of documentation website

```
poetry run python site_search/crawl.py 
```

It creates the file `data/abstracts.jsonl`.

### Encode & upload

To encode the scraped abstract and upload the vectors

```
poetry run python site_search/encode.py 
```

### Service

To access the search service built-on top of Qdrant.

```
poetry run python site_search/service.py
```

then go to

```
http://localhost:8000/docs
```
