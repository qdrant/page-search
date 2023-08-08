# Speedy Semantic Search

This project has two implementations, one in Python and one in Rust. See [dev-docs.md](./dev-docs.md) for an overview over the Python version and tools, and the [rust_search README](./rust_search) for one on the Rust version and helpers.

Both the python and Rust version contain a service that is able to use a Qdrant vector search engine to do a semantic search in the matter of milliseconds. The python version does a text search below 4 characters, while the Rust version always does a full semantic search.

Both versions use the [all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) model made available by Hugging Face for embedding, and [Qdrant](https://qdrant.tech) for searching.

The services both have the same `GET /api/search` endpoint with the following arguments:

* `q` contains the term that is going to be searched.
* `section` contains a section that is added by the python version's crawler that can be used to additionally filter the search to a particular section of the page. This argument is optional.

The code in this project powers the [Qdrant documentation](https://qdrant.tech/documentation)'s search box.
