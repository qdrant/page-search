Site search, fast

### Preparation

This version of the search uses a local model. So you need the model in the working path of the service. To get the ONNX file, run

```bash
curl -L -o all-MiniLM-L6-v2.onnx https://huggingface.co/optimum/all-MiniLM-L6-v2/resolve/main/model.onnx
```

in the `rust_search` directory.

Before we run anything, we need the URL and possibly API key. All binaries take these from environment variables. Set them as follows:

```bash
export QDRANT_URL=#<your Qdrant address including port>
export QDRANT_API_KEY=#<your Qdrant API key as needed>
```

Since the embeddings are the same as with the python search, you can easily re-use its collection. Alternatively you can run the `setup_collection` binary, after running `crawl` (see the above directory).

We also need to set up a prefix cache collection for the recommender function. To do that, run

```bash
cargo run --release --bin index_prefix
```

Running the service can be done via

```bash
export SERVICE_URL=#<the URL the service will be listening to>
cargo run --release --bin service
```

## Docs MCP server

The service also serves an MCP server at `/mcp`, so `https://search.qdrant.tech/mcp`
in production, that lets AI assistants search and read qdrant.tech: the
documentation, articles, the blog, and courses. `search_docs` takes a
`sources` argument to narrow the search to some of them, and
`DOCS_MCP_SOURCES` decides which of them the server serves at all.

It has four read-only tools:

| Tool | What it does |
|---|---|
| `search_docs(query, sources?, partition?, limit?, format?)` | Searches the `site` collection the same way as `/api/search`, one query per source, and returns one result per page with the headings and passages that matched |
| `search_snippets(query, language?, limit?, format?)` | Searches the `snippet-search` collection the same way as `/snippets/search`, with the language optional, and returns code examples with their descriptions and pages |
| `get_page(page)` | Returns a page's Markdown from `<page>/index.md` on the site, resolving old URLs through the page index. Also reads the full snippet pages under `/documentation/snippets/` |
| `list_docs(section?, offset?, limit?, format?)` | Browses the page index: docs and courses in sidebar order, articles and blog posts newest first |

Tools answer in Markdown by default. `format: "json"` returns the same result
as JSON, in both the text and the structured content. `get_page` returns
Markdown only. New tools should offer both formats where a JSON form makes sense.

Pages and the page index (`/docs-manifest.json`, built by the landing_page
repository) are fetched from the site, not from Qdrant. Only the two search tools use Qdrant.

Settings, from the environment, in addition to the service's own:

| Variable | Default | |
|---|---|---|
| `MCP_ALLOWED_HOSTS` | `search.qdrant.tech,localhost,127.0.0.1` | `Host` values `/mcp` accepts; others get 403 |
| `DOCS_MCP_SOURCES` | all four | Comma-separated subset of `documentation,articles,blog,course` to serve. `deploy.yml` sets all four |
| `DOCS_MCP_BASE_URL` | `https://qdrant.tech` | Where pages are read from, such as a local `hugo serve` |
| `DOCS_MCP_MANIFEST_TTL` | `300` | Seconds to cache the page index |
| `DOCS_MCP_HTTP_TIMEOUT` | `15` | Seconds to wait for the site |
| `QUERY_EMBEDDING` | `cloud` | `local` embeds queries in-process, see below |

### Testing locally

Qdrant Cloud embeds the query text in production. A local Qdrant can't, so
build with the `local-embeddings` feature and set `QUERY_EMBEDDING=local`,
which embeds queries with the same model through fastembed. The model is
downloaded on first use into `.fastembed_cache`, which git ignores. This
applies to `/api/search` and `/md` as well as `/mcp`.

```bash
docker run -d -p 6333:6333 -p 6334:6334 -v page-search-qdrant:/qdrant/storage qdrant/qdrant
# Index the `site` and `sections` collections with the Python tools in ../site_search

QDRANT_URL=http://localhost:6334 QUERY_EMBEDDING=local \
  cargo run --release --features local-embeddings --bin service
```

Then connect a client to `http://localhost:8005/mcp`, for example
`claude mcp add --transport http qdrant-docs http://localhost:8005/mcp`.

`cargo test` runs the tools in-process against a fake site and canned search
results, without Qdrant or network access.
