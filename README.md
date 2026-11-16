# Speedy Semantic Search

The search behind [qdrant.tech](https://qdrant.tech): the search box of the
documentation, Markdown pages for agents at `/md`, skills and code example
search, and a docs MCP server at `/mcp`. It runs at `https://search.qdrant.tech`.

It has two parts:

- **The service**, in Rust, in [`service`](./service). It answers every
  request from [Qdrant](https://qdrant.tech) collections. See its
  [README](./service/README.md).
- **The indexers**, which crawl qdrant.tech and skills.qdrant.tech every night and
  fill the collections. Most are in Python, in [`indexer`](./indexer),
  see [dev-docs.md](./dev-docs.md); `index_pages` and `index_prefix` are Rust
  binaries in `service`.

| Collection | Filled by | Used by |
|---|---|---|
| `site` | `indexer.crawl`, then `indexer.encode` | `/api/search`, `search_docs` |
| `prefix-cache` | `index_prefix` (once, from `words.txt`) | `/api/search` for short queries |
| `sections` | `indexer.sections` | `/md`, `get_page` |
| `pages` | `index_pages`, from `/docs-manifest.json` | `list_docs`, page titles and old URLs, suggestions for missing pages |
| `skills` | `indexer.skills` | `/skills` |
| `snippet-search` | `indexer.snippets` | `/snippets/search`, `search_snippets` |

The [Update search index](./.github/workflows/update-search-index.yaml)
workflow runs the indexers nightly, and [Deploy](./.github/workflows/deploy.yml)
deploys the service on every push to `master`.
