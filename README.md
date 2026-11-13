# Speedy Semantic Search

The search behind [qdrant.tech](https://qdrant.tech): the search box of the
documentation, Markdown pages for agents at `/md`, skills and code example
search, and a docs MCP server at `/mcp`. It runs at `https://search.qdrant.tech`.

It has two parts:

- **The service**, in Rust, in [`rust_search`](./rust_search). It answers every
  request from [Qdrant](https://qdrant.tech) collections. See its
  [README](./rust_search/README.md).
- **The indexers**, which crawl qdrant.tech and skills.qdrant.tech every night and
  fill the collections. Most are in Python, in [`site_search`](./site_search),
  see [dev-docs.md](./dev-docs.md); `index_pages` and `index_prefix` are Rust
  binaries in `rust_search`.

| Collection | Filled by | Used by |
|---|---|---|
| `site` | `site_search.crawl`, then `site_search.encode` | `/api/search`, `search_docs` |
| `prefix-cache` | `index_prefix` (once, from `words.txt`) | `/api/search` for short queries |
| `sections` | `site_search.sections` | `/md`, `get_page` suggestions |
| `pages` | `index_pages`, after `sections` | Suggestions for missing pages |
| `skills` | `site_search.skills` | `/skills` |
| `snippet-search` | `site_search.snippets` | `/snippets/search`, `search_snippets` |

The [Update search index](./.github/workflows/update-search-index.yaml)
workflow runs the indexers nightly, and [Deploy](./.github/workflows/deploy.yml)
deploys the service on every push to `master`.
