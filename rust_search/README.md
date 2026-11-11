# Site search service

The service behind the search on qdrant.tech. It answers from the Qdrant
collections that the indexers fill (see the [root README](../README.md)) and
serves:

| Endpoint | What it does |
|---|---|
| `GET /api/search?q=&section=&partition=` | The site search box: lines of the `site` collection |
| `GET /md/<page>?s=&q=` | Pages and sections of the `sections` collection as Markdown |
| `GET /skills` | Searches the `skills` collection |
| `GET /snippets/search` | Searches code examples in the `snippet-search` collection |
| `/mcp` | The docs MCP server, see below |
| `/admin` | Request statistics, see below |

### Running

All binaries read the Qdrant address and API key from the environment:

```bash
export QDRANT_URL=#<your Qdrant address including the gRPC port, 6334>
export QDRANT_API_KEY=#<your Qdrant API key as needed>
```

Without `QDRANT_URL`, they use `https://$QDRANT_HOST:6334` if `QDRANT_HOST` is
set, and `http://localhost:6334` otherwise.

Queries are embedded by Qdrant Cloud with `sentence-transformers/all-MiniLM-L6-v2`.
For a local Qdrant, see [Testing locally](#testing-locally).

Short queries are matched against embeddings of word prefixes in the
`prefix-cache` collection. Fill it once with

```bash
cargo run --release --bin index_prefix
```

Run the service with

```bash
export SERVICE_URL=#<the address the service listens on, 0.0.0.0:8005 by default>
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
| `get_page(page)` | Returns a page's Markdown from `<page>/index.md` on the site, resolving old URLs through the page index. Also reads the full snippet pages under `/documentation/snippets/`. For a missing page, suggests existing ones |
| `list_docs(section?, offset?, limit?, format?)` | Browses the page index: docs and courses in sidebar order, articles and blog posts newest first |

Tools answer in Markdown by default. `format: "json"` returns the same result
as JSON, in both the text and the structured content. `get_page` returns
Markdown only. New tools should offer both formats where a JSON form makes sense.

Pages and the page index (`/docs-manifest.json`, built by the landing_page
repository) are fetched from the site, not from Qdrant. Only the two search tools
and the suggestions for missing pages use Qdrant.

## Suggestions for missing pages

When `/md` or `get_page` finds no page, it suggests existing ones from the
`pages` collection. Each page has three vectors, and Qdrant ranks them in one
query (`similar_query` in `src/similar_paths.rs`):

| Vector | Built from | Finds |
|---|---|---|
| `slug` | letter bigrams and trigrams of the slug, unit length (cosine) | misspellings |
| `words` | `qdrant/bm25` of the slug words and title, embedded by Qdrant, IDF | other forms of the same words, without stopwords |
| `meaning` | the title, slug words, and headings, embedded with the query model | rephrased paths |

Each vector is searched with a minimum score, so a path that resembles nothing
gets no suggestions. The hits are fused with RRF, and a formula raises pages in
the same source and below the same parents as the missing path. The closest
existing parent page and its main page follow, checked in the same request.

When the page exists but the `?s=` section doesn't, `/md` lists the page's
sections in order, using `order_by` on the integer index of `line` in
`sections`, and links the whole page.

## The `pages` collection

The `index_pages` binary builds `pages` from the pages, titles, and headings of
the `sections` collection, so run it after `site_search.sections`, as the update
workflow does. It encodes pages with the same code as the queries, `page_point`
in `src/similar_paths.rs`, and embeds `meaning` like queries, as
`QUERY_EMBEDDING` says: Qdrant Cloud embeds it on upsert by default.

It also stores what the site's page index, `/docs-manifest.json`, says about
each page, so the service won't need to read the page index from the site.
`index_pages` is the only part of this project that reads it. The payload of
a page (`PageMeta` in `src/pages.rs`):

| Field | Index | |
|---|---|---|
| `page` | keyword | Path without slashes at the ends: `documentation/search/filtering` |
| `title` | | From the page index, or the first heading of pages it doesn't list |
| `source`, `ancestors` | keyword | First segment of the path, and the paths above the page |
| `description`, `kind`, `partition`, `weight` | | From the page index. `kind` is `page` or `section` |
| `aliases` | keyword | Old paths of the page, written like `page` |
| `date`, `lastmod`, `external_url` | | From the page index, when it has them |
| `rendered` | bool | `false` for a section that only groups pages in the sidebar |

Pages the page index doesn't list, such as `/pricing/`, get the defaults: no
description, kind `page`, weight 0. Pages it lists that have no sections can't
be read, so they are left out, and `index_pages` prints them. Sections without
a page of their own are kept, for listing, but never suggested.

| Variable | Default | |
|---|---|---|
| `DOCS_MANIFEST_URL` | `https://qdrant.tech/docs-manifest.json` | Where `index_pages` reads the page index |

```bash
QDRANT_URL=http://localhost:6334 QUERY_EMBEDDING=local \
  cargo run --release --features local-embeddings --bin index_pages
```

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
  cargo run --release --features local-embeddings --bin index_pages

QDRANT_URL=http://localhost:6334 QUERY_EMBEDDING=local \
  cargo run --release --features local-embeddings --bin service
```

Then connect a client to `http://localhost:8005/mcp`, for example
`claude mcp add --transport http qdrant-docs http://localhost:8005/mcp`.

`cargo test` runs the tools in-process against a fake site and canned search
results, without Qdrant or network access.

## Admin panel

`/admin` shows statistics of the requests to `/api/search`, `/md`, `/skills`,
`/snippets/search`, and `/mcp`: requests per day by endpoint, errors,
latencies, and the most frequent queries. Selecting a request runs it again
and shows what the service returns for it now. Responses aren't stored.

The search box searches as you type, so the most frequent queries leave out
partial ones: a search is left out when a longer search starting with it
follows within 10 seconds from the same user agent. The log has no IPs, so
the user agent stands in for the client. The request list and the counts
still include partial queries.

Requests are logged to SQLite, `data/requests.sqlite` in the working
directory, which is `/service/data` in the container. `deploy.yml` mounts the
`qdrant-site-search-data` volume there so the log survives deploys. Records
older than the retention period are deleted hourly. The log keeps the request
path, query string, status, latency, and user agent, plus the JSON-RPC body
of MCP requests. It doesn't keep IP addresses.

Admins sign in with Google. Accounts from `ADMIN_ALLOWED_DOMAINS` (checked
against Google's hosted-domain claim, so only Workspace accounts of that
domain) or listed in `ADMIN_ALLOWED_EMAILS` get in; a sign-in lasts 12 hours.
Without the OAuth settings or an allow list, `/admin` answers 503 and the rest
of the service works as before.

| Variable | Default | |
|---|---|---|
| `GOOGLE_OAUTH_CLIENT_ID` | | OAuth client of type "Web application" in GCP |
| `GOOGLE_OAUTH_CLIENT_SECRET` | | |
| `ADMIN_SESSION_KEY` | | Random secret of at least 32 bytes, such as `openssl rand -hex 32`. Changing it signs everyone out |
| `ADMIN_ALLOWED_DOMAINS` | | Comma-separated Workspace domains. `deploy.yml` sets `qdrant.com` |
| `ADMIN_ALLOWED_EMAILS` | | Comma-separated accounts allowed whatever their domain |
| `ADMIN_PUBLIC_URL` | `https://search.qdrant.tech` | Where the service is reached. The OAuth client needs `<ADMIN_PUBLIC_URL>/admin/auth/callback` as an authorized redirect URI |
| `REQUEST_LOG_PATH` | `data/requests.sqlite` | |
| `REQUEST_LOG_RETENTION_DAYS` | `90` | |

To try it locally, create an OAuth client with
`http://localhost:8005/admin/auth/callback` as a redirect URI and set
`ADMIN_PUBLIC_URL=http://localhost:8005`.
