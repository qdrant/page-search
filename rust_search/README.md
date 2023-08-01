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
# if the service URL uses HTTPS, also supply a certificate file
export CERTS=#<PEM file with your service's public and private key>
cargo run --release --bin service
```
