# Leveraging the pre-built Docker images with
# cargo-chef and the Rust toolchain
# https://www.lpalmieri.com/posts/fast-rust-docker-builds/
FROM --platform=${BUILDPLATFORM:-linux/amd64} lukemathwalker/cargo-chef:latest-rust-1.82 AS chef
WORKDIR /service

FROM chef AS planner
COPY . .
RUN cargo chef prepare --recipe-path recipe.json

FROM chef as builder

WORKDIR /service

COPY --from=planner /service/recipe.json recipe.json

RUN apt-get update && apt-get install -y clang cmake && rustup component add rustfmt

# Build dependencies - this is the caching Docker layer!
RUN cargo chef cook --release --recipe-path recipe.json

RUN curl -L -o all-MiniLM-L6-v2.onnx https://huggingface.co/optimum/all-MiniLM-L6-v2/resolve/main/model.onnx

COPY . .

# Build actual target here
RUN cargo build --release

RUN mv target/release/service /service/service

RUN mv target/release/libonnxruntime* /service/

FROM debian:12-slim
ARG APP=/service

RUN apt-get update \
    && apt-get install -y ca-certificates tzdata \
    && rm -rf /var/lib/apt/lists/*

ENV TZ=Etc/UTC \
    RUN_MODE=production

RUN mkdir -p ${APP}

COPY --from=builder /service/service ${APP}/service
COPY --from=builder /service/all-MiniLM-L6-v2.onnx ${APP}/all-MiniLM-L6-v2.onnx
COPY --from=builder /service/libonnxruntime* ${APP}/

COPY vocab.txt ${APP}/vocab.txt
COPY special_tokens_map.json ${APP}/special_tokens_map.json

WORKDIR ${APP}

CMD ["./service"]
