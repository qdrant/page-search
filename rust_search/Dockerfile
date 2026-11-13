# Leveraging the pre-built Docker images with
# cargo-chef and the Rust toolchain
# https://www.lpalmieri.com/posts/fast-rust-docker-builds/
FROM --platform=${BUILDPLATFORM:-linux/amd64} lukemathwalker/cargo-chef:latest-rust-1.93 AS chef
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

COPY . .

# Build actual target here
RUN cargo build --release

RUN mv target/release/service /service/service

FROM debian:13-slim
ARG APP=/service

RUN apt-get update \
    && apt-get install -y ca-certificates tzdata \
    && rm -rf /var/lib/apt/lists/*

ENV TZ=Etc/UTC \
    RUN_MODE=production

# data/ holds the request log; deploy.yml mounts a volume there.
RUN mkdir -p ${APP}/data

COPY --from=builder /service/service ${APP}/service
WORKDIR ${APP}

CMD ["./service"]
