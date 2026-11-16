import asyncio
import time
from collections.abc import Awaitable
from typing import Callable, ParamSpec, TypeVar

import aiohttp
import httpx
import openai
import requests
from loguru import logger
from qdrant_client.http import exceptions
from qdrant_client.http.exceptions import ResponseHandlingException
from usp.objects.sitemap import InvalidSitemap

P = ParamSpec("P")
T = TypeVar("T")


class TooManyRetriesError(Exception):
    pass


class InvalidResultError(Exception):
    pass


RETRIEABLE_EXCEPTIONS: tuple[type[Exception], ...] = (
    InvalidResultError,
    ResponseHandlingException,
    aiohttp.ClientConnectionError,
    aiohttp.ClientConnectorError,
    aiohttp.ClientPayloadError,
    aiohttp.ServerTimeoutError,
    aiohttp.SocketTimeoutError,
    exceptions.ResponseHandlingException,
    httpx.NetworkError,
    httpx.RemoteProtocolError,
    httpx.TimeoutException,
    openai.APIError,
    requests.ReadTimeout,
    requests.exceptions.ConnectionError,
)


def retry(
    fn: Callable[P, T],
    max_retries: int | None,
    wait: int = 1,
) -> Callable[P, T]:
    def inner(*args: P.args, **kwargs: P.kwargs) -> T:
        num_tries = 0
        while True:
            try:
                result = fn(*args, **kwargs)
                if isinstance(result, InvalidSitemap):
                    raise InvalidResultError
                return result
            except Exception as e:
                if max_retries is not None and num_tries >= max_retries:
                    raise TooManyRetriesError

                if isinstance(e, RETRIEABLE_EXCEPTIONS):
                    logger.warning(
                        f"{repr(fn)} failed with {repr(e)}, retrying after {wait}"
                    )
                    num_tries += 1
                    time.sleep(wait)
                    continue
                else:
                    raise

    return inner


def retry_async(
    fn: Callable[P, Awaitable[T]],
    max_retries: int | None,
    wait: int = 1,
) -> Callable[P, Awaitable[T]]:
    async def inner(*args: P.args, **kwargs: P.kwargs) -> T:
        num_tries = 0
        while True:
            try:
                result = await fn(*args, **kwargs)
                if isinstance(result, InvalidSitemap):
                    raise InvalidResultError
                return result
            except Exception as e:
                if max_retries is not None and num_tries >= max_retries:
                    raise TooManyRetriesError

                if isinstance(e, RETRIEABLE_EXCEPTIONS):
                    logger.warning(
                        f"{repr(fn)} failed with {repr(e)}, retrying after {wait}"
                    )
                    num_tries += 1
                    await asyncio.sleep(wait)
                    continue
                else:
                    raise

    return inner
