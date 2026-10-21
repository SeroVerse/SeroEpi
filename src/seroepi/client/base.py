"""Base abstractions for genotype parsers and asynchronous API clients."""

import asyncio
from collections.abc import AsyncIterator, Coroutine
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
from types import TracebackType
from typing import Any, Self

import aiohttp


class Client:
    """High-performance asynchronous HTTP client powered by ``aiohttp``.

    Args:
        base_url: Base endpoint URL.
        api_key_header: Optional HTTP header key for API authorization.
        api_key: Optional API key string.
        user_agent: User-Agent header value.
        timeout: Total request timeout in seconds.

    Attributes:
        base_url: Normalized base URL with trailing slashes stripped.
        api_key_header: Header name used to send ``api_key``, or ``None``.
        user_agent: HTTP ``User-Agent`` string sent on every request.
        timeout: ``aiohttp.ClientTimeout`` configuration.
    """

    def __init__(
        self,
        base_url: str,
        api_key_header: str | None = None,
        api_key: str | None = None,
        user_agent: str = "seroepi-client/0.1",
        timeout: float = 30.0,
    ) -> None:
        """Initializes the asynchronous API client.

        Args:
            base_url: Base endpoint URL.
            api_key_header: Optional HTTP header key for API authorization.
            api_key: Optional API key string.
            user_agent: User-Agent header value.
            timeout: Total request timeout in seconds.
        """
        self.base_url = base_url.rstrip("/")
        self.api_key_header = api_key_header
        self._api_key: str | None = api_key
        self.user_agent = user_agent
        self.timeout = aiohttp.ClientTimeout(total=timeout)
        self._session: aiohttp.ClientSession | None = None

    @property
    def api_key(self) -> str | None:
        """Returns the configured API key string, or ``None`` if not configured.

        Returns:
            API key string or ``None``.
        """
        return self._api_key

    def __repr__(self) -> str:
        """Masks API key credentials in string representation.

        Returns:
            Redacted representation string.
        """
        key_repr = "'**********'" if self._api_key else "None"
        return f"{self.__class__.__name__}(base_url='{self.base_url}', api_key={key_repr})"

    def _headers(self) -> dict[str, str]:
        headers = {"User-Agent": self.user_agent, "Accept": "application/json"}
        if self.api_key_header and self.api_key:
            headers[self.api_key_header] = self.api_key
        return headers

    async def get_session(self) -> aiohttp.ClientSession:
        """Retrieves or instantiates the underlying ``aiohttp.ClientSession`` for the active event loop.

        Returns:
            An open ``aiohttp.ClientSession`` bound to the current running event loop.
        """
        current_loop = asyncio.get_running_loop()
        if (
            self._session is None
            or self._session.closed
            or getattr(self._session, "_loop", None) is not current_loop
            or getattr(self._session, "_loop", current_loop).is_closed()
        ):
            self._session = aiohttp.ClientSession(
                headers=self._headers(),
                timeout=self.timeout,
            )
        return self._session

    async def __aenter__(self) -> Self:
        """Enters the async context and initializes the session.

        Returns:
            The active ``Client`` instance (``self``).
        """
        await self.get_session()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        """Exits the async context and closes the session.

        Args:
            exc_type: Exception class if an error occurred, otherwise ``None``.
            exc_val: Exception instance if an error occurred, otherwise ``None``.
            exc_tb: Traceback if an error occurred, otherwise ``None``.
        """
        await self.close()

    def __enter__(self) -> Self:
        """Enters the synchronous context.

        Returns:
            The ``Client`` instance (``self``).
        """
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        """Exits the synchronous context and closes the session.

        Args:
            exc_type: Exception class if an error occurred, otherwise ``None``.
            exc_val: Exception instance if an error occurred, otherwise ``None``.
            exc_tb: Traceback if an error occurred, otherwise ``None``.
        """
        self.close_sync()

    async def close(self) -> None:
        """Closes the underlying ``aiohttp.ClientSession`` if open."""
        if self._session is not None:
            if not self._session.closed:
                loop = getattr(self._session, "_loop", None)
                if loop is None or not loop.is_closed():
                    await self._session.close()
            self._session = None

    def close_sync(self) -> None:
        """Closes the underlying ``aiohttp.ClientSession`` synchronously."""
        if self._session is not None and not self._session.closed:
            loop = getattr(self._session, "_loop", None)
            if loop is not None and not loop.is_closed():
                self.run_sync(self.close())
            else:
                self._session = None

    def _url(self, endpoint: str) -> str:
        return f"{self.base_url}/{endpoint.lstrip('/')}"

    async def request(self, method: str, endpoint: str, **kwargs: Any) -> Any:  # noqa: ANN401
        """Sends an async HTTP request and returns parsed JSON or text.

        Args:
            method: HTTP method name (e.g., ``"GET"``, ``"POST"``, ``"DELETE"``).
            endpoint: Relative API endpoint path.
            **kwargs: Additional keyword arguments forwarded to ``session.request()``.

        Returns:
            Decoded JSON payload when ``Content-Type`` is ``application/json``, otherwise response text.

        Raises:
            aiohttp.ClientResponseError: If the HTTP response status is ``>= 400``.
        """
        session = await self.get_session()
        url = self._url(endpoint)
        async with session.request(method, url, **kwargs) as resp:
            if resp.status >= 400:
                body = (await resp.text()).strip()
                reason = resp.reason or "HTTP Error"
                detail = f"{reason}: {body[:300]}" if body else reason
                raise aiohttp.ClientResponseError(
                    resp.request_info,
                    resp.history,
                    status=resp.status,
                    message=detail,
                    headers=resp.headers,
                )
            if resp.content_type == "application/json":
                return await resp.json()
            return await resp.text()

    async def get(self, endpoint: str, **kwargs: Any) -> Any:  # noqa: ANN401
        """Performs an async HTTP GET request.

        Args:
            endpoint: Relative API endpoint path.
            **kwargs: Additional keyword arguments forwarded to ``request()``.

        Returns:
            Parsed JSON or text response payload.
        """
        return await self.request("GET", endpoint, **kwargs)

    async def post(self, endpoint: str, **kwargs: Any) -> Any:  # noqa: ANN401
        """Performs an async HTTP POST request.

        Args:
            endpoint: Relative API endpoint path.
            **kwargs: Additional keyword arguments forwarded to ``request()``.

        Returns:
            Parsed JSON or text response payload.
        """
        return await self.request("POST", endpoint, **kwargs)

    async def delete(self, endpoint: str, **kwargs: Any) -> Any:  # noqa: ANN401
        """Performs an async HTTP DELETE request.

        Args:
            endpoint: Relative API endpoint path.
            **kwargs: Additional keyword arguments forwarded to ``request()``.

        Returns:
            Parsed JSON or text response payload.
        """
        return await self.request("DELETE", endpoint, **kwargs)

    async def stream_bytes(self, method: str, endpoint: str, **kwargs: Any) -> bytes:
        """Sends an async HTTP request and reads raw response bytes.

        Args:
            method: HTTP method name.
            endpoint: Relative API endpoint path.
            **kwargs: Additional keyword arguments forwarded to ``session.request()``.

        Returns:
            Raw response body as ``bytes``.

        Raises:
            aiohttp.ClientResponseError: If the response status indicates an error.
        """
        session = await self.get_session()
        url = self._url(endpoint)
        async with session.request(method, url, **kwargs) as resp:
            resp.raise_for_status()
            return await resp.read()

    async def stream_buffer(self, method: str, endpoint: str, **kwargs: Any) -> BytesIO:
        """Sends an async HTTP request and returns an in-memory ``BytesIO`` stream.

        Args:
            method: HTTP method name.
            endpoint: Relative API endpoint path.
            **kwargs: Additional keyword arguments forwarded to ``stream_bytes()``.

        Returns:
            In-memory ``BytesIO`` buffer containing the response bytes.
        """
        b = await self.stream_bytes(method, endpoint, **kwargs)
        return BytesIO(b)

    async def stream_chunks(
        self,
        method: str,
        endpoint: str,
        chunk_size: int = 65536,
        **kwargs: Any,
    ) -> AsyncIterator[bytes]:
        """Streams response bytes in chunks as an async generator.

        Args:
            method: HTTP method name.
            endpoint: Relative API endpoint path.
            chunk_size: Maximum byte size per yielded chunk.
            **kwargs: Additional keyword arguments forwarded to ``session.request()``.

        Yields:
            Byte chunks read from the HTTP response stream.
        """
        session = await self.get_session()
        url = self._url(endpoint)
        async with session.request(method, url, **kwargs) as resp:
            resp.raise_for_status()
            async for chunk in resp.content.iter_chunked(chunk_size):
                yield chunk

    def run_sync[T](self, coro: Coroutine[Any, Any, T]) -> T:
        """Executes a coroutine synchronously and closes any session bound to the temporary event loop.

        Args:
            coro: Awaitable coroutine to run to completion.

        Returns:
            The result returned by ``coro``.
        """

        async def _runner() -> T:
            try:
                return await coro
            finally:
                await self.close()

        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop and loop.is_running():
            # If inside an existing running loop (e.g. PyShiny, Jupyter, or Marimo), run in worker thread
            with ThreadPoolExecutor(max_workers=1) as pool:
                future = pool.submit(lambda: asyncio.run(_runner()))
                return future.result()
        return asyncio.run(_runner())
