import redis, contextlib
from redis.asyncio import Redis as AsyncRedis
from typing import Any, AsyncGenerator, Generator
from motor.motor_asyncio import AsyncIOMotorClient

class MongoAsynchronous:
    """A class used to asynchronous handling :class:`~motor.motor_asyncio.AsyncIOMotorClient`
    cursor data.

    Parameters
    ----------
    host : str
        Server host IP or use mongo URL directly: `mongodb://localhost:27017`
    **kwargs : Any
        Additional keywords passed to :class:`~motor.motor_asyncio.AsyncIOMotorClient`.

    Examples
    --------
    >>> async with MongoAsynchronous().connect() as m:
    ...     db = m.flask
    ...     scripts = await db.scripts_name.find({}).to_list()
    """
    def __init__(self, host='localhost:27017', **kwargs: Any) -> None:
        self.host = host
        self.mongo_kwargs = kwargs

    @contextlib.asynccontextmanager
    async def connect(self) -> AsyncGenerator[AsyncIOMotorClient, None]:
        """Fetch Mongo's connection to be asynchronous context manager.

        Yields:
            AsyncGenerator[Redis, None]: a Mongo's connection asynchronous generator.
        """
        self.client = AsyncIOMotorClient(self.host, **self.mongo_kwargs)
        try:
            yield self.client
        finally:
            self.client.close()

class RedisAsynchronous:
    """A class used to asynchronous handling :class:`~redis.asyncio.Redis`
    cache.

    Parameters
    ----------
    **kwargs : Any
        Additional keywords passed to :class:`~redis.asyncio.Redis`.

    See Also
    --------
    RedisAsynchronous.connect : an asynchronous redis connection handler.
    RedisAsynchronous.sync_connect : use the origin synchronous mechanism\
        to create the redis connection handler.

    Examples
    --------
    1. Use asynchronous handle:
    >>> async with RedisAsynchronous(decode_responses=True).connect() as r:
    ...     data = dict(items_a=['a', 'b', 'c'], items_b=[1, 2, 3])
    ...     await r.hset('my-data', mapping=data)
    ...     await r.expire('my-data', 3600)

    2. Use synchronous handle:
    >>> with RedisAsynchronous(decode_responses=True).sync_connect() as r:
    ...     data = dict(items_a=['a', 'b', 'c'], items_b=[1, 2, 3])
    ...     r.hset('my-data', mapping=data)
    ...     r.expire('my-data', 3600)
    """
    def __init__(self, **kwargs: Any) -> None:
        self.redis_kwargs = kwargs

    @contextlib.asynccontextmanager
    async def connect(self) -> AsyncGenerator[AsyncRedis, None]:
        """Fetch Redis's connection to be asynchronous context manager.

        Yields:
            AsyncGenerator[Redis, None]: a Redis's connection asynchronous generator.
        """
        self.redis = AsyncRedis(**self.redis_kwargs)
        try:
            yield self.redis
        finally:
            await self.redis.close()

    @contextlib.contextmanager
    def sync_connect(self) -> Generator[redis.Redis, Any, None]:
        """Fetch Redis's connection with provided connection pool.

        Yields:
            Generator[Redis, None]: a Redis's connection generator.
        """
        self.redis = redis.Redis(connection_pool=redis.ConnectionPool(**self.redis_kwargs))
        try:
            yield self.redis
        finally:
            if self.redis is not None:
                self.redis.close()
