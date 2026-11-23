"""One-port TCP relay for the quickstart's offline mode.

    python relay.py LISTEN_HOST:PORT TARGET_HOST:PORT

The offline stack's backend network is `internal`: nothing on it can reach
anything off it. Local inference still needs the host's Ollama, so this
forwards exactly one listening port to exactly one fixed target. It takes no
destination from the client, so it cannot be used to reach anything else.
"""

import asyncio
import sys


async def _pipe(reader, writer):
    try:
        while data := await reader.read(65536):
            writer.write(data)
            await writer.drain()
    except (ConnectionError, asyncio.CancelledError):
        pass
    finally:
        writer.close()


async def main(listen: str, target: str) -> None:
    lhost, lport = listen.rsplit(":", 1)
    thost, tport = target.rsplit(":", 1)

    async def handle(cr, cw):
        try:
            tr, tw = await asyncio.open_connection(thost, int(tport))
        except OSError as exc:
            print(f"relay: cannot reach {target}: {exc}", flush=True)
            cw.close()
            return
        await asyncio.gather(_pipe(cr, tw), _pipe(tr, cw))

    server = await asyncio.start_server(handle, lhost, int(lport))
    print(f"relay: {listen} -> {target}", flush=True)
    async with server:
        await server.serve_forever()


if __name__ == "__main__":
    asyncio.run(main(sys.argv[1], sys.argv[2]))
