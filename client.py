import asyncio

import websockets


async def send_request():
    uri = "ws://localhost:8765"
    async with websockets.connect(uri) as websocket:
        await websocket.send("Top 10 companies with most patents")
        response = await websocket.recv()
        print(response)


asyncio.get_event_loop().run_until_complete(send_request())
