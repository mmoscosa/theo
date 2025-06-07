import asyncio
import aiohttp
import time

async def test():
    async with aiohttp.ClientSession() as session:
        # Send 3 events with unique timestamps
        tasks = []
        for i in range(3):
            payload = {
                'event': {
                    'type': 'app_mention',
                    'user': 'U1234567890',
                    'text': f'<@BOTID> Test {i}',
                    'ts': f'{int(time.time())}.{i:06d}',
                    'channel': 'C08LZPGQ4QK'
                }
            }
            
            async def send_event(p, idx):
                async with session.post('http://localhost:8000/slack/events', json=p) as resp:
                    data = await resp.json()
                    print(f'Event {idx}: {resp.status} - {data.get("message")}')
                    return resp.status == 200
            
            tasks.append(send_event(payload, i))
        
        # Send all concurrently
        results = await asyncio.gather(*tasks)
        print(f"Successful: {sum(results)}/3")
        
        # Check stats
        await asyncio.sleep(1)
        async with session.get('http://localhost:8000/conversations/stats') as resp:
            stats = await resp.json()
            print(f"Stats: {stats['stats']}")

asyncio.run(test()) 