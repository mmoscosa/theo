#!/usr/bin/env python3
"""
Simple concurrent test for Theo threading
"""
import asyncio
import aiohttp
import time

async def send_slack_event(session, event_id):
    """Send a single Slack event"""
    payload = {
        "event": {
            "type": "app_mention",
            "user": "U1234567890",
            "text": f"<@BOTID> Simple test {event_id}",
            "ts": f"1234567890.{event_id:06d}",
            "channel": "C08LZPGQ4QK"
        }
    }
    
    start_time = time.time()
    try:
        async with session.post(
            "http://localhost:8000/slack/events",
            json=payload,
            timeout=aiohttp.ClientTimeout(total=2)  # 2 second timeout for quick response
        ) as response:
            response_data = await response.json()
            response_time = time.time() - start_time
            
            print(f"Event {event_id}: {response.status} in {response_time:.3f}s - {response_data.get('message')}")
            return response.status == 200
    except Exception as e:
        response_time = time.time() - start_time
        print(f"Event {event_id}: ERROR in {response_time:.3f}s - {e}")
        return False

async def get_stats(session):
    """Get conversation stats"""
    try:
        async with session.get("http://localhost:8000/conversations/stats") as response:
            if response.status == 200:
                return await response.json()
    except Exception as e:
        print(f"Stats error: {e}")
    return None

async def main():
    """Run simple concurrent test"""
    print("🧪 Simple Concurrent Test for Theo Threading")
    print("=" * 50)
    
    async with aiohttp.ClientSession() as session:
        # Get initial stats
        initial_stats = await get_stats(session)
        print(f"Initial stats: {initial_stats['stats'] if initial_stats else 'N/A'}")
        
        # Send 5 concurrent events
        print(f"\n📤 Sending 5 concurrent events...")
        start_time = time.time()
        
        tasks = [send_slack_event(session, i) for i in range(5)]
        results = await asyncio.gather(*tasks)
        
        send_duration = time.time() - start_time
        successful = sum(results)
        
        print(f"\n✅ Sent 5 events in {send_duration:.3f}s")
        print(f"   Successful: {successful}/5")
        
        # Wait a moment for processing
        print(f"\n⏳ Waiting 3 seconds for background processing...")
        await asyncio.sleep(3)
        
        # Check stats after processing
        final_stats = await get_stats(session)
        if final_stats:
            print(f"\nFinal stats:")
            print(f"   Total: {final_stats['stats']['total']}")
            print(f"   Completed: {final_stats['stats']['completed']}")
            print(f"   Processing: {final_stats['stats']['processing']}")
            print(f"   Active: {final_stats['active_conversations']}")
        
        # Wait a bit more to see if processing completes
        if final_stats and final_stats['stats']['processing'] > 0:
            print(f"\n⏳ Still processing, waiting 5 more seconds...")
            await asyncio.sleep(5)
            
            final_final_stats = await get_stats(session)
            if final_final_stats:
                print(f"\nFinal final stats:")
                print(f"   Total: {final_final_stats['stats']['total']}")
                print(f"   Completed: {final_final_stats['stats']['completed']}")
                print(f"   Processing: {final_final_stats['stats']['processing']}")
                print(f"   Errors: {final_final_stats['stats']['error']}")

if __name__ == "__main__":
    asyncio.run(main()) 