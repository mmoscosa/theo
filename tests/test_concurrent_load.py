#!/usr/bin/env python3
"""
Load test for concurrent conversation processing
"""
import asyncio
import aiohttp
import time
import json
from datetime import datetime

async def simulate_slack_event(session, event_id):
    """Simulate a Slack event being sent to our API"""
    
    # Create a mock Slack event payload
    payload = {
        "event": {
            "type": "app_mention",
            "user": "U1234567890",
            "text": f"<@BOTID> Test question {event_id} - can you help me with this?",
            "ts": f"1234567890.{event_id:06d}",
            "channel": "C08LZPGQ4QK",
            "thread_ts": f"1234567890.{event_id:06d}"
        }
    }
    
    start_time = time.time()
    
    try:
        async with session.post(
            "http://localhost:8000/slack/events",
            json=payload,
            timeout=aiohttp.ClientTimeout(total=5)  # Quick timeout since we expect immediate response
        ) as response:
            response_data = await response.json()
            response_time = time.time() - start_time
            
            print(f"Event {event_id}: HTTP {response.status} in {response_time:.3f}s - {response_data.get('message', 'no message')}")
            
            if response.status == 200 and "conversation_id" in response_data:
                return {
                    "event_id": event_id,
                    "success": True,
                    "response_time": response_time,
                    "conversation_id": response_data["conversation_id"]
                }
            else:
                return {
                    "event_id": event_id,
                    "success": False,
                    "response_time": response_time,
                    "error": f"HTTP {response.status}"
                }
                
    except Exception as e:
        response_time = time.time() - start_time
        print(f"Event {event_id}: ERROR in {response_time:.3f}s - {str(e)}")
        return {
            "event_id": event_id,
            "success": False,
            "response_time": response_time,
            "error": str(e)
        }

async def monitor_conversation_stats(session, duration_seconds=30):
    """Monitor conversation statistics during the test"""
    print(f"\n📊 Monitoring conversation stats for {duration_seconds} seconds...")
    
    end_time = time.time() + duration_seconds
    
    while time.time() < end_time:
        try:
            async with session.get("http://localhost:8000/conversations/stats") as response:
                if response.status == 200:
                    stats = await response.json()
                    timestamp = datetime.now().strftime("%H:%M:%S")
                    print(f"[{timestamp}] Active: {stats['active_conversations']}, "
                          f"Total: {stats['stats']['total']}, "
                          f"Completed: {stats['stats']['completed']}, "
                          f"Processing: {stats['stats']['processing']}, "
                          f"Error: {stats['stats']['error']}")
                else:
                    print(f"[{timestamp}] Failed to get stats: HTTP {response.status}")
        except Exception as e:
            timestamp = datetime.now().strftime("%H:%M:%S")
            print(f"[{timestamp}] Stats error: {e}")
        
        await asyncio.sleep(2)  # Check every 2 seconds

async def run_load_test():
    """Run the concurrent load test"""
    print("🚀 Starting concurrent load test...\n")
    
    # Test parameters
    num_events = 15  # More than our concurrent limit of 10
    monitor_duration = 45  # Monitor for 45 seconds
    
    async with aiohttp.ClientSession() as session:
        print(f"📤 Sending {num_events} concurrent Slack events...")
        
        # Start monitoring task
        monitor_task = asyncio.create_task(monitor_conversation_stats(session, monitor_duration))
        
        # Send all events concurrently
        start_time = time.time()
        event_tasks = [simulate_slack_event(session, i) for i in range(num_events)]
        results = await asyncio.gather(*event_tasks, return_exceptions=True)
        send_duration = time.time() - start_time
        
        print(f"\n✅ Sent {num_events} events in {send_duration:.3f}s")
        
        # Analyze results
        successful = [r for r in results if isinstance(r, dict) and r.get("success")]
        failed = [r for r in results if isinstance(r, dict) and not r.get("success")]
        exceptions = [r for r in results if isinstance(r, Exception)]
        
        print(f"\n📊 Results Summary:")
        print(f"   ✅ Successful: {len(successful)}")
        print(f"   ❌ Failed: {len(failed)}")
        print(f"   💥 Exceptions: {len(exceptions)}")
        
        if successful:
            response_times = [r["response_time"] for r in successful]
            print(f"   ⏱️  Avg response time: {sum(response_times)/len(response_times):.3f}s")
            print(f"   ⏱️  Max response time: {max(response_times):.3f}s")
            print(f"   ⏱️  Min response time: {min(response_times):.3f}s")
        
        if failed:
            print(f"\n❌ Failed events:")
            for failure in failed[:5]:  # Show first 5 failures
                print(f"   Event {failure['event_id']}: {failure.get('error', 'unknown error')}")
        
        if exceptions:
            print(f"\n💥 Exceptions:")
            for exc in exceptions[:3]:  # Show first 3 exceptions
                print(f"   {type(exc).__name__}: {str(exc)}")
        
        # Wait for monitoring to finish
        print(f"\n⏳ Waiting for background processing to complete...")
        try:
            await monitor_task
        except asyncio.CancelledError:
            pass
        
        # Get final stats
        try:
            async with session.get("http://localhost:8000/conversations/stats") as response:
                if response.status == 200:
                    final_stats = await response.json()
                    print(f"\n🏁 Final Statistics:")
                    print(f"   Total conversations: {final_stats['stats']['total']}")
                    print(f"   Completed: {final_stats['stats']['completed']}")
                    print(f"   Still processing: {final_stats['stats']['processing']}")
                    print(f"   Errors: {final_stats['stats']['error']}")
                    print(f"   Timeouts: {final_stats['stats']['timeout']}")
                    print(f"   Currently active: {final_stats['active_conversations']}")
        except Exception as e:
            print(f"Failed to get final stats: {e}")

async def main():
    """Main test function"""
    print("🧪 Theo Concurrent Processing Load Test")
    print("=" * 50)
    
    # First, check if the server is responding
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get("http://localhost:8000/health") as response:
                if response.status == 200:
                    print("✅ Server is responding")
                else:
                    print(f"❌ Server health check failed: HTTP {response.status}")
                    return
    except Exception as e:
        print(f"❌ Cannot connect to server: {e}")
        return
    
    await run_load_test()
    print("\n🎉 Load test completed!")

if __name__ == "__main__":
    asyncio.run(main()) 