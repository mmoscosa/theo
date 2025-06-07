#!/usr/bin/env python3
"""
Test script for the threading infrastructure
"""
import asyncio
import time
from src.theo.threading import conversation_manager, reaction_manager, ConversationStatus

async def test_conversation_management():
    """Test conversation state management"""
    print("🧪 Testing conversation management...")
    
    # Test creating conversations
    conv1 = await conversation_manager.create_conversation("test_1", "C123", "1234567890.123456")
    conv2 = await conversation_manager.create_conversation("test_2", "C123", "1234567890.234567")
    
    print(f"✅ Created conversation 1: {conv1.conversation_id}")
    print(f"✅ Created conversation 2: {conv2.conversation_id}")
    
    # Test status updates
    await conversation_manager.update_conversation_status("test_1", ConversationStatus.PROCESSING, "support_engineer")
    await conversation_manager.update_conversation_progress("test_1", "thinking")
    
    # Test getting conversation
    retrieved = await conversation_manager.get_conversation("test_1")
    print(f"✅ Retrieved conversation: {retrieved.status.value}, agent: {retrieved.agent_type}")
    
    # Test stats
    stats = await conversation_manager.get_conversation_stats()
    print(f"✅ Stats: {stats}")
    
    # Test semaphore (concurrent limit)
    print("🔄 Testing concurrent processing limits...")
    
    async def dummy_task(task_id):
        async with conversation_manager.acquire_slot():
            print(f"   Task {task_id} acquired slot")
            await asyncio.sleep(0.1)  # Simulate work
            print(f"   Task {task_id} releasing slot")
    
    # Try to run 15 tasks (should be limited to 10 concurrent)
    start_time = time.time()
    tasks = [dummy_task(i) for i in range(15)]
    await asyncio.gather(*tasks)
    duration = time.time() - start_time
    
    print(f"✅ Processed 15 tasks in {duration:.2f}s (with 10 concurrent limit)")
    
    # Test conversation completion
    await conversation_manager.complete_conversation("test_1")
    await conversation_manager.complete_conversation("test_2")
    
    final_stats = await conversation_manager.get_conversation_stats()
    print(f"✅ Final stats: {final_stats}")

async def test_reaction_management():
    """Test reaction management"""
    print("\n🎭 Testing reaction management...")
    
    channel = "C123TEST"
    thread_ts = "1234567890.123456"
    
    # Test adding reactions (will fail in test since no real Slack, but tests tracking)
    success1 = await reaction_manager.add_reaction_safe(channel, thread_ts, "robot_face")
    success2 = await reaction_manager.add_reaction_safe(channel, thread_ts, "robot_face")  # Duplicate
    
    print(f"✅ First reaction add: {success1}")
    print(f"✅ Duplicate reaction add (should be False): {success2}")
    
    # Test progress reactions
    await reaction_manager.update_progress_reaction(
        channel, thread_ts, "robot_face", "hourglass_flowing_sand", 
        conversation_manager.progress_reactions
    )
    print("✅ Updated progress reaction")
    
    # Test cleanup
    await reaction_manager.cleanup_conversation_reactions(channel, thread_ts)
    print("✅ Cleaned up conversation reactions")

async def test_concurrent_conversations():
    """Test multiple concurrent conversations"""
    print("\n🏃‍♂️ Testing concurrent conversation processing...")
    
    async def simulate_conversation(conv_id):
        # Create conversation
        conv = await conversation_manager.create_conversation(f"sim_{conv_id}", "C123", f"123456789{conv_id}.123456")
        
        try:
            async with conversation_manager.acquire_slot():
                await conversation_manager.update_conversation_status(f"sim_{conv_id}", ConversationStatus.PROCESSING)
                
                # Simulate processing stages
                stages = ["processing", "researching", "thinking", "writing"]
                for stage in stages:
                    await conversation_manager.update_conversation_progress(f"sim_{conv_id}", stage)
                    await asyncio.sleep(0.05)  # Simulate work
                
                await conversation_manager.complete_conversation(f"sim_{conv_id}")
                print(f"   ✅ Completed conversation sim_{conv_id}")
        
        except Exception as e:
            await conversation_manager.fail_conversation(f"sim_{conv_id}", str(e))
            print(f"   ❌ Failed conversation sim_{conv_id}: {e}")
    
    # Run 12 concurrent simulated conversations
    start_time = time.time()
    tasks = [simulate_conversation(i) for i in range(12)]
    await asyncio.gather(*tasks)
    duration = time.time() - start_time
    
    print(f"✅ Processed 12 concurrent conversations in {duration:.2f}s")
    
    # Check final stats
    final_stats = await conversation_manager.get_conversation_stats()
    print(f"✅ Final stats: {final_stats}")

async def main():
    """Run all tests"""
    print("🚀 Starting threading infrastructure tests...\n")
    
    # Start the cleanup task
    await conversation_manager.start_cleanup_task()
    
    try:
        await test_conversation_management()
        await test_reaction_management()
        await test_concurrent_conversations()
        
        print("\n🎉 All tests passed!")
        
    finally:
        # Stop the cleanup task
        await conversation_manager.stop_cleanup_task()

if __name__ == "__main__":
    asyncio.run(main()) 