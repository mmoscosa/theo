import asyncio
import time
from datetime import datetime, timedelta
from typing import Dict, Optional, Set, Any
from dataclasses import dataclass
from enum import Enum
import logging
from contextlib import asynccontextmanager

logger = logging.getLogger(__name__)

class ConversationStatus(Enum):
    QUEUED = "queued"
    PROCESSING = "processing"
    COMPLETED = "completed"
    ERROR = "error"
    TIMEOUT = "timeout"

@dataclass
class ConversationState:
    conversation_id: str
    channel: str
    thread_ts: str
    status: ConversationStatus
    agent_type: Optional[str] = None
    start_time: Optional[datetime] = None
    last_updated: Optional[datetime] = None
    retry_count: int = 0
    error_message: Optional[str] = None
    progress_stage: Optional[str] = None
    context: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.context is None:
            self.context = {}
        if self.start_time is None:
            self.start_time = datetime.utcnow()
        if self.last_updated is None:
            self.last_updated = datetime.utcnow()
    
    def update_status(self, status: ConversationStatus, agent_type: Optional[str] = None, error_message: Optional[str] = None):
        self.status = status
        self.last_updated = datetime.utcnow()
        if agent_type:
            self.agent_type = agent_type
        if error_message:
            self.error_message = error_message
    
    def update_progress(self, stage: str):
        self.progress_stage = stage
        self.last_updated = datetime.utcnow()
    
    def increment_retry(self):
        self.retry_count += 1
        self.last_updated = datetime.utcnow()
    
    def is_expired(self, timeout_seconds: int = 60) -> bool:
        if not self.start_time:
            return False
        return datetime.utcnow() - self.start_time > timedelta(seconds=timeout_seconds)

class ConversationManager:
    """Thread-safe conversation state manager"""
    
    def __init__(self, max_concurrent: int = 10, timeout_seconds: int = 60):
        self._conversations: Dict[str, ConversationState] = {}
        self._lock = asyncio.Lock()
        self._semaphore = asyncio.Semaphore(max_concurrent)
        self._timeout_seconds = timeout_seconds
        self._cleanup_task = None
        
        # Progress reaction mapping
        self.progress_reactions = {
            "received": "robot_face",
            "processing": "hourglass_flowing_sand",  # ⏳
            "thinking": "brain",                      # 🧠
            "researching": "mag",                     # 🔍
            "writing": "memo",                        # 📝
            "completing": "white_check_mark",
            "error": "internet-problems"
        }
    
    async def start_cleanup_task(self):
        """Start the background cleanup task"""
        if self._cleanup_task is None:
            self._cleanup_task = asyncio.create_task(self._cleanup_expired_conversations())
    
    async def stop_cleanup_task(self):
        """Stop the background cleanup task"""
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
            self._cleanup_task = None
    
    async def _cleanup_expired_conversations(self):
        """Background task to clean up expired conversations"""
        while True:
            try:
                await asyncio.sleep(30)  # Check every 30 seconds
                async with self._lock:
                    expired_conversations = [
                        conv_id for conv_id, conv in self._conversations.items()
                        if conv.is_expired(self._timeout_seconds)
                    ]
                    
                    for conv_id in expired_conversations:
                        conv = self._conversations.pop(conv_id)
                        conv.update_status(ConversationStatus.TIMEOUT)
                        logger.warning(f"Conversation {conv_id} expired after {self._timeout_seconds}s")
                        
                        # TODO: Send timeout message to Slack
                        # await self._send_timeout_message(conv)
                        
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in cleanup task: {e}")
    
    async def create_conversation(self, conversation_id: str, channel: str, thread_ts: str) -> ConversationState:
        """Create a new conversation state"""
        async with self._lock:
            if conversation_id in self._conversations:
                return self._conversations[conversation_id]
            
            conversation = ConversationState(
                conversation_id=conversation_id,
                channel=channel,
                thread_ts=thread_ts,
                status=ConversationStatus.QUEUED
            )
            self._conversations[conversation_id] = conversation
            logger.info(f"Created conversation {conversation_id}")
            return conversation
    
    async def get_conversation(self, conversation_id: str) -> Optional[ConversationState]:
        """Get conversation state by ID"""
        async with self._lock:
            return self._conversations.get(conversation_id)
    
    async def update_conversation_status(self, conversation_id: str, status: ConversationStatus, 
                                       agent_type: Optional[str] = None, error_message: Optional[str] = None):
        """Update conversation status"""
        async with self._lock:
            if conversation_id in self._conversations:
                self._conversations[conversation_id].update_status(status, agent_type, error_message)
                logger.info(f"Updated conversation {conversation_id} status to {status.value}")
    
    async def update_conversation_progress(self, conversation_id: str, stage: str):
        """Update conversation progress stage"""
        async with self._lock:
            if conversation_id in self._conversations:
                self._conversations[conversation_id].update_progress(stage)
                logger.debug(f"Updated conversation {conversation_id} progress to {stage}")
    
    async def complete_conversation(self, conversation_id: str):
        """Mark conversation as completed and clean up"""
        async with self._lock:
            if conversation_id in self._conversations:
                self._conversations[conversation_id].update_status(ConversationStatus.COMPLETED)
                # We can keep completed conversations for a short time for debugging
                logger.info(f"Completed conversation {conversation_id}")
    
    async def fail_conversation(self, conversation_id: str, error_message: str):
        """Mark conversation as failed"""
        async with self._lock:
            if conversation_id in self._conversations:
                self._conversations[conversation_id].update_status(ConversationStatus.ERROR, error_message=error_message)
                logger.error(f"Failed conversation {conversation_id}: {error_message}")
    
    async def increment_retry(self, conversation_id: str) -> int:
        """Increment retry count and return new count"""
        async with self._lock:
            if conversation_id in self._conversations:
                self._conversations[conversation_id].increment_retry()
                return self._conversations[conversation_id].retry_count
            return 0
    
    @asynccontextmanager
    async def acquire_slot(self):
        """Acquire a processing slot (semaphore)"""
        await self._semaphore.acquire()
        try:
            yield
        finally:
            self._semaphore.release()
    
    async def get_active_conversations(self) -> Dict[str, ConversationState]:
        """Get all currently active conversations"""
        async with self._lock:
            return {
                conv_id: conv for conv_id, conv in self._conversations.items()
                if conv.status in [ConversationStatus.QUEUED, ConversationStatus.PROCESSING]
            }
    
    async def get_conversation_stats(self) -> Dict[str, int]:
        """Get conversation statistics"""
        async with self._lock:
            stats = {
                "total": len(self._conversations),
                "queued": 0,
                "processing": 0,
                "completed": 0,
                "error": 0,
                "timeout": 0
            }
            
            for conv in self._conversations.values():
                stats[conv.status.value] += 1
            
            return stats

class SlackReactionManager:
    """Thread-safe Slack reaction manager to prevent conflicts"""
    
    def __init__(self):
        self._active_reactions: Dict[tuple, Set[str]] = {}  # {(channel, thread_ts): {"hourglass", "loading"}}
        self._lock = asyncio.Lock()
    
    async def add_reaction_safe(self, channel: str, thread_ts: str, reaction: str) -> bool:
        """Add reaction if not already present"""
        async with self._lock:
            key = (channel, thread_ts)
            if key not in self._active_reactions:
                self._active_reactions[key] = set()
            
            if reaction in self._active_reactions[key]:
                logger.debug(f"Reaction {reaction} already active for {thread_ts}")
                return False
            
            # Add to tracking before making API call
            self._active_reactions[key].add(reaction)
        
        try:
            # Import here to avoid circular imports
            from theo.tools.slack import add_slack_reaction
            success = add_slack_reaction(channel, thread_ts, reaction)
            if not success:
                # Remove from tracking if API call failed
                async with self._lock:
                    self._active_reactions[key].discard(reaction)
            return success
        except Exception as e:
            logger.error(f"Error adding reaction {reaction}: {e}")
            # Remove from tracking if API call failed
            async with self._lock:
                self._active_reactions[key].discard(reaction)
            return False
    
    async def remove_reaction_safe(self, channel: str, thread_ts: str, reaction: str) -> bool:
        """Remove reaction if present"""
        async with self._lock:
            key = (channel, thread_ts)
            if key not in self._active_reactions or reaction not in self._active_reactions[key]:
                logger.debug(f"Reaction {reaction} not tracked for {thread_ts}")
                return False
            
            # Remove from tracking before making API call
            self._active_reactions[key].discard(reaction)
        
        try:
            # Import here to avoid circular imports
            from theo.tools.slack import remove_slack_reaction
            return remove_slack_reaction(channel, thread_ts, reaction)
        except Exception as e:
            logger.error(f"Error removing reaction {reaction}: {e}")
            return False
    
    async def update_progress_reaction(self, channel: str, thread_ts: str, old_stage: Optional[str], new_stage: str, progress_reactions: Dict[str, str]):
        """Update progress by removing old reaction and adding new one"""
        if old_stage and old_stage in progress_reactions:
            await self.remove_reaction_safe(channel, thread_ts, progress_reactions[old_stage])
        
        if new_stage in progress_reactions:
            await self.add_reaction_safe(channel, thread_ts, progress_reactions[new_stage])
    
    async def remove_all_progress_reactions(self, channel: str, thread_ts: str, progress_reactions: Dict[str, str]):
        """Remove all progress-related reactions (for completion) but keep robot_face"""
        progress_emojis = ["hourglass_flowing_sand", "brain", "mag", "memo"]  # ⏳ 🧠 🔍 📝
        
        for emoji in progress_emojis:
            await self.remove_reaction_safe(channel, thread_ts, emoji)
        
        logger.debug(f"Removed all progress reactions for {thread_ts}, robot_face remains as independent ownership indicator")
    
    async def cleanup_conversation_reactions(self, channel: str, thread_ts: str):
        """Clean up all reactions for a conversation"""
        async with self._lock:
            key = (channel, thread_ts)
            if key in self._active_reactions:
                reactions = self._active_reactions[key].copy()
                del self._active_reactions[key]
            else:
                reactions = set()
        
        # Remove all tracked reactions
        for reaction in reactions:
            try:
                from theo.tools.slack import remove_slack_reaction
                success = remove_slack_reaction(channel, thread_ts, reaction)
                if success:
                    logger.debug(f"Cleaned up reaction {reaction} for {thread_ts}")
                else:
                    logger.warning(f"Failed to clean up reaction {reaction} for {thread_ts}")
            except Exception as e:
                logger.error(f"Error cleaning up reaction {reaction}: {e}")
        
        logger.info(f"Finished cleaning up {len(reactions)} reactions for conversation {thread_ts}")

# Global instances
conversation_manager = ConversationManager(max_concurrent=10, timeout_seconds=60)
reaction_manager = SlackReactionManager()

async def with_retry(func, max_retries: int = 2, conversation_id: Optional[str] = None):
    """Retry wrapper for async functions"""
    last_exception = None
    
    for attempt in range(max_retries + 1):
        try:
            return await func()
        except Exception as e:
            last_exception = e
            if conversation_id:
                retry_count = await conversation_manager.increment_retry(conversation_id)
                logger.warning(f"Attempt {attempt + 1} failed for conversation {conversation_id}: {e}")
                
                if attempt < max_retries:
                    await asyncio.sleep(1)  # Brief delay between retries
            else:
                logger.warning(f"Attempt {attempt + 1} failed: {e}")
                if attempt < max_retries:
                    await asyncio.sleep(1)
    
    # All retries exhausted
    if conversation_id:
        await conversation_manager.fail_conversation(conversation_id, str(last_exception))
    raise last_exception

async def timeout_wrapper(coro, timeout_seconds: int = 60):
    """Wrapper to add timeout to any coroutine"""
    try:
        return await asyncio.wait_for(coro, timeout=timeout_seconds)
    except asyncio.TimeoutError:
        logger.error(f"Operation timed out after {timeout_seconds} seconds")
        raise 