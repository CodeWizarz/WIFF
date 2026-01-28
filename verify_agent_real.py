import asyncio
import sys
import os
from app.database import init_db, async_session_maker
from app.services.agent import AgentService
from app.services.ingestion import IngestionService

# Force use of SelectorEventLoop on Windows to avoid some Proactor issues with older drivers
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

async def main():
    print("--- Initializing Real components ---")
    try:
        await init_db()
    except Exception as e:
        print(f"Warning: init_db failed (maybe pgvector extension issue?), continuing to see if it works anyway: {e}")
    
    try:
        agent_service = AgentService()
        ingestion_service = IngestionService()
    except Exception as e:
        print(f"Failed to initialize services (env vars missing?): {e}")
        return
    
    async with async_session_maker() as db:
        # 0. Pre-seed some data so retrieval has something to find
        test_content = (
            "In recent astronomical fiction, it has been established that the capital city of Mars "
            "is officially named 'Olympus City'. This city is located at the base of Olympus Mons "
            "and serves as the primary trade hub for the Martian colonies. It was founded in 2085."
        )
        print(f"[Step 0] Seeding DB with test fact...")
        chunks, msg = await ingestion_service.ingest_document(
            db, 
            content=test_content,
            source_type="document",
            metadata={"test": True, "author": "VerificationScript"}
        )
        print(f"         > Result: {msg} (Chunks: {len(chunks)})")

        # 1. Run Agent Flow
        query = "What is the capital of Mars and when was it founded?"
        print(f"\n[Step 1] Running Agent Flow for: '{query}'")
        
        try:
            response_text, context, tokens = await agent_service.run_agent(
                db=db,
                query=query
            )
            
            print("\n[Step 5] Response Received:")
            print(f"Agent: {response_text}")
            
            print(f"\n[Stats] Context Chunks: {len(context)}")
            if context:
                print(f"        Top Chunk: {context[0].content[:50]}...")
                
            # Double check if we got the right answer
            if "Olympus City" in response_text:
                print("\n✅ Verification SUCCESS: Agent retrieved the correct entity.")
            else:
                print("\n❌ Verification WARNING: Agent did not mention Olympus City.")
                
        except Exception as e:
            print(f"\n❌ Execution Failed: {e}")
            import traceback
            traceback.print_exc()

    print("\n--- Verification Complete ---")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
