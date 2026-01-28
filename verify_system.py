import requests
import time
import json
import sys

BASE_URL = "http://localhost:8000/api/v1"

def print_step(msg):
    print(f"\n[STEP] {msg}")

def print_success(msg):
    print(f"✅ {msg}")

def print_fail(msg):
    print(f"❌ {msg}")

def test_system():
    print("🚀 Starting System Verification...")
    
    # 1. Ingest specific knowledge
    print_step("Ingesting Fact: 'The code name for Project X is Chimera'")
    try:
        res = requests.post(f"{BASE_URL}/ingest/document", json={
            "content": "CONFIDENTIAL: The internal code name for Project X is Chimera. It uses a Rust backend.",
            "source_type": "document",
            "doc_metadata": {"level": "top_secret"}
        })
        res.raise_for_status()
        print_success(f"Ingestion successful: {res.json()}")
    except Exception as e:
        print_fail(f"Ingestion failed: {e}")
        return

    # Wait for processing (if async)
    time.sleep(1)

    # 2. Query Agent (Testing RAG)
    print_step("Asking Agent: 'What is the code name for Project X?'")
    try:
        res = requests.post(f"{BASE_URL}/agent/", json={
            "query": "What is the code name for Project X?",
            "token_budget": 1000
        })
        res.raise_for_status()
        data = res.json()
        response_text = data["response"]
        print(f"🤖 Agent Response: {response_text}")
        
        if "Chimera" in response_text:
            print_success("Agent retrieved correct context!")
        else:
            print_fail("Agent failed to retrieve context.")
            
    except Exception as e:
        print_fail(f"Agent query failed: {e}")
        return

    # 3. Verify Learning (Testing Post-Interaction Loop)
    print_step("Verifying Post-Interaction Learning...")
    # The learner should have picked up that we discussed Project X/Chimera
    # and maybe stored a semantic fact or episode.
    
    time.sleep(2) # Give background task time to run (it's awaited in MVP but good practice)
    
    try:
        # Check recent memories for source_type='learned_fact' or 'episode'
        res = requests.get(f"{BASE_URL}/memory/recent?limit=5")
        res.raise_for_status()
        memories = res.json()
        
        found_learning = False
        for mem in memories:
            # We look for memories created typically AFTER our ingestion
            # source_type might be 'learned_fact' or 'episode'
            if mem["source_type"] in ["learned_fact", "episode"] and "Project X" in mem["content"]:
                found_learning = True
                print_success(f"Found new memory: [{mem['source_type']}] {mem['content'][:50]}...")
                break
        
        if not found_learning:
            print_fail("No new 'learned_fact' or 'episode' found in recent memories.")
            print("Recent memories found:", json.dumps(memories, indent=2))
            
    except Exception as e:
        print_fail(f"Memory check failed: {e}")

if __name__ == "__main__":
    try:
        test_system()
    except KeyboardInterrupt:
        print("\nAborted.")
