from apscheduler.schedulers.background import BackgroundScheduler
from ingestion import get_all_chunks
from vector_store import add_chunks_to_store

def update_knowledge_base():
    print("\n[Scheduler] Checking for updates...")
    chunks = get_all_chunks()
    add_chunks_to_store(chunks)
    print(f"[Scheduler] Knowledge base refreshed with {len(chunks)} chunks.\n")

def start_scheduler():
    scheduler = BackgroundScheduler()
    scheduler.add_job(update_knowledge_base, trigger="interval", seconds=10)
    scheduler.start()
    return scheduler