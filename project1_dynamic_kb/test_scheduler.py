import time
from apscheduler.schedulers.background import BackgroundScheduler
from ingestion import get_all_chunks
from vector_store import add_chunks_to_store

def update_knowledge_base():
    print(f"\n[Scheduler] Checking for updates...")
    chunks = get_all_chunks()
    add_chunks_to_store(chunks)
    print(f"[Scheduler] Knowledge base refreshed with {len(chunks)} chunks.\n")

scheduler = BackgroundScheduler()
scheduler.add_job(update_knowledge_base, trigger="interval",seconds=10)
scheduler.start()

print("Scheduler running. Add a new .txt file to data/sources/now and watch it get picked up. ")
print("Press Ctrl+C to stop.\n")

try:
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    scheduler.shutdown()
    print("Scheduler stopped.")