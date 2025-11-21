# Graph RAG + Celery Implementation Guide

## Complete Guide to Building Scalable Knowledge Graph Ingestion

This guide explains how to implement **Graph RAG with Celery** for production-grade, scalable document processing and knowledge graph construction.

**Table of Contents:**
1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Prerequisites](#prerequisites)
4. [Complete Setup Guide](#complete-setup-guide)
5. [Implementation Patterns](#implementation-patterns)
6. [Production Deployment](#production-deployment)
7. [Troubleshooting](#troubleshooting)
8. [Best Practices](#best-practices)

---

## Overview

### What is Graph RAG + Celery?

**Graph RAG** extracts entities and relationships from documents and stores them in a Neo4j knowledge graph. **Celery** with **Redis** enables non-blocking async processing, allowing you to:

- Submit documents and return immediately with task IDs
- Process documents in the background without blocking your API
- Scale horizontally by running multiple workers
- Monitor task progress in real-time
- Automatically retry failed tasks

### When to Use This Pattern

✅ **Use Graph RAG + Celery if you:**
- Need to process large documents (long entity extraction time)
- Want responsive APIs that don't block on processing
- Need to scale document ingestion horizontally
- Want to monitor individual task progress
- Need automatic retry logic for failed extractions

❌ **Don't use if you:**
- Processing documents is very fast (< 1 second)
- You can afford to block the API during processing
- You don't need task monitoring or retries
- Your deployment is single-machine only

---

## Architecture

### Component Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                        FastAPI Server                        │
├─────────────────────────────────────────────────────────────┤
│  POST /graph-rag/add-documents                              │
│  ├─ GraphRAGClient.submit_add_documents_async()            │
│  └─ Return: { task_id: "abc-123" }                         │
│                                                             │
│  GET /tasks/{task_id}                                      │
│  └─ TaskMonitoring.get_task_status()                       │
└─────────────────────────────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────────┐
│                     Redis (Broker)                           │
├─────────────────────────────────────────────────────────────┤
│  Database /0: Task Queue "default"                          │
│  Database /1: Result Storage (task results)                 │
│                                                             │
│  Message: {                                                 │
│    "task": "insta_rag.tasks.add_documents_task",           │
│    "args": [documents_list, "collection_name"],            │
│    "task_id": "abc-123"                                    │
│  }                                                          │
└─────────────────────────────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────────┐
│               Celery Worker Pool (Processes)                │
├─────────────────────────────────────────────────────────────┤
│  Worker-0  │  Worker-1  │  Worker-2                        │
│  (PID: xxx)│  (PID: yyy)│  (PID: zzz)                      │
│                                                             │
│  1. Poll Redis "default" queue                             │
│  2. Pick up task message                                   │
│  3. Deserialize documents from JSON                        │
│  4. Call add_documents_task(documents, collection)         │
└─────────────────────────────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────────┐
│            GraphRAGClient (Async Context)                   │
├─────────────────────────────────────────────────────────────┤
│  await client.initialize()  → Neo4j connection             │
│  await client.add_documents() → Entity extraction           │
│    ├─ Convert documents to Graphiti episodes              │
│    ├─ Call LLM for entity extraction                       │
│    ├─ Call LLM for relationship extraction                 │
│    └─ Store in Neo4j graph                                 │
│  return result                                             │
└─────────────────────────────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────────┐
│                   Neo4j Graph Database                      │
├─────────────────────────────────────────────────────────────┤
│  Nodes: [Alice, TechCorp, Engineer, ...]                   │
│  Edges: [works_at, builds, builds_for, ...]                │
│                                                             │
│  Sample node: {                                            │
│    name: "Alice",                                          │
│    labels: ["Person"],                                    │
│    properties: { job_title: "Engineer" }                   │
│  }                                                         │
│                                                             │
│  Sample edge: {                                            │
│    fact: "Alice works at TechCorp",                        │
│    relationship_type: "works_at",                          │
│    source: Alice_uuid,                                     │
│    target: TechCorp_uuid                                   │
│  }                                                         │
└─────────────────────────────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────────┐
│          Task Result Stored in Redis Backend                │
├─────────────────────────────────────────────────────────────┤
│  Key: "celery-task-result-abc-123"                         │
│  Value: {                                                   │
│    "status": "success",                                    │
│    "data": {                                               │
│      "nodes_created": 5,                                   │
│      "edges_created": 3,                                   │
│      "collection_name": "company"                          │
│    }                                                       │
│  }                                                         │
└─────────────────────────────────────────────────────────────┘
```

### Request/Response Flow

**Step 1: Client Submits Documents**
```
POST /graph-rag/add-documents
Body: {
  "documents": [
    { "text": "Alice works at TechCorp as an engineer" },
    { "text": "TechCorp builds AI products" }
  ]
}

Response:
{
  "task_id": "3f7a8c2d-1b9f-4e5c-9a1d-2c8f5e9a1b3d",
  "status": "submitted"
}
(API returns immediately - no waiting!)
```

**Step 2: API Endpoint Logic**
```python
async def add_documents(documents: List[DocumentInput]):
    async with GraphRAGClient() as client:
        await client.initialize()

        # Submit to Celery (non-blocking)
        task_id = await client.submit_add_documents_async(
            documents,
            collection_name="documents"
        )

    # Return immediately with task_id
    return {"task_id": task_id, "status": "submitted"}
```

**Step 3: Documents Serialized and Queued**
```python
# GraphRAGClient.submit_add_documents_async()
doc_dicts = [doc.model_dump() for doc in documents]

# Call Celery task with .delay() (non-blocking)
task = add_documents_task.delay(doc_dicts, "documents")

return task.id  # Return task_id immediately
```

**Step 4: Task Placed in Redis Queue**
```
Redis Queue "default":
  [
    {
      "task": "insta_rag.tasks.add_documents_task",
      "args": [[{...documents...}], "documents"],
      "id": "3f7a8c2d-1b9f-4e5c-9a1d-2c8f5e9a1b3d"
    }
  ]
```

**Step 5: Worker Picks Up Task**
```
Worker-0:
  1. Detect new message in Redis queue "default"
  2. Deserialize: { documents: [...], collection_name: "documents" }
  3. Call: add_documents_task([...], "documents")
  4. Status → STARTED (in Redis backend)
```

**Step 6: Task Execution (Neo4j Operations)**
```python
# add_documents_task (sync Celery task)
async def _add_documents_async(documents, collection_name):
    async with GraphRAGClient() as client:
        await client.initialize()

        # Neo4j entity extraction via Graphiti
        result = await client.add_documents(documents, collection_name)

        return result  # { nodes_created: 5, edges_created: 3 }

# Celery task runs async code in event loop
result = loop.run_until_complete(_add_documents_async(...))
return {"status": "success", "data": result}
```

**Step 7: Results Stored in Redis**
```
Redis Backend Database /1:
  celery-task-result-3f7a8c2d-1b9f-4e5c-9a1d-2c8f5e9a1b3d:
    {
      "status": "success",
      "data": {
        "nodes_created": 5,
        "edges_created": 3,
        "collection_name": "company"
      }
    }

  Task status changed to: SUCCESS
```

**Step 8: Client Polls Task Status**
```
GET /tasks/3f7a8c2d-1b9f-4e5c-9a1d-2c8f5e9a1b3d

Response:
{
  "task_id": "3f7a8c2d-1b9f-4e5c-9a1d-2c8f5e9a1b3d",
  "status": "SUCCESS",
  "result": {
    "nodes_created": 5,
    "edges_created": 3,
    "collection_name": "company"
  }
}
```

---

## Prerequisites

### Required Services

1. **Redis** (Message Broker + Result Backend)
   ```bash
   # Docker
   docker run -d -p 6379:6379 redis:7

   # Or install locally
   brew install redis  # macOS
   sudo apt-get install redis-server  # Ubuntu
   ```

2. **Neo4j** (Graph Database)
   ```bash
   # Docker
   docker run -d -p 7687:7687 -e NEO4J_AUTH=neo4j/password neo4j:5

   # Or use managed: Aura Cloud, AuraDB, etc.
   ```

### Python Dependencies

```bash
# Core async processing
celery>=5.3.0
redis>=4.5.0

# HTTP clients
httpcore>=1.0.0
httpx>=0.24.0

# Already in insta_rag
graphiti-core>=0.1.0  # Entity extraction
qdrant-client>=1.7.0  # Vector DB (optional)
openai>=1.12.0        # LLM calls
pydantic>=2.5.0       # Data validation
```

### Environment Setup

```bash
# 1. Install package with all dependencies
uv pip install -e . --group dev

# 2. Create .env file with credentials
cat > testing_api/.env << 'EOF'
# Redis Configuration
CELERY_BROKER_URL=redis://default:password@localhost:6379/0
CELERY_RESULT_BACKEND=redis://default:password@localhost:6379/1

# Neo4j Configuration
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=password
NEO4J_DATABASE=neo4j

# LLM Configuration (for entity extraction)
GRAPHITI_LLM_MODEL=gpt-4.1
GRAPHITI_EMBEDDING_MODEL=text-embedding-3-large
AZURE_OPENAI_API_KEY=your_key
AZURE_OPENAI_ENDPOINT=your_endpoint

# Vector DB (optional)
QDRANT_URL=https://your-qdrant:6333
QDRANT_API_KEY=your_key
EOF

# 3. Verify connectivity
redis-cli ping          # Should return PONG
cypher-shell -u neo4j -p password  # Should connect
```

---

## Complete Setup Guide

### Phase 1: Local Development Setup

#### Step 1: Start Infrastructure

```bash
# Terminal 1: Redis
docker run -d -p 6379:6379 --name redis redis:7

# Terminal 2: Neo4j
docker run -d -p 7687:7687 \
  -e NEO4J_AUTH=neo4j/password \
  --name neo4j neo4j:5

# Terminal 3: Verify connectivity
redis-cli ping
# PONG

# Check Neo4j
curl -u neo4j:password -X GET http://localhost:7474/db/data/
```

#### Step 2: Start Celery Workers

```bash
# Terminal 4: Start single worker
celery -A insta_rag.celery_app worker -l debug -Q default -c 4

# Or start worker pool (multiple workers)
python3 << 'EOF'
from insta_rag import start_worker_pool
start_worker_pool(num_workers=2, concurrency_per_worker=4)

# Keep running...
input("Workers started. Press Enter to stop...")
EOF
```

#### Step 3: Start FastAPI Server

```bash
# Terminal 5: Start API
cd testing_api
uvicorn main:app --reload --host 0.0.0.0 --port 8000

# Swagger available at: http://localhost:8000/docs
```

#### Step 4: Test End-to-End

```bash
# Terminal 6: Test the flow
python3 << 'EOF'
import asyncio
import httpx
import time

async def test_async_flow():
    client = httpx.AsyncClient()

    # 1. Submit documents
    response = await client.post(
        "http://localhost:8000/graph-rag/add-documents",
        json={
            "documents": [
                {"text": "Alice works at TechCorp as a Senior Engineer"},
                {"text": "TechCorp builds AI products for enterprises"}
            ]
        }
    )

    task_id = response.json()["task_id"]
    print(f"✓ Task submitted: {task_id}")

    # 2. Poll task status
    for i in range(30):  # Poll for up to 30 seconds
        response = await client.get(f"http://localhost:8000/tasks/{task_id}")
        task = response.json()

        print(f"  Status: {task['status']}")

        if task["status"] == "SUCCESS":
            print(f"✓ Task completed!")
            print(f"  Result: {task.get('result')}")
            break

        await asyncio.sleep(1)
    else:
        print("✗ Task didn't complete in 30 seconds")

asyncio.run(test_async_flow())
EOF
```

### Phase 2: Production Deployment

#### Option A: Docker Compose

```yaml
# docker-compose.yml
version: '3.8'

services:
  redis:
    image: redis:7
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data
    environment:
      - REDIS_PASSWORD=your_secure_password

  neo4j:
    image: neo4j:5
    ports:
      - "7687:7687"
      - "7474:7474"
    environment:
      - NEO4J_AUTH=neo4j/your_secure_password
    volumes:
      - neo4j_data:/var/lib/neo4j/data

  api:
    build: .
    ports:
      - "8000:8000"
    depends_on:
      - redis
      - neo4j
    environment:
      - CELERY_BROKER_URL=redis://:your_secure_password@redis:6379/0
      - CELERY_RESULT_BACKEND=redis://:your_secure_password@redis:6379/1
      - NEO4J_URI=bolt://neo4j:7687
      - NEO4J_USER=neo4j
      - NEO4J_PASSWORD=your_secure_password
    command: uvicorn testing_api.main:app --host 0.0.0.0 --port 8000

  worker:
    build: .
    depends_on:
      - redis
      - neo4j
    environment:
      - CELERY_BROKER_URL=redis://:your_secure_password@redis:6379/0
      - CELERY_RESULT_BACKEND=redis://:your_secure_password@redis:6379/1
      - NEO4J_URI=bolt://neo4j:7687
      - NEO4J_USER=neo4j
      - NEO4J_PASSWORD=your_secure_password
    command: celery -A insta_rag.celery_app worker -l info -Q default -c 4
    scale: 2  # Run 2 workers

volumes:
  redis_data:
  neo4j_data:
```

Start with: `docker-compose up`

#### Option B: Kubernetes

```yaml
# kubernetes/celery-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: celery-worker
spec:
  replicas: 3  # 3 workers
  selector:
    matchLabels:
      app: celery-worker
  template:
    metadata:
      labels:
        app: celery-worker
    spec:
      containers:
      - name: worker
        image: your-registry/insta-rag:latest
        command:
          - celery
          - -A
          - insta_rag.celery_app
          - worker
          - -l
          - info
          - -Q
          - default
          - -c
          - "4"
        env:
        - name: CELERY_BROKER_URL
          valueFrom:
            secretKeyRef:
              name: app-secrets
              key: celery-broker-url
        - name: CELERY_RESULT_BACKEND
          valueFrom:
            secretKeyRef:
              name: app-secrets
              key: celery-result-backend
        - name: NEO4J_URI
          valueFrom:
            secretKeyRef:
              name: app-secrets
              key: neo4j-uri
        resources:
          limits:
            memory: "2Gi"
            cpu: "1000m"
          requests:
            memory: "1Gi"
            cpu: "500m"
```

---

## Implementation Patterns

### Pattern 1: Simple Non-Blocking Ingestion

**Use Case:** Submit documents and get task ID immediately

```python
@app.post("/ingest")
async def ingest_documents(documents: List[DocumentInput]):
    """Submit documents for async processing."""
    async with GraphRAGClient() as client:
        await client.initialize()
        task_id = await client.submit_add_documents_async(
            documents,
            collection_name="docs"
        )

    return {
        "task_id": task_id,
        "status": "queued",
        "message": f"Processing {len(documents)} documents"
    }
```

**Client Usage:**
```python
# Get task ID immediately
response = await client.post("/ingest", json={"documents": [...]})
task_id = response.json()["task_id"]

# Check status later
while True:
    result = await client.get(f"/tasks/{task_id}")
    if result.json()["status"] == "SUCCESS":
        print(result.json()["result"])
        break
    await asyncio.sleep(1)
```

### Pattern 2: Batch Processing with Progress

**Use Case:** Process many documents in batches with progress tracking

```python
@app.post("/batch-ingest")
async def batch_ingest(batch: BatchRequest):
    """Submit batch of documents for processing."""
    task_ids = []

    # Submit each document as separate task for progress tracking
    async with GraphRAGClient() as client:
        await client.initialize()

        for doc in batch.documents:
            task_id = await client.submit_add_documents_async(
                [doc],
                collection_name=batch.collection_name
            )
            task_ids.append(task_id)

    return {
        "batch_id": str(uuid.uuid4()),
        "task_ids": task_ids,
        "total": len(task_ids),
        "status": "processing"
    }


@app.get("/batch/{batch_id}")
async def get_batch_status(batch_id: str, task_ids: List[str]):
    """Get status of all tasks in batch."""
    monitor = get_task_monitoring()

    completed = 0
    failed = 0
    pending = 0

    results = {}
    for task_id in task_ids:
        status = monitor.get_task_status(task_id)
        results[task_id] = status

        if status == "SUCCESS":
            completed += 1
        elif status == "FAILURE":
            failed += 1
        else:
            pending += 1

    return {
        "batch_id": batch_id,
        "total": len(task_ids),
        "completed": completed,
        "failed": failed,
        "pending": pending,
        "tasks": results
    }
```

### Pattern 3: Auto-Scaling Based on Queue Depth

**Use Case:** Automatically scale workers based on task volume

```python
# In main.py startup
import asyncio
from apscheduler.schedulers.background import BackgroundScheduler

scheduler = BackgroundScheduler()

def check_queue_and_scale():
    """Auto-scale workers based on queue depth."""
    from insta_rag.task_monitoring import get_task_monitoring
    from insta_rag.worker_pool import auto_scale_if_needed

    monitor = get_task_monitoring()
    queue_depth = monitor.get_queue_length()

    print(f"Queue depth: {queue_depth}")

    # Scale based on queue depth
    auto_scale_if_needed(
        queue_depth_threshold=10,  # If > 10 tasks, add workers
        min_workers=1,
        max_workers=8
    )

scheduler.add_job(
    check_queue_and_scale,
    trigger="interval",
    seconds=30  # Check every 30 seconds
)

@app.on_event("startup")
async def startup():
    scheduler.start()
    start_worker_pool(num_workers=2, concurrency_per_worker=4)

@app.on_event("shutdown")
async def shutdown():
    scheduler.shutdown()
    stop_worker_pool()
```

### Pattern 4: Priority Queues

**Use Case:** Process high-priority documents faster

```python
# Multiple queues based on priority
# celery_app.py

app.conf.update(
    task_routes={
        'insta_rag.tasks.add_documents_task': {'queue': 'default'},
    }
)

# Start workers for different queues
# Worker 1 (high priority): celery -A insta_rag.celery_app worker -Q priority
# Worker 2 (default): celery -A insta_rag.celery_app worker -Q default

# Submit to priority queue
@app.post("/ingest/priority")
async def ingest_priority(documents: List[DocumentInput]):
    """Submit high-priority documents."""
    task = add_documents_task.apply_async(
        args=[doc_dicts, collection_name],
        queue='priority'
    )
    return {"task_id": task.id}
```

### Pattern 5: Scheduled/Delayed Processing

**Use Case:** Process documents at scheduled time

```python
@app.post("/schedule-ingest")
async def schedule_ingest(
    documents: List[DocumentInput],
    delay_seconds: int
):
    """Schedule documents for processing after delay."""
    async with GraphRAGClient() as client:
        await client.initialize()
        doc_dicts = [doc.model_dump() for doc in documents]

    # Schedule for processing after delay
    task = add_documents_task.apply_async(
        args=[doc_dicts, "documents"],
        countdown=delay_seconds  # Delay in seconds
    )

    return {
        "task_id": task.id,
        "scheduled_in_seconds": delay_seconds
    }
```

---

## Production Deployment

### Security Checklist

- [ ] **Redis Authentication**
  ```
  CELERY_BROKER_URL=redis://:your_strong_password@host:6379/0
  ```

- [ ] **Redis TLS/SSL**
  ```
  CELERY_BROKER_URL=rediss://:password@host:6379/0
  CELERY_ACCEPT_CONTENT=['json']  # Don't use pickle
  ```

- [ ] **Neo4j Authentication**
  ```
  NEO4J_URI=bolt://neo4j:password@host:7687
  ```

- [ ] **Network Isolation**
  - Redis not exposed to internet
  - Neo4j only accessible from worker pods
  - Only API exposed publicly

- [ ] **Rate Limiting**
  ```python
  from slowapi import Limiter
  from slowapi.util import get_remote_address

  limiter = Limiter(key_func=get_remote_address)

  @app.post("/ingest")
  @limiter.limit("10/minute")  # Max 10 requests per minute
  async def ingest_documents(request: Request, ...):
      pass
  ```

- [ ] **Input Validation**
  ```python
  class DocumentInput(BaseModel):
      text: str = Field(..., max_length=1000000)  # Max 1MB
      metadata: Optional[Dict] = Field(default={}, max_items=10)
  ```

- [ ] **Monitoring & Alerts**
  ```python
  # Monitor task failures
  @app.get("/health/tasks")
  async def task_health():
      monitor = get_task_monitoring()
      failed_count = len(monitor.get_failed_tasks())

      if failed_count > 10:
          # Alert!
          send_alert(f"Too many failed tasks: {failed_count}")

      return {"failed_tasks": failed_count}
  ```

### Performance Tuning

**Redis Configuration:**
```ini
# redis.conf
maxmemory 4gb
maxmemory-policy allkeys-lru
appendonly yes  # Persistence
```

**Celery Configuration:**
```python
# Tune for your workload
app.conf.update(
    # Prefetch tasks
    worker_prefetch_multiplier=1,  # Don't prefetch (better for long tasks)

    # Worker recycling
    worker_max_tasks_per_child=1000,  # Recycle after 1000 tasks

    # Task timeouts
    task_soft_time_limit=3600,  # 1 hour soft limit
    task_time_limit=7200,       # 2 hour hard limit

    # Result expiration
    result_expires=86400,  # Keep results for 24 hours

    # Concurrency
    worker_concurrency=4,  # Depends on CPU/memory
)
```

**Neo4j Performance:**
```cypher
-- Create indices for faster lookups
CREATE INDEX FOR (n:GraphNode) ON (n.name);
CREATE INDEX FOR (n:GraphNode) ON (n.group_id);
CREATE INDEX FOR (e:GraphEdge) ON (e.relationship_type);

-- Monitor slowest queries
PROFILE MATCH (n:GraphNode)-[:works_at]->(c) RETURN n, c;
```

### Monitoring & Observability

**Prometheus Metrics:**
```python
from prometheus_client import Counter, Histogram, Gauge

task_submitted = Counter('celery_task_submitted', 'Tasks submitted', ['task_name'])
task_duration = Histogram('celery_task_duration', 'Task duration', ['task_name'])
worker_count = Gauge('celery_worker_count', 'Active workers')
queue_depth = Gauge('celery_queue_depth', 'Pending tasks')

@app.get("/metrics")
async def metrics():
    from prometheus_client import generate_latest
    return Response(generate_latest(), media_type="text/plain")
```

**Logging:**
```python
import logging
from pythonjsonlogger import jsonlogger

logger = logging.getLogger()
logHandler = logging.StreamHandler()
formatter = jsonlogger.JsonFormatter()
logHandler.setFormatter(formatter)
logger.addHandler(logHandler)

logger.info("Task started", extra={
    "task_id": task_id,
    "collection": collection_name,
    "document_count": len(documents)
})
```

---

## Troubleshooting

### Common Issues

#### Issue 1: Tasks Always PENDING

**Symptoms:** Tasks submitted but never processed (status stays PENDING)

**Causes:**
1. Workers not listening to correct queue
2. Redis not connected
3. Worker not started

**Solution:**
```bash
# Check workers are running on "default" queue
celery -A insta_rag.celery_app inspect active_queues

# Should show:
# worker1@hostname: - default

# If not showing, restart with -Q default
celery -A insta_rag.celery_app worker -l debug -Q default

# Check Redis connectivity
redis-cli ping
# PONG
```

#### Issue 2: "ConnectionError: No Redis"

**Symptoms:** Worker crashes with "Error: No broker"

**Cause:** CELERY_BROKER_URL environment variable not set

**Solution:**
```bash
# Check environment variable
echo $CELERY_BROKER_URL
# Should print Redis URL

# If empty, load from .env
source .env
export CELERY_BROKER_URL

# Or verify .env is in correct location
ls testing_api/.env
```

#### Issue 3: Load_dotenv() Not Loading

**Symptoms:** Redis credentials from .env not found

**Cause:** load_dotenv() called AFTER imports

**Solution:**
```python
# testing_api/main.py - CORRECT ORDER

from dotenv import load_dotenv

# Call FIRST - before any insta_rag imports
load_dotenv()

# Now these imports will find environment variables
from insta_rag import ...
from insta_rag.celery_app import app
```

#### Issue 4: Memory Growing Unbounded

**Symptoms:** Worker memory usage increases over time

**Cause:** Results not expiring from Redis

**Solution:**
```python
# celery_app.py

app.conf.update(
    result_expires=3600,  # Results expire after 1 hour (not 24)
    worker_max_tasks_per_child=500,  # Recycle worker frequently
)

# Or manually clean old results
python3 << 'EOF'
import redis
from datetime import datetime, timedelta

r = redis.Redis.from_url(os.getenv('CELERY_RESULT_BACKEND'))

# Delete results older than 1 hour
for key in r.scan_iter(match='celery-task-result-*'):
    if r.ttl(key) == -1:  # No TTL set
        r.delete(key)
EOF
```

#### Issue 5: Worker Crashes on Large Documents

**Symptoms:** Worker process dies when processing large documents

**Cause:** Memory limit hit during entity extraction

**Solution:**
```python
# Increase worker memory limit
celery -A insta_rag.celery_app worker \
  --max-memory-per-child 500000 \  # 500MB
  -l debug -Q default -c 2

# Or batch large documents
documents_batches = [
    documents[i:i+10]
    for i in range(0, len(documents), 10)
]

for batch in documents_batches:
    task_id = await client.submit_add_documents_async(batch, collection)
    # Wait for completion before next batch
```

---

## Best Practices

### 1. Always Use Async Context Manager

```python
# ✅ GOOD
async with GraphRAGClient() as client:
    await client.initialize()
    task_id = await client.submit_add_documents_async(docs, collection)

# ❌ BAD
client = GraphRAGClient()
await client.initialize()
task_id = await client.submit_add_documents_async(docs, collection)
# No cleanup!
```

### 2. Monitor Task Progress

```python
# ✅ GOOD - Store task IDs and check progress
async def submit_batch():
    task_ids = []
    for doc in documents:
        task_id = await client.submit_add_documents_async([doc], collection)
        task_ids.append(task_id)

        # Log task
        logger.info(f"Submitted task {task_id}")

    return task_ids

# ❌ BAD - Fire and forget
task_id = await client.submit_add_documents_async(documents, collection)
# No tracking of success/failure
```

### 3. Handle Failures Gracefully

```python
# ✅ GOOD - Check status and retry
monitor = get_task_monitoring()
status = monitor.get_task_status(task_id)

if status == "FAILURE":
    logger.error(f"Task {task_id} failed, retrying...")
    # Resubmit
    new_task_id = await client.submit_add_documents_async(docs, collection)

# ❌ BAD - Assume success
# No failure handling
```

### 4. Limit Concurrency Per Task

```python
# ✅ GOOD - Reasonable limits
# Worker with concurrency=2 means max 2 tasks in parallel
celery -A insta_rag.celery_app worker -c 2

# Prevents memory exhaustion

# ❌ BAD - Too high
celery -A insta_rag.celery_app worker -c 100
# Will crash with OOM
```

### 5. Use Appropriate Timeouts

```python
# ✅ GOOD - Reasonable timeout for entity extraction
app.conf.update(
    task_soft_time_limit=3600,  # 1 hour
    task_time_limit=7200,       # 2 hours
)

# ❌ BAD - Too short
app.conf.update(
    task_time_limit=10,  # 10 seconds - not enough for LLM calls
)
```

### 6. Log Comprehensively

```python
# ✅ GOOD - Log all stages
logger.info("Task submitted", extra={"task_id": task_id})
logger.debug("Processing started", extra={"task_id": task_id})
logger.info("Task completed", extra={
    "task_id": task_id,
    "nodes_created": result['nodes_created'],
    "edges_created": result['edges_created']
})

# ❌ BAD - No logging
# Can't debug issues
```

### 7. Clean Up Old Results

```python
# ✅ GOOD - Regular cleanup
@app.on_event("startup")
async def cleanup_task():
    # Clean old results every hour
    async def cleanup_job():
        while True:
            monitor = get_task_monitoring()
            # Results expire automatically based on result_expires config
            await asyncio.sleep(3600)

    asyncio.create_task(cleanup_job())

# ❌ BAD - Results pile up forever
# Redis memory keeps growing
```

---

## Complete Example: End-to-End Implementation

```python
# testing_api/main.py - Complete setup

from fastapi import FastAPI, HTTPException
from dotenv import load_dotenv
import asyncio
import logging

# CRITICAL: Load environment BEFORE imports
load_dotenv()

from insta_rag import (
    DocumentInput,
    start_worker_pool,
    stop_worker_pool,
    GraphRAGClient
)
from insta_rag.task_monitoring import get_task_monitoring

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Graph RAG API")

# Auto-start/stop workers
@app.on_event("startup")
async def startup_event():
    logger.info("Starting worker pool...")
    start_worker_pool(num_workers=2, concurrency_per_worker=4)
    logger.info("✓ Worker pool started")

@app.on_event("shutdown")
async def shutdown_event():
    logger.info("Stopping worker pool...")
    stop_worker_pool()
    logger.info("✓ Worker pool stopped")

# Submit documents asynchronously
@app.post("/graph-rag/add-documents")
async def add_documents(documents: list[DocumentInput]):
    """Submit documents for async processing."""
    try:
        async with GraphRAGClient() as client:
            await client.initialize()

            task_id = await client.submit_add_documents_async(
                documents,
                collection_name="documents"
            )

        logger.info(f"Task submitted: {task_id}")

        return {
            "task_id": task_id,
            "status": "submitted",
            "message": f"Processing {len(documents)} document(s)"
        }

    except Exception as e:
        logger.error(f"Error submitting task: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Check task status
@app.get("/tasks/{task_id}")
async def get_task_status(task_id: str):
    """Get status and results of a task."""
    try:
        monitor = get_task_monitoring()
        status = monitor.get_task_status(task_id)

        response = {
            "task_id": task_id,
            "status": status
        }

        if status == "SUCCESS":
            result = monitor.get_task_result(task_id)
            response["result"] = result
        elif status == "FAILURE":
            error = monitor.get_task_result(task_id)
            response["error"] = str(error)

        return response

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Health check
@app.get("/health")
async def health():
    """Health check endpoint."""
    try:
        monitor = get_task_monitoring()
        queue_depth = monitor.get_queue_length()

        return {
            "status": "healthy",
            "queue_depth": queue_depth
        }
    except Exception as e:
        return {"status": "unhealthy", "error": str(e)}, 503

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

---

## Summary

**Key Takeaways:**

1. **Graph RAG** extracts entities/relationships → Neo4j
2. **Celery** handles async task processing
3. **Redis** stores task queue and results
4. **Workers** process documents in background
5. **Client** gets task ID immediately, polls for results

**Critical Steps:**
- ✅ Load environment BEFORE imports
- ✅ Start workers on "default" queue
- ✅ Use async context manager for GraphRAGClient
- ✅ Store credentials in .env, not code
- ✅ Monitor queue depth and worker health

**Next Steps:**
1. Follow "Complete Setup Guide" to get running locally
2. Deploy with Docker Compose or Kubernetes
3. Implement monitoring and alerting
4. Scale workers based on load
5. Iterate on entity extraction quality

