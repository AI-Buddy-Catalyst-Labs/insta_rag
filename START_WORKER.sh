#!/bin/bash

# Celery Worker Startup Script for Insta RAG
# Starts a Celery worker for async document ingestion tasks

set -e

echo "=========================================="
echo "Insta RAG - Celery Worker Startup"
echo "=========================================="

# Check if .env file exists
if [ ! -f ".env" ]; then
    echo "❌ Error: .env file not found!"
    echo ""
    echo "Please create a .env file with the following variables:"
    echo "  CELERY_BROKER_URL=redis://default:...@52.140.76.45:6379/0"
    echo "  CELERY_RESULT_BACKEND=redis://default:...@52.140.76.45:6379/1"
    echo "  NEO4J_URI=bolt://..."
    echo "  NEO4J_USER=neo4j"
    echo "  NEO4J_PASSWORD=..."
    exit 1
fi

# Load environment variables
export $(cat .env | xargs)

# Test Redis connection
echo "✓ Testing Redis connection..."
python3 << 'EOF'
import socket
import sys

redis_host = "52.140.76.45"
redis_port = 6379

try:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(5)
    result = sock.connect_ex((redis_host, redis_port))

    if result == 0:
        print("✓ Redis connection successful")
        sock.close()
        sys.exit(0)
    else:
        print("❌ Cannot connect to Redis")
        sys.exit(1)
except Exception as e:
    print(f"❌ Redis connection failed: {e}")
    sys.exit(1)
EOF

if [ $? -ne 0 ]; then
    exit 1
fi

# Parse arguments
CONCURRENCY=4
LOG_LEVEL="info"
QUEUE="default"

while [[ $# -gt 0 ]]; do
    case $1 in
        -c|--concurrency)
            CONCURRENCY="$2"
            shift 2
            ;;
        -l|--log-level)
            LOG_LEVEL="$2"
            shift 2
            ;;
        -q|--queue)
            QUEUE="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: ./START_WORKER.sh [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  -c, --concurrency    Number of concurrent tasks (default: 4)"
            echo "  -l, --log-level      Logging level: debug, info, warning (default: info)"
            echo "  -q, --queue          Queue to process (default: default)"
            echo "  -h, --help           Show this help message"
            echo ""
            echo "Examples:"
            echo "  ./START_WORKER.sh"
            echo "  ./START_WORKER.sh --concurrency 8"
            echo "  ./START_WORKER.sh --log-level debug --concurrency 2"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

echo "=========================================="
echo "Worker Configuration:"
echo "  Concurrency: $CONCURRENCY tasks"
echo "  Log Level: $LOG_LEVEL"
echo "  Queue: $QUEUE"
echo "=========================================="
echo ""

# Start Celery worker
echo "✓ Starting Celery worker..."
echo ""
echo "Press Ctrl+C to stop the worker"
echo "=========================================="
echo ""

celery -A insta_rag.celery_app worker \
    -l $LOG_LEVEL \
    -c $CONCURRENCY \
    -Q $QUEUE \
    --loglevel=$LOG_LEVEL
