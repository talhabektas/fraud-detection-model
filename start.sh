#!/bin/bash
# Quick start script - Runs all components

set -e

echo "=================================================="
echo "🚀 Fraud Detection System - Quick Start"
echo "=================================================="

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Check if model exists
if [ ! -f "src/ml_model/model.pkl" ]; then
    echo -e "${YELLOW}⚠️  Model not found. Training model first...${NC}"
    python src/ml_model/train_model.py
    echo -e "${GREEN}✅ Model trained${NC}"
fi

# Function to run in background
run_bg() {
    echo -e "${YELLOW}$2${NC}"
    $1 > "logs/$3.log" 2>&1 &
    echo $! > "logs/$3.pid"
    echo -e "${GREEN}✅ Started (PID: $(cat logs/$3.pid))${NC}"
}

# Create logs directory
mkdir -p logs

# Start producer
run_bg "python src/producer/kafka_producer.py --delay 0.5 --limit 1000" \
       "📤 Starting Kafka Producer..." \
       "producer"

# Wait a bit
sleep 5

# Start consumer
run_bg "spark-submit --packages org.apache.spark:spark-sql-kafka-0-10_2.12:3.4.0,org.mongodb.spark:mongo-spark-connector_2.12:10.2.0 src/consumer/spark_consumer.py" \
       "⚡ Starting Spark Consumer..." \
       "consumer"

# Wait a bit
sleep 5

# Start dashboard
run_bg "streamlit run src/dashboard/app.py" \
       "📊 Starting Dashboard..." \
       "dashboard"

echo -e "\n=================================================="
echo -e "${GREEN}✅ All services started!${NC}"
echo -e "=================================================="
echo -e "\n${YELLOW}📋 Process IDs:${NC}"
echo -e "  Producer:  $(cat logs/producer.pid)"
echo -e "  Consumer:  $(cat logs/consumer.pid)"
echo -e "  Dashboard: $(cat logs/dashboard.pid)"
echo -e "\n${YELLOW}📊 Dashboard:${NC} http://localhost:8501"
echo -e "\n${YELLOW}📝 Logs:${NC}"
echo -e "  Producer:  tail -f logs/producer.log"
echo -e "  Consumer:  tail -f logs/consumer.log"
echo -e "  Dashboard: tail -f logs/dashboard.log"
echo -e "\n${YELLOW}🛑 Stop all:${NC} ./stop.sh"
echo -e "=================================================="
