#!/bin/bash
# Stop all running services

echo "=================================================="
echo "🛑 Stopping Fraud Detection System"
echo "=================================================="

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Function to stop process
stop_process() {
    if [ -f "logs/$1.pid" ]; then
        PID=$(cat "logs/$1.pid")
        if ps -p $PID > /dev/null 2>&1; then
            echo -e "${YELLOW}Stopping $2 (PID: $PID)...${NC}"
            kill $PID
            rm "logs/$1.pid"
            echo -e "${GREEN}✅ Stopped${NC}"
        else
            echo -e "${YELLOW}⚠️  $2 not running${NC}"
            rm "logs/$1.pid"
        fi
    else
        echo -e "${YELLOW}⚠️  No PID file for $2${NC}"
    fi
}

# Stop services
stop_process "producer" "Producer"
stop_process "consumer" "Consumer"
stop_process "dashboard" "Dashboard"

echo -e "\n${GREEN}✅ All services stopped${NC}"
echo "=================================================="
