#!/bin/bash
# Setup script for Fraud Detection System

set -e

echo "=================================================="
echo "🚀 Fraud Detection System - Setup Script"
echo "=================================================="

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if Python is installed
echo -e "\n${YELLOW}📦 Checking Python installation...${NC}"
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}❌ Python3 is not installed. Please install Python 3.9+${NC}"
    exit 1
fi

PYTHON_VERSION=$(python3 --version | awk '{print $2}')
echo -e "${GREEN}✅ Python $PYTHON_VERSION found${NC}"

# Check if Docker is installed
echo -e "\n${YELLOW}🐳 Checking Docker installation...${NC}"
if ! command -v docker &> /dev/null; then
    echo -e "${RED}❌ Docker is not installed. Please install Docker${NC}"
    exit 1
fi

DOCKER_VERSION=$(docker --version | awk '{print $3}' | sed 's/,//')
echo -e "${GREEN}✅ Docker $DOCKER_VERSION found${NC}"

# Check if Docker Compose is installed
echo -e "\n${YELLOW}🐳 Checking Docker Compose installation...${NC}"
if ! command -v docker-compose &> /dev/null; then
    echo -e "${RED}❌ Docker Compose is not installed. Please install Docker Compose${NC}"
    exit 1
fi

COMPOSE_VERSION=$(docker-compose --version | awk '{print $4}' | sed 's/,//')
echo -e "${GREEN}✅ Docker Compose $COMPOSE_VERSION found${NC}"

# Create virtual environment
echo -e "\n${YELLOW}🔧 Creating Python virtual environment...${NC}"
if [ ! -d "venv" ]; then
    python3 -m venv venv
    echo -e "${GREEN}✅ Virtual environment created${NC}"
else
    echo -e "${GREEN}✅ Virtual environment already exists${NC}"
fi

# Activate virtual environment
echo -e "\n${YELLOW}🔧 Activating virtual environment...${NC}"
source venv/bin/activate
echo -e "${GREEN}✅ Virtual environment activated${NC}"

# Upgrade pip
echo -e "\n${YELLOW}📦 Upgrading pip...${NC}"
pip install --upgrade pip > /dev/null 2>&1
echo -e "${GREEN}✅ pip upgraded${NC}"

# Install Python dependencies
echo -e "\n${YELLOW}📦 Installing Python dependencies...${NC}"
pip install -r requirements.txt
echo -e "${GREEN}✅ Python dependencies installed${NC}"

# Start Docker services
echo -e "\n${YELLOW}🐳 Starting Docker services (Kafka, Zookeeper, MongoDB)...${NC}"
cd docker
docker-compose up -d
cd ..
echo -e "${GREEN}✅ Docker services started${NC}"

# Wait for services to be ready
echo -e "\n${YELLOW}⏳ Waiting for services to be ready (30 seconds)...${NC}"
sleep 30

# Check Kafka
echo -e "\n${YELLOW}🔍 Checking Kafka...${NC}"
if docker ps | grep -q fraud-kafka; then
    echo -e "${GREEN}✅ Kafka is running${NC}"
else
    echo -e "${RED}❌ Kafka is not running${NC}"
fi

# Check MongoDB
echo -e "\n${YELLOW}🔍 Checking MongoDB...${NC}"
if docker ps | grep -q fraud-mongodb; then
    echo -e "${GREEN}✅ MongoDB is running${NC}"
else
    echo -e "${RED}❌ MongoDB is not running${NC}"
fi

# Create Kafka topic
echo -e "\n${YELLOW}📡 Creating Kafka topic...${NC}"
docker exec fraud-kafka kafka-topics --create \
    --topic fraud-transactions \
    --bootstrap-server localhost:9092 \
    --replication-factor 1 \
    --partitions 3 \
    --if-not-exists
echo -e "${GREEN}✅ Kafka topic created${NC}"

# Summary
echo -e "\n=================================================="
echo -e "${GREEN}✅ Setup Complete!${NC}"
echo -e "=================================================="
echo -e "\n${YELLOW}📋 Next Steps:${NC}"
echo -e "  1. Train the ML model:"
echo -e "     ${GREEN}python src/ml_model/train_model.py${NC}"
echo -e "\n  2. Start the Kafka producer:"
echo -e "     ${GREEN}python src/producer/kafka_producer.py${NC}"
echo -e "\n  3. Start the Spark consumer (in another terminal):"
echo -e "     ${GREEN}spark-submit --packages org.apache.spark:spark-sql-kafka-0-10_2.12:3.4.0,org.mongodb.spark:mongo-spark-connector_2.12:10.2.0 src/consumer/spark_consumer.py${NC}"
echo -e "\n  4. Launch the dashboard (in another terminal):"
echo -e "     ${GREEN}streamlit run src/dashboard/app.py${NC}"
echo -e "\n${YELLOW}🌐 Access Points:${NC}"
echo -e "  - Dashboard:      http://localhost:8501"
echo -e "  - Mongo Express:  http://localhost:8081 (admin/admin)"
echo -e "\n${YELLOW}🛠️  Management:${NC}"
echo -e "  - Stop services:  ${GREEN}cd docker && docker-compose down${NC}"
echo -e "  - View logs:      ${GREEN}cd docker && docker-compose logs -f${NC}"
echo -e "\n=================================================="
