#!/bin/bash
# Test script - Verify project setup

echo "=================================================="
echo "🧪 Fraud Detection - Project Test"
echo "=================================================="

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

passed=0
failed=0

# Function to test
test_check() {
    if [ $2 -eq 0 ]; then
        echo -e "${GREEN}✅ $1${NC}"
        ((passed++))
    else
        echo -e "${RED}❌ $1${NC}"
        ((failed++))
    fi
}

echo -e "\n${YELLOW}📁 Checking project structure...${NC}"

# Check directories
test -d "src/producer" && test_check "src/producer/ exists" 0 || test_check "src/producer/ exists" 1
test -d "src/consumer" && test_check "src/consumer/ exists" 0 || test_check "src/consumer/ exists" 1
test -d "src/ml_model" && test_check "src/ml_model/ exists" 0 || test_check "src/ml_model/ exists" 1
test -d "src/dashboard" && test_check "src/dashboard/ exists" 0 || test_check "src/dashboard/ exists" 1
test -d "docker" && test_check "docker/ exists" 0 || test_check "docker/ exists" 1
test -d "data" && test_check "data/ exists" 0 || test_check "data/ exists" 1

# Check files
echo -e "\n${YELLOW}📄 Checking Python files...${NC}"
test -f "src/producer/kafka_producer.py" && test_check "kafka_producer.py" 0 || test_check "kafka_producer.py" 1
test -f "src/consumer/spark_consumer.py" && test_check "spark_consumer.py" 0 || test_check "spark_consumer.py" 1
test -f "src/ml_model/preprocessing.py" && test_check "preprocessing.py" 0 || test_check "preprocessing.py" 1
test -f "src/ml_model/train_model.py" && test_check "train_model.py" 0 || test_check "train_model.py" 1
test -f "src/dashboard/app.py" && test_check "dashboard/app.py" 0 || test_check "dashboard/app.py" 1

# Check config files
echo -e "\n${YELLOW}⚙️  Checking configuration files...${NC}"
test -f "docker/docker-compose.yml" && test_check "docker-compose.yml" 0 || test_check "docker-compose.yml" 1
test -f "requirements.txt" && test_check "requirements.txt" 0 || test_check "requirements.txt" 1
test -f "README.md" && test_check "README.md" 0 || test_check "README.md" 1
test -f "USAGE.md" && test_check "USAGE.md" 0 || test_check "USAGE.md" 1

# Check scripts
echo -e "\n${YELLOW}🔧 Checking scripts...${NC}"
test -x "setup.sh" && test_check "setup.sh (executable)" 0 || test_check "setup.sh (executable)" 1
test -x "start.sh" && test_check "start.sh (executable)" 0 || test_check "start.sh (executable)" 1
test -x "stop.sh" && test_check "stop.sh (executable)" 0 || test_check "stop.sh (executable)" 1

# Check dataset
echo -e "\n${YELLOW}📊 Checking dataset...${NC}"
test -f "data/creditcard.csv" && test_check "creditcard.csv" 0 || test_check "creditcard.csv" 1

# Check Python syntax
echo -e "\n${YELLOW}🐍 Checking Python syntax...${NC}"
python3 -m py_compile src/producer/kafka_producer.py 2>/dev/null && test_check "Producer syntax" 0 || test_check "Producer syntax" 1
python3 -m py_compile src/consumer/spark_consumer.py 2>/dev/null && test_check "Consumer syntax" 0 || test_check "Consumer syntax" 1
python3 -m py_compile src/ml_model/preprocessing.py 2>/dev/null && test_check "Preprocessing syntax" 0 || test_check "Preprocessing syntax" 1
python3 -m py_compile src/ml_model/train_model.py 2>/dev/null && test_check "Train model syntax" 0 || test_check "Train model syntax" 1
python3 -m py_compile src/dashboard/app.py 2>/dev/null && test_check "Dashboard syntax" 0 || test_check "Dashboard syntax" 1

# Summary
echo -e "\n=================================================="
echo -e "${YELLOW}📊 Test Summary${NC}"
echo -e "=================================================="
echo -e "  Passed: ${GREEN}$passed${NC}"
echo -e "  Failed: ${RED}$failed${NC}"
echo -e "=================================================="

if [ $failed -eq 0 ]; then
    echo -e "\n${GREEN}✅ All tests passed! Project is ready.${NC}"
    echo -e "${YELLOW}Next steps:${NC}"
    echo -e "  1. ./setup.sh"
    echo -e "  2. python src/ml_model/train_model.py"
    echo -e "  3. ./start.sh"
    exit 0
else
    echo -e "\n${RED}❌ Some tests failed. Please fix the issues.${NC}"
    exit 1
fi
