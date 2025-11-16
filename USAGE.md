# 📖 Fraud Detection System - Usage Guide

## 🚀 Quick Start (5 Minutes)

### 1. Initial Setup (One-time)

```bash
# Make setup script executable and run
chmod +x setup.sh
./setup.sh
```

This will:
- ✅ Check Python, Docker, Docker Compose
- ✅ Create virtual environment
- ✅ Install dependencies
- ✅ Start Kafka, Zookeeper, MongoDB
- ✅ Create Kafka topic

### 2. Train ML Model

```bash
# Activate virtual environment
source venv/bin/activate

# Train Random Forest model
python src/ml_model/train_model.py

# Or train XGBoost
python src/ml_model/train_model.py --model xgboost
```

### 3. Start All Services

Option A - **Automated** (Recommended):
```bash
./start.sh
```

Option B - **Manual** (for development):

Terminal 1 - Kafka Producer:
```bash
source venv/bin/activate
python src/producer/kafka_producer.py
```

Terminal 2 - Spark Consumer:
```bash
source venv/bin/activate
spark-submit --packages org.apache.spark:spark-sql-kafka-0-10_2.12:3.4.0,org.mongodb.spark:mongo-spark-connector_2.12:10.2.0 \
  src/consumer/spark_consumer.py
```

Terminal 3 - Dashboard:
```bash
source venv/bin/activate
streamlit run src/dashboard/app.py
```

### 4. Access Dashboard

Open browser: **http://localhost:8501**

### 5. Stop Services

```bash
./stop.sh

# Stop Docker services
cd docker && docker-compose down
```

---

## 📊 Detailed Usage

### ML Model Training

#### Train with SMOTE (Default)
```bash
python src/ml_model/train_model.py --model random_forest
```

#### Train without SMOTE
```bash
python src/ml_model/train_model.py --model random_forest --no-smote
```

#### Train XGBoost
```bash
python src/ml_model/train_model.py --model xgboost
```

**Output:**
- `src/ml_model/model.pkl` - Trained model
- `src/ml_model/scaler.pkl` - Fitted scaler
- `src/ml_model/feature_importance_*.png` - Feature importance plot

### Kafka Producer Options

```bash
# Send all transactions with 0.1s delay
python src/producer/kafka_producer.py

# Send 1000 transactions with 0.5s delay
python src/producer/kafka_producer.py --limit 1000 --delay 0.5

# Send without shuffling
python src/producer/kafka_producer.py --no-shuffle

# Custom Kafka broker and topic
python src/producer/kafka_producer.py --broker localhost:9092 --topic my-topic

# Batch size for progress updates
python src/producer/kafka_producer.py --batch-size 50
```

### Spark Consumer Options

```bash
# Default settings
spark-submit --packages org.apache.spark:spark-sql-kafka-0-10_2.12:3.4.0,org.mongodb.spark:mongo-spark-connector_2.12:10.2.0 \
  src/consumer/spark_consumer.py

# Custom Kafka broker
spark-submit --packages org.apache.spark:spark-sql-kafka-0-10_2.12:3.4.0,org.mongodb.spark:mongo-spark-connector_2.12:10.2.0 \
  src/consumer/spark_consumer.py --kafka-broker localhost:9092

# Custom MongoDB URI
spark-submit --packages org.apache.spark:spark-sql-kafka-0-10_2.12:3.4.0,org.mongodb.spark:mongo-spark-connector_2.12:10.2.0 \
  src/consumer/spark_consumer.py --mongodb-uri mongodb://user:pass@localhost:27017

# Custom model path
spark-submit --packages org.apache.spark:spark-sql-kafka-0-10_2.12:3.4.0,org.mongodb.spark:mongo-spark-connector_2.12:10.2.0 \
  src/consumer/spark_consumer.py --model src/ml_model/xgboost_model.pkl
```

### Dashboard Configuration

Edit directly in Streamlit sidebar or modify `src/dashboard/app.py`:

- **MongoDB URI**: Connection string
- **Auto Refresh**: Enable/disable auto-refresh
- **Refresh Interval**: Seconds between refreshes
- **Data Limit**: Number of recent transactions to display

---

## 🐳 Docker Management

### Start Services
```bash
cd docker
docker-compose up -d
```

### View Logs
```bash
cd docker
docker-compose logs -f           # All services
docker-compose logs -f kafka     # Kafka only
docker-compose logs -f mongodb   # MongoDB only
```

### Stop Services
```bash
cd docker
docker-compose down
```

### Restart Services
```bash
cd docker
docker-compose restart
```

### Remove All Data
```bash
cd docker
docker-compose down -v  # WARNING: Deletes all MongoDB data
```

---

## 📡 Kafka Management

### Create Topic
```bash
docker exec fraud-kafka kafka-topics --create \
  --topic fraud-transactions \
  --bootstrap-server localhost:9092 \
  --replication-factor 1 \
  --partitions 3
```

### List Topics
```bash
docker exec fraud-kafka kafka-topics --list \
  --bootstrap-server localhost:9092
```

### Describe Topic
```bash
docker exec fraud-kafka kafka-topics --describe \
  --topic fraud-transactions \
  --bootstrap-server localhost:9092
```

### Consume Messages (Debug)
```bash
docker exec fraud-kafka kafka-console-consumer \
  --topic fraud-transactions \
  --bootstrap-server localhost:9092 \
  --from-beginning \
  --max-messages 10
```

### Delete Topic
```bash
docker exec fraud-kafka kafka-topics --delete \
  --topic fraud-transactions \
  --bootstrap-server localhost:9092
```

---

## 🗄️ MongoDB Management

### Access MongoDB Shell
```bash
docker exec -it fraud-mongodb mongosh \
  -u admin -p fraudadmin123 --authenticationDatabase admin
```

### MongoDB Commands
```javascript
// Use database
use fraud_detection

// Count predictions
db.predictions.countDocuments()

// Find fraud cases
db.predictions.find({ prediction: 1 }).limit(10)

// Count fraud vs normal
db.predictions.aggregate([
  { $group: { _id: "$prediction", count: { $sum: 1 } } }
])

// Clear all data
db.predictions.deleteMany({})

// Create index
db.predictions.createIndex({ processing_time: -1 })
```

### MongoDB Express (GUI)
Access: **http://localhost:8081**
- Username: `admin`
- Password: `admin`

---

## 🧪 Testing & Development

### Test Preprocessing
```bash
python src/ml_model/preprocessing.py
```

### Test Producer (Limited)
```bash
python src/producer/kafka_producer.py --limit 100 --delay 0
```

### Jupyter Notebook
```bash
jupyter notebook notebooks/eda.ipynb
```

---

## 📈 Performance Tuning

### Kafka Producer
- **`--delay`**: Lower = faster streaming (e.g., `--delay 0.01`)
- **`--batch-size`**: Adjust for performance monitoring

### Spark Consumer
- **Increase partitions**: More parallel processing
  ```bash
  # In docker exec kafka-topics --create
  --partitions 10
  ```
- **Adjust Spark config**: Edit `spark_consumer.py`
  ```python
  .config("spark.executor.memory", "4g")
  .config("spark.executor.cores", "4")
  ```

### MongoDB
- **Create indexes**:
  ```javascript
  db.predictions.createIndex({ processing_time: -1 })
  db.predictions.createIndex({ prediction: 1 })
  db.predictions.createIndex({ fraud_probability: -1 })
  ```

---

## 🐛 Troubleshooting

### Issue: Kafka connection refused
```bash
# Check if Kafka is running
docker ps | grep kafka

# Check Kafka logs
docker logs fraud-kafka

# Restart Kafka
cd docker && docker-compose restart kafka
```

### Issue: MongoDB connection error
```bash
# Check MongoDB
docker ps | grep mongodb

# Test connection
docker exec fraud-mongodb mongosh -u admin -p fraudadmin123
```

### Issue: Model not found
```bash
# Train model first
python src/ml_model/train_model.py
```

### Issue: Spark errors
```bash
# Check Java version (needs 11+)
java -version

# Check Spark installation
spark-submit --version
```

### Issue: Dashboard not loading data
1. Check MongoDB connection in sidebar
2. Verify consumer is running and processing
3. Check MongoDB has data: `docker exec -it fraud-mongodb mongosh -u admin -p fraudadmin123`

---

## 📊 Expected Results

### Model Performance
- **Accuracy**: ~99%
- **Precision**: ~90-95%
- **Recall**: ~80-90%
- **F1-Score**: ~85-92%
- **ROC-AUC**: ~97-99%

### Streaming Throughput
- **Producer**: ~1000-5000 tx/s (depending on delay)
- **Consumer**: ~500-2000 tx/s (depending on resources)

---

## 🎯 Use Cases

### Demo Mode (Fast)
```bash
# Send 500 transactions very quickly
python src/producer/kafka_producer.py --limit 500 --delay 0.01
```

### Realistic Simulation
```bash
# Send continuously with realistic timing
python src/producer/kafka_producer.py --delay 1
```

### High Volume Test
```bash
# Send all 284K transactions
python src/producer/kafka_producer.py --delay 0
```

---

## 📦 Project Structure Reference

```
fraud/
├── data/
│   └── creditcard.csv              # Dataset (284K transactions)
├── src/
│   ├── producer/
│   │   └── kafka_producer.py       # Streams data to Kafka
│   ├── consumer/
│   │   └── spark_consumer.py       # Consumes, predicts, saves
│   ├── ml_model/
│   │   ├── preprocessing.py        # Data preprocessing
│   │   ├── train_model.py          # Model training
│   │   ├── model.pkl               # Trained model (generated)
│   │   └── scaler.pkl              # Fitted scaler (generated)
│   └── dashboard/
│       └── app.py                  # Streamlit dashboard
├── docker/
│   └── docker-compose.yml          # Kafka, MongoDB setup
├── notebooks/
│   └── eda.ipynb                   # Exploratory analysis
├── logs/                           # Log files
├── setup.sh                        # One-time setup
├── start.sh                        # Start all services
├── stop.sh                         # Stop all services
├── requirements.txt                # Python dependencies
└── README.md                       # Project overview
```

---

## 🎓 Learning Resources

- **Kafka**: https://kafka.apache.org/documentation/
- **Spark Streaming**: https://spark.apache.org/streaming/
- **MongoDB**: https://docs.mongodb.com/
- **Streamlit**: https://docs.streamlit.io/
- **Imbalanced Learning**: https://imbalanced-learn.org/

---

## 💡 Tips

1. **Start small**: Use `--limit 100` first to test
2. **Monitor resources**: Use `docker stats` to check resource usage
3. **Check logs**: Always check logs if something fails
4. **Clean data**: Use MongoDB Express to clear old data
5. **Version control**: Commit after each successful change

---

## 🆘 Support

If you encounter issues:
1. Check logs: `tail -f logs/*.log`
2. Check Docker: `docker ps` and `docker logs <container>`
3. Verify connections: MongoDB, Kafka
4. Retrain model if needed

---

**Good luck with your project! 🚀**
