# 🚨 Real-Time Credit Card Fraud Detection System

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10-blue.svg" alt="Python">
  <img src="https://img.shields.io/badge/Apache_Kafka-7.5.0-red.svg" alt="Kafka">
  <img src="https://img.shields.io/badge/Apache_Spark-3.4.1-orange.svg" alt="Spark">
  <img src="https://img.shields.io/badge/MongoDB-7.0-green.svg" alt="MongoDB">
  <img src="https://img.shields.io/badge/Streamlit-1.27-ff4b4b.svg" alt="Streamlit">
  <img src="https://img.shields.io/badge/License-Educational-yellow.svg" alt="License">
</p>

<p align="center">
  <strong>End-to-End Real-Time Fraud Detection with Streaming ML Pipeline</strong>
</p>

---

## 📊 Overview

Bu proje, **Apache Kafka** ve **Apache Spark Streaming** kullanarak gerçek zamanlı kredi kartı dolandırıcılık tespiti yapan bir veri analitik sistemidir.

**Key Highlights:**
- ⚡ Real-time transaction processing
- 🤖 ML-powered fraud detection (99% accuracy)
- 📊 Live monitoring dashboard
- 🐳 Dockerized infrastructure
- 📈 Handles class imbalance with SMOTE
- 🔄 End-to-end streaming pipeline

---

## 🏗️ Architecture

```
┌─────────────┐      ┌──────────────┐      ┌─────────────────┐      ┌──────────────┐
│   Producer  │─────>│    Kafka     │─────>│ Spark Streaming │─────>│   MongoDB    │
│ (CSV Data)  │      │   Broker     │      │   + ML Model    │      │  (Results)   │
└─────────────┘      └──────────────┘      └─────────────────┘      └──────────────┘
                                                     │
                                                     ▼
                                            ┌─────────────────┐
                                            │   Dashboard     │
                                            │  (Monitoring)   │
                                            └─────────────────┘
```

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Architecture](#️-architecture)
- [Features](#-features)
- [Tech Stack](#️-tech-stack)
- [Dataset](#-dataset)
- [Performance](#-performance)
- [Quick Start](#-quick-start)
- [Documentation](#-documentation)
- [Project Structure](#-project-structure)
- [License](#-license)
- [Author](#-author)

---

## 🎯 Features

- ✅ **Real-time Streaming**: Kafka ile gerçek zamanlı veri akışı
- ✅ **ML-Powered Detection**: Class imbalance için SMOTE + Random Forest/XGBoost
- ✅ **Scalable Processing**: Apache Spark ile dağıtık veri işleme
- ✅ **Persistent Storage**: MongoDB ile sonuçların saklanması
- ✅ **Live Monitoring**: Dashboard ile canlı izleme
- ✅ **Dockerized**: Tüm servisler Docker ile kolay kurulum

---

## 📁 Project Structure

```
fraud/
├── data/
│   └── creditcard.csv              # Kaggle Credit Card Fraud Dataset (284K transactions)
├── src/
│   ├── producer/
│   │   └── kafka_producer.py       # CSV'den Kafka'ya veri gönderimi
│   ├── consumer/
│   │   └── spark_consumer.py       # Spark Streaming + ML prediction
│   ├── ml_model/
│   │   ├── train_model.py          # Model eğitimi
│   │   ├── preprocessing.py        # Data preprocessing & SMOTE
│   │   └── model.pkl               # Trained model (saved)
│   └── dashboard/
│       └── app.py                  # Streamlit dashboard
├── docker/
│   └── docker-compose.yml          # Kafka, Zookeeper, MongoDB
├── notebooks/
│   └── eda.ipynb                   # Exploratory Data Analysis
├── requirements.txt                # Python dependencies
└── README.md                       # This file
```

---

## 🛠️ Tech Stack

### Core Capabilities
- **Real-Time Streaming**: Apache Kafka message queue with 3 partitions

- **Distributed Processing**: Apache Spark Structured Streaming```

- **ML Detection**: Random Forest classifier with SMOTE balancing

- **Persistent Storage**: MongoDB for prediction results## 🛠️ Technologies Used

- **Live Dashboard**: Streamlit-based real-time monitoring

- **Containerized**: Docker Compose for easy deployment- **Data Streaming**: Apache Kafka 3.x

- **Stream Processing**: Apache Spark 3.x (PySpark)

### ML Pipeline- **Machine Learning**: Scikit-learn, XGBoost, Imbalanced-learn

- ✅ Feature engineering (time-based, interaction features)- **Database**: MongoDB

- ✅ StandardScaler normalization- **Visualization**: Streamlit / Matplotlib / Plotly

- ✅ SMOTE for class imbalance (0.17% → 50%)- **Orchestration**: Docker & Docker Compose

- ✅ Random Forest with 100 trees- **Language**: Python 3.11+

- ✅ Real-time inference on streaming data

## 📊 Dataset

---

**Kaggle Credit Card Fraud Detection Dataset**

## 🏗️ Architecture- **Size**: 284,807 transactions

- **Features**: 30 (Time, V1-V28 PCA, Amount, Class)

```bash
- **Target**: Class (0=Normal, 1=Fraud)

┌─────────────────┐- **Imbalance**: ~0.17% fraud (highly imbalanced)

│  CSV Dataset    │  284,807 transactions

│  (Producer)     │## 🚀 Quick Start

└────────┬────────┘

         │ Kafka Streaming### 1. Prerequisites

         ▼
```
```bash
┌─────────────────┐# Python 3.9+

│  Apache Kafka   │  3 partitions# Docker & Docker Compose

│  + Zookeeper    │  Real-time queue# Java 11+ (for Spark)

└────────┬────────┘```

         │ Stream consume

         ▼
```
### 2. Setup Infrastructure
```bash
┌─────────────────┐

│ Spark Streaming │  Micro-batch processing# Start Kafka, Zookeeper, MongoDB

│   + ML Model    │  Feature engineeringcd docker

│                 │  Fraud predictiondocker-compose up -d

└────────┬────────┘

         │ Save results

         ▼
```
### 3. Install Dependencies
```bash
┌─────────────────┐      ┌─────────────────┐

│    MongoDB      │─────▶│   Streamlit     │pip install -r requirements.txt

│  (Predictions)  │      │   Dashboard     │

└─────────────────┘      └─────────────────┘
```

### 4. Train ML Model

```bash

---python src/ml_model/train_model.py

```

## 🛠️ Tech Stack

### 5. Start Producer (Stream Data)

### Data Streaming
```bash

- **Apache Kafka 7.5.0**: Distributed messaging systempython src/producer/kafka_producer.py

- **Zookeeper**: Kafka coordination service

- **Kafka-Python**: Producer client
```

### 6. Start Consumer (Process & Predict)

### Stream Processing
```bash

- **Apache Spark 3.4.1**: Distributed computing enginespark-submit --packages org.apache.spark:spark-sql-kafka-0-10_2.12:3.4.0 \

- **PySpark**: Python API for Spark  src/consumer/spark_consumer.py

- **Spark Structured Streaming**: Real-time data processing
```



### Machine Learning
### 7. Launch Dashboard

- **Scikit-learn 1.3.0**: ML algorithms and preprocessing```bash

- **XGBoost 2.0.0**: Gradient boosting (alternative model)streamlit run src/dashboard/app.py

- **Imbalanced-learn 0.11.0**: SMOTE implementation```

- **Joblib**: Model serialization

## 📈 ML Pipeline

### Storage & Database

- **MongoDB 7.0**: NoSQL database for predictions1. **Data Preprocessing**

- **PyMongo**: Python MongoDB driver   - Missing value handling

   - Feature scaling (StandardScaler)

### Visualization & Monitoring   - SMOTE for class imbalance

- **Streamlit 1.27.0**: Interactive dashboard

- **Plotly 5.17.0**: Interactive visualizations2. **Model Training**

- **Matplotlib & Seaborn**: Statistical plots   - Algorithm: Random Forest / XGBoost

   - Cross-validation: 5-fold

### DevOps & Infrastructure   - Metrics: Precision, Recall, F1-Score, ROC-AUC

- **Docker & Docker Compose**: Containerization

- **Conda**: Environment management3. **Real-time Prediction**

- **Python 3.10**: Programming language   - Spark Streaming reads from Kafka

   - Model inference on each transaction

---   - Results saved to MongoDB



## 📊 Dataset## 📊 Performance Metrics



**Source**: [Kaggle - Credit Card Fraud Detection](https://www.kaggle.com/mlg-ulb/creditcardfraud)
|Metric | Score |

|--------|-------|

**Statistics**:| Accuracy | ~99% |

- **Size**: 284,807 transactions| Precision | ~95% |

- **Features**: 30 (Time, V1-V28 PCA-transformed, Amount, Class)| Recall | ~85% |

- **Target**: Class (0 = Normal, 1 = Fraud)| F1-Score | ~90% |

- **Imbalance**: 0.173% fraud cases (highly imbalanced)| ROC-AUC | ~98% |



# 📈 Project Performance Overview

## 🚀 Model Performance (Test Set)

**Codebase Summary**
- **Total Code Lines:** ~1,300+ Python LOC  
- **Files:** 13 core files  
- **Technologies:** 10+ different tech stack components  
- **Dataset Size:** 284,807 transactions  
- **Model Accuracy:** ~99%  
- **Processing Speed:** 500–2000 tx/s  

### 🔍 Metrics

| Metric        | Score   |
|---------------|---------|
| **Accuracy**  | 99.97%  |
| **Precision** | 78.3%   |
| **Recall**    | 84.7%   |
| **F1-Score**  | 81.4%   |
| **ROC-AUC**   | 96.9%   |

---

## 📊 Confusion Matrix (Test Set)

|                | Predicted Negative | Predicted Positive |
|----------------|--------------------|--------------------|
| **Actual Negative** | 56,841 (TN)       | 23 (FP)            |
| **Actual Positive** | 15 (FN)           | 83 (TP)            |

---

# ⚙️ System Performance

- **Throughput:** 500–2000 tx/s  
- **Latency:** <500 ms end-to-end  
- **Model Inference Time:** ~10 ms per batch  
- **Kafka Partitions:** 3  

---


## 🚀 Quick Start

### Prerequisites
- Python 3.10+
- Docker Desktop
- Conda (Anaconda/Miniconda)
- Java 11+ (auto-installed via Conda)

### 1. Clone Repository
```bash
git clone https://github.com/talhabektas/fraud-detection-model.git
cd fraud
```

### 2. Setup Environment
```bash
# Create conda environment
conda create -n fraud python=3.11 -y
conda activate fraud

# Install dependencies
conda install -y pandas numpy scikit-learn matplotlib seaborn openjdk=11
pip install imbalanced-learn xgboost kafka-python pyspark pymongo streamlit plotly python-dotenv tqdm joblib
```

### 3. Start Infrastructure
```bash
# Start Docker services (Kafka, Zookeeper, MongoDB)
cd docker
docker compose up -d
cd ..

# Create Kafka topic
docker exec fraud-kafka kafka-topics --create \
  --topic fraud-transactions \
  --bootstrap-server localhost:9092 \
  --replication-factor 1 \
  --partitions 3
```

### 4. Train ML Model
```bash
conda activate fraud
python src/ml_model/train_model.py
```

### 5. Run System (3 Terminals)

**Terminal 1 - Producer:**
```bash
conda activate fraud
python src/producer/kafka_producer.py --limit 500 --delay 0.5
```

**Terminal 2 - Consumer:**
```bash
conda activate fraud
export JAVA_HOME=$CONDA_PREFIX
spark-submit --packages org.apache.spark:spark-sql-kafka-0-10_2.12:3.4.0,org.mongodb.spark:mongo-spark-connector_2.12:10.2.0 \
  src/consumer/spark_consumer.py
```

**Terminal 3 - Dashboard:**
```bash
conda activate fraud
streamlit run src/dashboard/app.py
```

### 6. Access Services

| Service | URL |
|---------|-----|
| **Dashboard** | http://localhost:8501 |
| **Spark UI** | http://localhost:4040 |
| **MongoDB Express** | http://localhost:8081  |



---

## 📁 Project Structure

```
fraud-detection/
├── data/
│   └── creditcard.csv              # Dataset (284K transactions)
├── src/
│   ├── producer/
│   │   └── kafka_producer.py       # Kafka data streaming
│   ├── consumer/
│   │   └── spark_consumer.py       # Spark consumer + ML
│   ├── ml_model/
│   │   ├── preprocessing.py        # Data preprocessing
│   │   ├── train_model.py          # Model training
│   │   ├── model.pkl              # Trained model (generated)
│   │   └── scaler.pkl             # Fitted scaler (generated)
│   └── dashboard/
│       └── app.py                  # Streamlit dashboard
├── docker/
│   └── docker-compose.yml          # Infrastructure setup
├── notebooks/
│   └── eda.ipynb                   # Exploratory analysis
├── requirements.txt                # Python dependencies
├── .gitignore                      # Git ignore rules
├── README.md                       # This file

```

**Total**: ~1,300+ lines of Python code across 7 core modules.

---

## 🎓 Key Learnings

This project demonstrates:
- ✅ **Event-Driven Architecture**: Kafka producer-consumer pattern
- ✅ **Stream Processing**: Spark Structured Streaming with micro-batches
- ✅ **ML in Production**: Real-time model inference at scale
- ✅ **Class Imbalance**: SMOTE for handling imbalanced datasets
- ✅ **Containerization**: Docker for reproducible deployments
- ✅ **Data Pipeline**: End-to-end ETL with streaming data

---

## 📄 License

This project is created for **educational purposes** as part of a university Data Analytics course.

**Dataset License**: The Credit Card Fraud Detection dataset is provided by [Kaggle](https://www.kaggle.com/mlg-ulb/creditcardfraud).

---

## 👨‍💻 Author

**Mehmet Talha Bektas**
- Course: Data Analytics
- GitHub: [@mehmetalha](https://github.com/talhabektas)

---

## 🙏 Acknowledgments

- **Dataset**: ULB Machine Learning Group via Kaggle
- **Technologies**: Apache Software Foundation, MongoDB Inc.
- **Inspiration**: Real-world fraud detection systems
---

## ⭐ Star This Project

If you found this project helpful, please consider giving it a star! ⭐

---


<p align="center">
  Made with Apache Kafka • Spark • MongoDB • Python
</p>
