# 🎯 Fraud Detection System - Tüm Komutlar

Bu dosya projedeki tüm komutları içerir. Sırayla takip et!

---

## 📋 İçindekiler

1. [Ortam Kurulumu](#-ortam-kurulumu)
2. [Docker Servisleri](#-docker-servisleri)
3. [ML Model Eğitimi](#-ml-model-eğitimi)
4. [Kafka Producer](#-kafka-producer)
5. [Spark Consumer](#️-spark-consumer)
6. [Dashboard](#-dashboard)
7. [MongoDB Yönetimi](#-mongodb-yönetimi)
8. [Kafka Yönetimi](#-kafka-yönetimi)
9. [Test Komutları](#-test-komutları)
10. [Git İşlemleri](#-git-i̇şlemleri)
11. [Screenshot Komutları](#-screenshot-komutları)

---

## 🔧 Ortam Kurulumu

### Conda Environment Oluştur ve Aktif Et
```bash
# Environment oluştur
conda create -n fraud python=3.10 -y

# Environment'ı aktif et
conda activate fraud

# Java 17 kur (Spark 3.4.1 için gerekli - Java 11 ÇALIŞMAZ!)
conda install -c conda-forge openjdk=17 -y
```

### Python Paketlerini Kur
```bash
# requirements.txt'den kur
pip install -r requirements.txt

# veya manuel olarak:
pip install kafka-python==2.0.2
pip install pyspark==3.4.1
pip install pymongo==4.5.0
pip install pandas==2.0.3
pip install numpy==1.24.3
pip install scikit-learn==1.3.0
pip install imbalanced-learn==0.11.0
pip install xgboost==2.0.0
pip install streamlit==1.27.0
pip install plotly==5.17.0
pip install matplotlib==3.7.2
pip install seaborn==0.12.2
```

### Docker PATH Ayarla (macOS)
```bash
export PATH="/Applications/Docker.app/Contents/Resources/bin:$PATH"
```

---

## 🐳 Docker Servisleri

### Docker Compose ile Servisleri Başlat
```bash
# fraud/docker klasörüne git
cd /Users/mehmetalha/Desktop/fraud/docker

# Servisleri başlat (detached mode)
docker compose up -d
File "/Users/mehmetalha/Desktop/fraud/.venv/lib/python3.13/site-packages/pyspark/...File "/Users/mehmetalha/Desktop/fraud/.venv/lib/python3.13/site-packages/pyspark/...
# Servisleri loglarla birlikte başlat
docker compose up

# Sadece belirli servisi başlat
docker compose up -d kafka
```

### Docker Servislerini Kontrol Et
```bash
# Çalışan container'ları listele
docker ps

# Tüm container'ları listele (durdurulmuş olanlar dahil)
docker ps -a

# Container loglarını gör
docker logs fraud-kafka
docker logs fraud-zookeeper
docker logs fraud-mongodb
docker logs fraud-mongo-express

# Container loglarını canlı takip et
docker logs -f fraud-kafka
```

### Docker Servislerini Durdur
```bash
# Tüm servisleri durdur
docker compose down

# Servisleri durdur ve volume'ları sil
docker compose down -v

# Sadece belirli servisi durdur
docker compose stop kafka
```

### Docker Servislerini Yeniden Başlat
```bash
# Tüm servisleri yeniden başlat
docker compose restart

# Sadece belirli servisi yeniden başlat
docker compose restart kafka
```

---

## 🤖 ML Model Eğitimi

### Model Eğit
```bash
# fraud environment'ını aktif et
conda activate fraud

# Ana dizine git
cd /Users/mehmetalha/Desktop/fraud

# Model eğitimini başlat
python src/ml_model/train_model.py
```

**Çıktı:**
- `src/ml_model/model.pkl` - Eğitilmiş Random Forest modeli
- `src/ml_model/scaler.pkl` - StandardScaler
- `src/ml_model/feature_importance_random_forest.png` - Feature importance grafiği
- Terminal'de: ROC-AUC, F1-Score, Precision, Recall metrikleri

### Model Dosyalarını Kontrol Et
```bash
# Model dosyalarının varlığını kontrol et
ls -lh src/ml_model/*.pkl

# Model boyutunu gör
du -h src/ml_model/model.pkl
```

---

## 📤 Kafka Producer

### Producer'ı Başlat (Normal)
```bash
# fraud environment'ını aktif et
conda activate fraud

# Ana dizine git
cd /Users/mehmetalha/Desktop/fraud

# Producer'ı başlat (default: tüm veri, 2 tx/s)
python src/producer/kafka_producer.py
```

### Producer Parametreleri
```bash
# İlk 500 transaction'ı gönder
python src/producer/kafka_producer.py --limit 500

# Gecikme süresini ayarla (0.5 saniye = 2 tx/s)
python src/producer/kafka_producer.py --limit 500 --delay 0.5

# Hızlı gönderim (0.1 saniye = 10 tx/s)
python src/producer/kafka_producer.py --limit 1000 --delay 0.1

# Çok hızlı (0.01 saniye = 100 tx/s)
python src/producer/kafka_producer.py --limit 5000 --delay 0.01
```

### Producer'ı Durdur
```bash
# Terminal'de Ctrl+C
```

---

## ⚡️ Spark Consumer

### Consumer'ı Başlat
```bash
# fraud environment'ını aktif et
conda activate fraud

# JAVA_HOME'u ayarla (Java 17 için doğru path)
export JAVA_HOME=/opt/anaconda3/envs/fraud/lib/jvm
export PATH=$JAVA_HOME/bin:$PATH

# Docker PATH'i ayarla (macOS)
export PATH="/Applications/Docker.app/Contents/Resources/bin:$PATH"

# Ana dizine git
cd /Users/mehmetalha/Desktop/fraud

# Consumer'ı başlat
spark-submit \
  --packages org.apache.spark:spark-sql-kafka-0-10_2.12:3.4.0,org.mongodb.spark:mongo-spark-connector_2.12:10.2.0 \
  src/consumer/spark_consumer.py
```

**Beklenen Çıktı:**
```
✅ Spark Session created
✅ Model loaded
✅ Scaler loaded
🚀 STARTING REAL-TIME FRAUD DETECTION
✅ Streaming started. Waiting for transactions...
```

**Not:** Sklearn version uyarıları (1.7.2 vs 1.3.0) normaldir, model çalışır.

### Spark UI'a Eriş
```bash
# Tarayıcıda aç (Consumer çalışırken)
open http://localhost:4040
```

### Consumer'ı Durdur
```bash
# Terminal'de Ctrl+C
```

---

## 📊 Dashboard

### Streamlit Dashboard'u Başlat
```bash
# fraud environment'ını aktif et
conda activate fraud

# Ana dizine git
cd /Users/mehmetalha/Desktop/fraud

# Dashboard'u başlat
streamlit run src/dashboard/app.py
```

### Dashboard'a Eriş
```bash
# Otomatik tarayıcıda açılır, yoksa manuel aç:
open http://localhost:8501
```

### Dashboard'u Durdur
```bash
# Terminal'de Ctrl+C
```

---

## 🗄️ MongoDB Yönetimi

### MongoDB Shell'e Bağlan
```bash
# MongoDB container'ına bağlan (authentication ile)
docker exec -it fraud-mongodb mongosh -u admin -p fraudadmin123 --authenticationDatabase admin

# Veritabanlarını listele
show dbs

# fraud_detection veritabanını seç
use fraud_detection

# Collection'ları listele
show collections

# predictions collection'ındaki dökümanları say
db.predictions.countDocuments()

# Son 10 prediction'ı gör
db.predictions.find().sort({timestamp: -1}).limit(10)

# Sadece fraud olan prediction'ları gör
db.predictions.find({prediction: 1})

# Collection'ı temizle
db.predictions.deleteMany({})

# Shell'den çık
exit
```

### Mongo Express Web UI
```bash
# Tarayıcıda aç
open http://localhost:8081

# Giriş bilgileri (docker-compose.yml'de tanımlı):
# Username: admin
# Password: admin
```

### MongoDB'yi Yedekle
```bash
# Veritabanını dışa aktar
docker exec fraud-mongodb mongodump --db fraud_detection --out /tmp/backup

# Backup'ı container'dan kopyala
docker cp fraud-mongodb:/tmp/backup ./mongodb_backup
```

---

## 📡 Kafka Yönetimi

### Kafka Container'ına Bağlan
```bash
docker exec -it fraud-kafka bash
```

### Kafka Topic İşlemleri
```bash
# Topic'leri listele
docker exec fraud-kafka kafka-topics --list --bootstrap-server localhost:9092

# fraud-transactions topic'inin detaylarını gör
docker exec fraud-kafka kafka-topics --describe --topic fraud-transactions --bootstrap-server localhost:9092

# Yeni topic oluştur
docker exec fraud-kafka kafka-topics --create \
  --topic test-topic \
  --partitions 3 \
  --replication-factor 1 \
  --bootstrap-server localhost:9092

# Topic'i sil
docker exec fraud-kafka kafka-topics --delete --topic test-topic --bootstrap-server localhost:9092
```

### Kafka Consumer ile Mesajları Oku
```bash
# fraud-transactions topic'inden tüm mesajları oku (baştan)
docker exec fraud-kafka kafka-console-consumer \
  --topic fraud-transactions \
  --from-beginning \
  --bootstrap-server localhost:9092

# Sadece yeni mesajları oku
docker exec fraud-kafka kafka-console-consumer \
  --topic fraud-transactions \
  --bootstrap-server localhost:9092

# Consumer group ile oku
docker exec fraud-kafka kafka-console-consumer \
  --topic fraud-transactions \
  --group test-group \
  --bootstrap-server localhost:9092
```

### Kafka Producer Test
```bash
# Manuel mesaj gönder
docker exec -it fraud-kafka kafka-console-producer \
  --topic fraud-transactions \
  --bootstrap-server localhost:9092

# Sonra mesajları yaz ve Enter'a bas
# Çıkmak için Ctrl+C
```

### Consumer Group Bilgisi
```bash
# Consumer group'ları listele
docker exec fraud-kafka kafka-consumer-groups --list --bootstrap-server localhost:9092

# Belirli bir group'un detaylarını gör
docker exec fraud-kafka kafka-consumer-groups \
  --describe \
  --group spark-fraud-detection \
  --bootstrap-server localhost:9092
```

---

## 🧪 Test Komutları

### Sistemin Çalışıp Çalışmadığını Test Et
```bash
# Docker servisleri çalışıyor mu?
docker ps | grep fraud

# Kafka'ya bağlanabiliyor muyuz?
nc -zv localhost 9092

# MongoDB'ye bağlanabiliyor muyuz?
nc -zv localhost 27017

# Streamlit port'u açık mı?
nc -zv localhost 8501

# Spark UI erişilebilir mi? (Consumer çalışırken)
nc -zv localhost 4040
```

### Python Import Test
```bash
conda activate fraud

python -c "import kafka; print('Kafka OK')"
python -c "import pyspark; print('PySpark OK')"
python -c "import pymongo; print('PyMongo OK')"
python -c "import sklearn; print('Sklearn OK')"
python -c "import imblearn; print('Imbalanced-learn OK')"
python -c "import xgboost; print('XGBoost OK')"
python -c "import streamlit; print('Streamlit OK')"
```

### End-to-End Test
```bash
# Terminal 1: Docker servisleri (arka planda)
cd /Users/mehmetalha/Desktop/fraud/docker
docker compose up -d

# 10 saniye bekle (servislerin başlaması için)
sleep 10

# Terminal 2: Spark Consumer
conda activate fraud
export JAVA_HOME=/opt/anaconda3/envs/fraud/lib/jvm
export PATH=$JAVA_HOME/bin:$PATH
export PATH="/Applications/Docker.app/Contents/Resources/bin:$PATH"
cd /Users/mehmetalha/Desktop/fraud
spark-submit --packages org.apache.spark:spark-sql-kafka-0-10_2.12:3.4.0,org.mongodb.spark:mongo-spark-connector_2.12:10.2.0 src/consumer/spark_consumer.py

# Terminal 3: Kafka Producer (100 transaction test)
conda activate fraud
cd /Users/mehmetalha/Desktop/fraud
python src/producer/kafka_producer.py --limit 100 --delay 0.5

# Terminal 4: Dashboard
conda activate fraud
cd /Users/mehmetalha/Desktop/fraud
streamlit run src/dashboard/app.py
```

**Doğrulama Komutları:**
```bash
# Container'ları kontrol et
docker ps --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"

# MongoDB'deki veri sayısı
docker exec -it fraud-mongodb mongosh -u admin -p fraudadmin123 --authenticationDatabase admin --eval "use fraud_detection; db.predictions.countDocuments()"

# Kafka topic durumu
docker exec fraud-kafka kafka-topics --describe --topic fraud-transactions --bootstrap-server localhost:9092
```

---

## 🔄 Git İşlemleri

### Repository'yi Clone Et
```bash
git clone https://github.com/talhabektas/fraud-detection-model.git
cd fraud-detection-model
```

### Git Durumunu Kontrol Et
```bash
# Değişiklikleri gör
git status

# Değişiklikleri detaylı gör
git diff

# Commit geçmişi
git log --oneline
```

### Değişiklikleri Commit Et
```bash
# Tüm değişiklikleri stage'e al
git add .

# Belirli dosyaları stage'e al
git add README.md PROJECT_REPORT.md

# Commit yap
git commit -m "Update documentation"

# Commit mesajını düzelt (son commit için)
git commit --amend -m "Fixed README structure"
```

### GitHub'a Push Et
```bash
# İlk push
git push -u origin main

# Sonraki push'lar
git push origin main

# Force push (DİKKATLİ!)
git push -f origin main
```

### Branch İşlemleri
```bash
# Branch'leri listele
git branch

# Yeni branch oluştur
git branch feature/new-model

# Branch'e geç
git checkout feature/new-model

# Branch oluştur ve geç (tek komut)
git checkout -b feature/new-model
```

### .gitignore Kontrolü
```bash
# Ignore edilen dosyaları gör
git status --ignored

# Belirli bir dosyanın ignore edilip edilmediğini kontrol et
git check-ignore -v data/creditcard.csv
```

---

## 📸 Screenshot Komutları

### Sistem Çalıştır (Tüm Komponentler)
```bash
# Terminal 1: Docker (arka planda)
cd /Users/mehmetalha/Desktop/fraud/docker
docker compose up -d
sleep 10  # Servislerin başlaması için bekle

# Terminal 2: Spark Consumer
conda activate fraud
export JAVA_HOME=/opt/anaconda3/envs/fraud/lib/jvm
export PATH=$JAVA_HOME/bin:$PATH
export PATH="/Applications/Docker.app/Contents/Resources/bin:$PATH"
cd /Users/mehmetalha/Desktop/fraud
spark-submit --packages org.apache.spark:spark-sql-kafka-0-10_2.12:3.4.0,org.mongodb.spark:mongo-spark-connector_2.12:10.2.0 src/consumer/spark_consumer.py

# Terminal 3: Producer (100 transaction, demo için yeterli)
conda activate fraud
cd /Users/mehmetalha/Desktop/fraud
python src/producer/kafka_producer.py --limit 100 --delay 0.5

# Terminal 4: Dashboard
conda activate fraud
cd /Users/mehmetalha/Desktop/fraud
streamlit run src/dashboard/app.py
```

### Screenshot Alınacak URL'ler
```bash
# 1. Streamlit Dashboard
open http://localhost:8501

# 2. MongoDB Express
open http://localhost:8081

# 3. Spark UI (Consumer çalışırken)
open http://localhost:4040
```

### Screenshot Alınacak Terminal Komutları
```bash
# 1. Docker container'ları
docker ps --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"

# 2. Model training sonuçları (opsiyonel - zaten eğitilmiş)
conda activate fraud
python src/ml_model/train_model.py

# 3. Kafka topic bilgisi
docker exec fraud-kafka kafka-topics --describe --topic fraud-transactions --bootstrap-server localhost:9092

# 4. MongoDB'deki prediction sayısı
docker exec -it fraud-mongodb mongosh -u admin -p fraudadmin123 --authenticationDatabase admin --eval "use fraud_detection; db.predictions.countDocuments()"

# 5. MongoDB'deki örnek veriler
docker exec -it fraud-mongodb mongosh -u admin -p fraudadmin123 --authenticationDatabase admin --eval "use fraud_detection; db.predictions.find().limit(3).pretty()"

# 6. Kafka consumer group bilgisi
docker exec fraud-kafka kafka-consumer-groups --describe --group spark-fraud-detection --bootstrap-server localhost:9092
```

---

## 🛑 Sistemi Temizle ve Durdur

### Tüm Servisleri Durdur
```bash
# Docker servislerini durdur
cd /Users/mehmetalha/Desktop/fraud/docker
docker compose down

# Terminal'lerde çalışan servisleri durdur (her terminal'de)
# Ctrl+C

# Environment'tan çık
conda deactivate
```

### Verileri Temizle
```bash
# MongoDB'yi temizle (authentication ile)
docker exec -it fraud-mongodb mongosh -u admin -p fraudadmin123 --authenticationDatabase admin --eval "use fraud_detection; db.predictions.deleteMany({})"

# Kafka topic'i sil ve yeniden oluştur
docker exec fraud-kafka kafka-topics --delete --topic fraud-transactions --bootstrap-server localhost:9092
docker exec fraud-kafka kafka-topics --create --topic fraud-transactions --partitions 3 --replication-factor 1 --bootstrap-server localhost:9092

# Spark checkpoint'lerini sil
rm -rf /tmp/fraud_detection_checkpoint
```

### Docker'ı Tamamen Temizle (DİKKAT!)
```bash
# Tüm container'ları durdur
docker stop $(docker ps -aq)

# Tüm container'ları sil
docker rm $(docker ps -aq)

# Kullanılmayan volume'ları sil
docker volume prune -f

# Kullanılmayan image'ları sil
docker image prune -a -f
```

---

## 🚀 Hızlı Başlatma (Tek Komut)

### setup.sh ile Otomatik Kurulum
```bash
chmod +x setup.sh
./setup.sh
```

### start.sh ile Tüm Servisleri Başlat
```bash
chmod +x start.sh
./start.sh
```

### stop.sh ile Tüm Servisleri Durdur
```bash
chmod +x stop.sh
./stop.sh
```

---

## 📝 Notlar

### Environment Variables
```bash
# Gerekli environment variable'ları set et
export KAFKA_BROKER=localhost:9092
export MONGODB_URI=mongodb://localhost:27017/
export SPARK_HOME=/opt/anaconda3/envs/fraud
export JAVA_HOME=/opt/anaconda3/envs/fraud/lib/jvm
export PATH=$JAVA_HOME/bin:$PATH
```

### Python Path Ayarı
```bash
# Proje root'unu PYTHONPATH'e ekle
export PYTHONPATH=/Users/mehmetalha/Desktop/fraud:$PYTHONPATH
```

### Performans İzleme
```bash
# CPU ve Memory kullanımı
docker stats

# Disk kullanımı
docker system df

# Belirli container'ın resource kullanımı
docker stats fraud-kafka fraud-mongodb
```

---

## 🆘 Troubleshooting

### Kafka Bağlantı Hatası
```bash
# Kafka container'ını yeniden başlat
docker compose restart kafka

# Kafka loglarını kontrol et
docker logs fraud-kafka --tail 100
```

### MongoDB Bağlantı Hatası
```bash
# MongoDB container'ını yeniden başlat
docker compose restart mongodb

# MongoDB loglarını kontrol et
docker logs fraud-mongodb --tail 100
```

### Spark Hatası
```bash
# Java versiyonunu kontrol et (Java 17 olmalı)
java -version
# Beklenen: openjdk version "17.0.17"

# JAVA_HOME'un doğru set edildiğini kontrol et
echo $JAVA_HOME
# Beklenen: /opt/anaconda3/envs/fraud/lib/jvm

# Java executable'ın varlığını kontrol et
ls -la $JAVA_HOME/bin/java

# Spark checkpoint'lerini temizle
rm -rf /tmp/fraud_detection_checkpoint

# Eğer "UnsupportedClassVersionError" hatası alıyorsan:
# Java 11 yerine Java 17 kullanman gerekiyor!
conda activate fraud
conda install -c conda-forge openjdk=17 -y
export JAVA_HOME=/opt/anaconda3/envs/fraud/lib/jvm
export PATH=$JAVA_HOME/bin:$PATH
```

### Port Çakışması
```bash
# Port'u kullanan process'i bul
lsof -i :9092  # Kafka
lsof -i :27017 # MongoDB
lsof -i :8501  # Streamlit
lsof -i :4040  # Spark UI

# Process'i öldür
kill -9 <PID>
```

---

**Son Güncelleme:** 25 Kasım 2025  
**Proje:** Real-Time Fraud Detection System  
**Repository:** [github.com/talhabektas/fraud-detection-model](https://github.com/talhabektas/fraud-detection-model)

---

## 🚀 BAŞTAN SONA EKSİKSİZ ÇALIŞTIRMA REHBERİ

### Adım 1: Ortamı Hazırla
```bash
# .venv varsa devre dışı bırak
deactivate || true

# Conda fraud environment'ını aktif et
conda activate fraud

# Port 9092'yi temizle (gerekirse)
lsof -ti:9092 | xargs kill -9 || true

# Docker PATH'i ayarla
export PATH="/Applications/Docker.app/Contents/Resources/bin:$PATH"

# Java 17 için JAVA_HOME ayarla
export JAVA_HOME=/opt/anaconda3/envs/fraud/lib/jvm
export PATH=$JAVA_HOME/bin:$PATH

# Java versiyonunu doğrula
java -version  # "openjdk version 17.0.17" görmeli
```

### Adım 2: Docker Servislerini Başlat
```bash
cd /Users/mehmetalha/Desktop/fraud/docker
docker compose up -d
sleep 10  # Servislerin başlaması için bekle

# Kontrol et
docker ps --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"
# 4 container çalışıyor olmalı: kafka, zookeeper, mongodb, mongo-express
```

### Adım 3: Spark Consumer Başlat (Terminal 1)
```bash
conda activate fraud
export JAVA_HOME=/opt/anaconda3/envs/fraud/lib/jvm
export PATH=$JAVA_HOME/bin:$PATH
cd /Users/mehmetalha/Desktop/fraud

spark-submit \
  --packages org.apache.spark:spark-sql-kafka-0-10_2.12:3.4.0,org.mongodb.spark:mongo-spark-connector_2.12:10.2.0 \
  src/consumer/spark_consumer.py

# "✅ Streaming started. Waiting for transactions..." mesajını bekle
```

### Adım 4: Producer Çalıştır (Terminal 2 - YENİ)
```bash
conda activate fraud
cd /Users/mehmetalha/Desktop/fraud
python src/producer/kafka_producer.py --limit 100 --delay 0.5

# Progress bar göreceksin: [100/100] (100.0%)
```

### Adım 5: Dashboard Başlat (Terminal 3 - YENİ)
```bash
conda activate fraud
cd /Users/mehmetalha/Desktop/fraud
streamlit run src/dashboard/app.py

# Tarayıcı otomatik açılacak: http://localhost:8501
```

### Adım 6: Web UI'ları Kontrol Et
```bash
# Streamlit Dashboard
open http://localhost:8501

# Mongo Express (DB görselleştirme)
open http://localhost:8081
# Username: admin, Password: admin

# Spark UI (streaming jobs)
open http://localhost:4040
```

### Adım 7: Verileri Doğrula
```bash
# MongoDB'deki prediction sayısı
docker exec -it fraud-mongodb mongosh -u admin -p fraudadmin123 --authenticationDatabase admin --eval "use fraud_detection; db.predictions.countDocuments()"

# Örnek veriler
docker exec -it fraud-mongodb mongosh -u admin -p fraudadmin123 --authenticationDatabase admin --eval "use fraud_detection; db.predictions.find().limit(3).pretty()"
```

### Adım 8: Sistemi Durdur
```bash
# Her terminal'de Ctrl+C ile durdur

# Docker'ı durdur
cd /Users/mehmetalha/Desktop/fraud/docker
docker compose down

# Environment'tan çık
conda deactivate
```

---

## ⚠️ SORUN GİDERME

### "spark-submit: command not found"
```bash
# Conda fraud environment aktif değil
conda activate fraud
```

### "UnsupportedClassVersionError: class file version 61.0"
```bash
# Java 17 değil, Java 11 kullanıyorsun
conda install -c conda-forge openjdk=17 -y
export JAVA_HOME=/opt/anaconda3/envs/fraud/lib/jvm
export PATH=$JAVA_HOME/bin:$PATH
```

### "NoBrokersAvailable" (Producer hatası)
```bash
# Kafka henüz başlamamış
cd /Users/mehmetalha/Desktop/fraud/docker
docker compose restart kafka
sleep 10
```

### Port 9092 zaten kullanımda
```bash
# Başka process port kullanıyor
lsof -ti:9092 | xargs kill -9
docker compose down
docker compose up -d
```

### MongoDB authentication hatası
```bash
# Username/password doğru kullan
docker exec -it fraud-mongodb mongosh -u admin -p fraudadmin123 --authenticationDatabase admin
```

---

## 📊 BAŞARILI ÇALIŞMA BELİRTİLERİ

✅ **Spark Consumer:**
```
✅ Spark Session created
✅ Model loaded
✅ Scaler loaded
✅ Streaming started. Waiting for transactions...
```

✅ **Producer:**
```
📤 [100/100] (100.0%) | Fraud: X | Normal: Y
✅ STREAMING COMPLETE!
```

✅ **MongoDB:**
```
db.predictions.countDocuments()
100  (veya gönderilen transaction sayısı)
```

✅ **Dashboard:**
- Metrics güncellenecek
- Grafiklerde veriler görünecek
- Fraud alerts oluşacak (eğer varsa)
