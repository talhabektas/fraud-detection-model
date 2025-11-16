"""
Spark Streaming Consumer - Real-time Fraud Detection
Consumes transactions from Kafka, performs ML prediction, saves to MongoDB
"""

from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    from_json, col, current_timestamp, udf, struct, to_json, lit
)
from pyspark.sql.types import (
    StructType, StructField, StringType, DoubleType, IntegerType, TimestampType
)
import joblib
import numpy as np
import pandas as pd
import json
import sys
import os


class FraudDetectionConsumer:
    """Spark Streaming Consumer for real-time fraud detection"""
    
    def __init__(self, 
                 kafka_bootstrap_servers='localhost:9092',
                 kafka_topic='fraud-transactions',
                 mongodb_uri='mongodb://admin:fraudadmin123@localhost:27017',
                 mongodb_database='fraud_detection',
                 mongodb_collection='predictions',
                 model_path='src/ml_model/model.pkl',
                 scaler_path='src/ml_model/scaler.pkl'):
        
        self.kafka_bootstrap_servers = kafka_bootstrap_servers
        self.kafka_topic = kafka_topic
        self.mongodb_uri = mongodb_uri
        self.mongodb_database = mongodb_database
        self.mongodb_collection = mongodb_collection
        self.model_path = model_path
        self.scaler_path = scaler_path
        
        self.spark = None
        self.model = None
        self.scaler = None
        
    def create_spark_session(self):
        """Create Spark session with Kafka and MongoDB support"""
        print("🚀 Creating Spark Session...")
        
        self.spark = SparkSession.builder \
            .appName("FraudDetectionStreaming") \
            .config("spark.jars.packages", 
                    "org.apache.spark:spark-sql-kafka-0-10_2.12:3.4.0,"
                    "org.mongodb.spark:mongo-spark-connector_2.12:10.2.0") \
            .config("spark.mongodb.write.connection.uri", self.mongodb_uri) \
            .config("spark.mongodb.write.database", self.mongodb_database) \
            .config("spark.mongodb.write.collection", self.mongodb_collection) \
            .getOrCreate()
        
        self.spark.sparkContext.setLogLevel("WARN")
        print("✅ Spark Session created")
        
    def load_model(self):
        """Load trained ML model and scaler"""
        print(f"\n📂 Loading model from {self.model_path}...")
        self.model = joblib.load(self.model_path)
        print("✅ Model loaded")
        
        print(f"📂 Loading scaler from {self.scaler_path}...")
        self.scaler = joblib.load(self.scaler_path)
        print("✅ Scaler loaded")
        
    def define_schema(self):
        """Define schema for incoming Kafka messages"""
        return StructType([
            StructField("Time", DoubleType(), True),
            StructField("V1", DoubleType(), True),
            StructField("V2", DoubleType(), True),
            StructField("V3", DoubleType(), True),
            StructField("V4", DoubleType(), True),
            StructField("V5", DoubleType(), True),
            StructField("V6", DoubleType(), True),
            StructField("V7", DoubleType(), True),
            StructField("V8", DoubleType(), True),
            StructField("V9", DoubleType(), True),
            StructField("V10", DoubleType(), True),
            StructField("V11", DoubleType(), True),
            StructField("V12", DoubleType(), True),
            StructField("V13", DoubleType(), True),
            StructField("V14", DoubleType(), True),
            StructField("V15", DoubleType(), True),
            StructField("V16", DoubleType(), True),
            StructField("V17", DoubleType(), True),
            StructField("V18", DoubleType(), True),
            StructField("V19", DoubleType(), True),
            StructField("V20", DoubleType(), True),
            StructField("V21", DoubleType(), True),
            StructField("V22", DoubleType(), True),
            StructField("V23", DoubleType(), True),
            StructField("V24", DoubleType(), True),
            StructField("V25", DoubleType(), True),
            StructField("V26", DoubleType(), True),
            StructField("V27", DoubleType(), True),
            StructField("V28", DoubleType(), True),
            StructField("Amount", DoubleType(), True),
            StructField("Class", IntegerType(), True),
            StructField("transaction_id", StringType(), True),
            StructField("timestamp", StringType(), True),
        ])
    
    def create_features(self, df):
        """Create engineered features matching training"""
        from pyspark.sql.functions import floor, log1p
        
        # Time-based features
        df = df.withColumn("Hour", (floor(col("Time") / 3600) % 24))
        df = df.withColumn("Day", floor(col("Time") / (3600 * 24)))
        
        # Amount-based features
        df = df.withColumn("Amount_log", log1p(col("Amount")))
        
        # Interaction features
        df = df.withColumn("V1_Amount", col("V1") * col("Amount"))
        df = df.withColumn("V2_Amount", col("V2") * col("Amount"))
        
        return df
    
    def predict_fraud(self, batch_df, batch_id):
        """
        Make predictions on a batch of transactions
        
        Args:
            batch_df: PySpark DataFrame with transactions
            batch_id: Batch identifier
        """
        if batch_df.isEmpty():
            return
        
        print(f"\n📦 Processing batch {batch_id} with {batch_df.count()} transactions...")
        
        try:
            # Convert to Pandas for ML prediction
            pandas_df = batch_df.toPandas()
            
            # Feature columns (excluding metadata)
            feature_cols = [col for col in pandas_df.columns 
                          if col not in ['Class', 'transaction_id', 'timestamp', 'Time']]
            
            X = pandas_df[feature_cols]
            
            # Scale features
            X_scaled = self.scaler.transform(X)
            
            # Predict
            predictions = self.model.predict(X_scaled)
            probabilities = self.model.predict_proba(X_scaled)[:, 1]
            
            # Add predictions to DataFrame
            pandas_df['prediction'] = predictions
            pandas_df['fraud_probability'] = probabilities
            pandas_df['processing_time'] = pd.Timestamp.now()
            
            # Calculate if it's a true/false positive/negative
            if 'Class' in pandas_df.columns:
                pandas_df['is_fraud'] = pandas_df['Class']
                pandas_df['correct_prediction'] = (pandas_df['Class'] == pandas_df['prediction'])
            else:
                pandas_df['is_fraud'] = None
                pandas_df['correct_prediction'] = None
            
            # Count fraud detections
            fraud_detected = (predictions == 1).sum()
            high_risk = (probabilities > 0.7).sum()
            
            print(f"   ⚠️  Fraud detected: {fraud_detected}/{len(predictions)}")
            print(f"   🔴 High risk (>70%): {high_risk}/{len(predictions)}")
            
            # Convert back to Spark DataFrame
            result_df = self.spark.createDataFrame(pandas_df)
            
            # Write to MongoDB
            result_df.write \
                .format("mongodb") \
                .mode("append") \
                .save()
            
            print(f"   ✅ Batch {batch_id} saved to MongoDB")
            
            # Also print fraud cases
            fraud_cases = pandas_df[pandas_df['prediction'] == 1]
            if len(fraud_cases) > 0:
                print(f"\n   🚨 FRAUD ALERTS:")
                for _, fraud in fraud_cases.head(5).iterrows():
                    print(f"      TX: {fraud['transaction_id']} | "
                          f"Amount: ${fraud['Amount']:.2f} | "
                          f"Probability: {fraud['fraud_probability']:.2%}")
            
        except Exception as e:
            print(f"   ❌ Error processing batch {batch_id}: {e}")
            import traceback
            traceback.print_exc()
    
    def start_streaming(self):
        """Start consuming from Kafka and processing"""
        print("\n" + "=" * 70)
        print("🚀 STARTING REAL-TIME FRAUD DETECTION")
        print("=" * 70)
        print(f"   Kafka Topic: {self.kafka_topic}")
        print(f"   MongoDB: {self.mongodb_database}.{self.mongodb_collection}")
        print("=" * 70 + "\n")
        
        # Read from Kafka
        df = self.spark \
            .readStream \
            .format("kafka") \
            .option("kafka.bootstrap.servers", self.kafka_bootstrap_servers) \
            .option("subscribe", self.kafka_topic) \
            .option("startingOffsets", "latest") \
            .load()
        
        # Parse JSON
        schema = self.define_schema()
        parsed_df = df.select(
            from_json(col("value").cast("string"), schema).alias("data")
        ).select("data.*")
        
        # Create features
        featured_df = self.create_features(parsed_df)
        
        # Write stream with predictions
        query = featured_df \
            .writeStream \
            .foreachBatch(self.predict_fraud) \
            .outputMode("append") \
            .option("checkpointLocation", "/tmp/fraud_detection_checkpoint") \
            .start()
        
        print("✅ Streaming started. Waiting for transactions...")
        print("   Press Ctrl+C to stop\n")
        
        query.awaitTermination()
    
    def run(self):
        """Main run method"""
        try:
            self.create_spark_session()
            self.load_model()
            self.start_streaming()
        except KeyboardInterrupt:
            print("\n⚠️  Stopping consumer...")
        except Exception as e:
            print(f"\n❌ Consumer error: {e}")
            import traceback
            traceback.print_exc()
        finally:
            if self.spark:
                self.spark.stop()
                print("✅ Spark session stopped")


def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Spark Streaming Consumer for Fraud Detection')
    parser.add_argument('--kafka-broker', type=str, default='localhost:9092',
                       help='Kafka broker address')
    parser.add_argument('--kafka-topic', type=str, default='fraud-transactions',
                       help='Kafka topic name')
    parser.add_argument('--mongodb-uri', type=str, 
                       default='mongodb://admin:fraudadmin123@localhost:27017',
                       help='MongoDB connection URI')
    parser.add_argument('--model', type=str, default='src/ml_model/model.pkl',
                       help='Path to trained model')
    parser.add_argument('--scaler', type=str, default='src/ml_model/scaler.pkl',
                       help='Path to fitted scaler')
    
    args = parser.parse_args()
    
    # Create and run consumer
    consumer = FraudDetectionConsumer(
        kafka_bootstrap_servers=args.kafka_broker,
        kafka_topic=args.kafka_topic,
        mongodb_uri=args.mongodb_uri,
        model_path=args.model,
        scaler_path=args.scaler
    )
    
    consumer.run()


if __name__ == "__main__":
    main()
