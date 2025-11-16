"""
Kafka Producer - Streams credit card transactions to Kafka
Reads from CSV and sends transactions in real-time simulation
"""

import pandas as pd
import json
import time
from kafka import KafkaProducer
from kafka.errors import KafkaError
import argparse
from datetime import datetime
import sys


class FraudDataProducer:
    """Kafka Producer for streaming fraud detection data"""
    
    def __init__(self, 
                 bootstrap_servers='localhost:9092',
                 topic='fraud-transactions',
                 batch_size=100):
        """
        Initialize Kafka Producer
        
        Args:
            bootstrap_servers: Kafka broker address
            topic: Kafka topic name
            batch_size: Number of transactions to send per batch
        """
        self.bootstrap_servers = bootstrap_servers
        self.topic = topic
        self.batch_size = batch_size
        self.producer = None
        self.total_sent = 0
        
    def create_producer(self):
        """Create Kafka producer instance"""
        print(f"🔌 Connecting to Kafka broker: {self.bootstrap_servers}")
        
        try:
            self.producer = KafkaProducer(
                bootstrap_servers=self.bootstrap_servers,
                value_serializer=lambda v: json.dumps(v).encode('utf-8'),
                key_serializer=lambda k: str(k).encode('utf-8'),
                acks='all',
                retries=3,
                max_in_flight_requests_per_connection=1
            )
            print("✅ Successfully connected to Kafka")
            return True
        except Exception as e:
            print(f"❌ Failed to connect to Kafka: {e}")
            return False
    
    def load_data(self, filepath):
        """Load transaction data from CSV"""
        print(f"\n📊 Loading data from {filepath}...")
        try:
            df = pd.read_csv(filepath)
            print(f"✅ Loaded {len(df):,} transactions")
            print(f"   - Fraud cases: {df['Class'].sum():,} ({df['Class'].mean()*100:.3f}%)")
            print(f"   - Normal cases: {(df['Class']==0).sum():,}")
            return df
        except Exception as e:
            print(f"❌ Failed to load data: {e}")
            return None
    
    def send_transaction(self, transaction, transaction_id):
        """
        Send a single transaction to Kafka
        
        Args:
            transaction: Dictionary of transaction data
            transaction_id: Unique transaction ID
        """
        try:
            # Add metadata
            transaction['transaction_id'] = transaction_id
            transaction['timestamp'] = datetime.now().isoformat()
            
            # Send to Kafka
            future = self.producer.send(
                self.topic,
                key=str(transaction_id),
                value=transaction
            )
            
            # Wait for confirmation (optional, for reliability)
            record_metadata = future.get(timeout=10)
            
            return True
            
        except KafkaError as e:
            print(f"❌ Kafka error: {e}")
            return False
        except Exception as e:
            print(f"❌ Error sending transaction: {e}")
            return False
    
    def stream_data(self, df, delay=0.1, shuffle=True, limit=None):
        """
        Stream transactions to Kafka
        
        Args:
            df: DataFrame with transactions
            delay: Delay between transactions (seconds)
            shuffle: Whether to shuffle transactions
            limit: Maximum number of transactions to send (None = all)
        """
        print("\n" + "=" * 70)
        print("🚀 STARTING DATA STREAMING")
        print("=" * 70)
        print(f"   Topic: {self.topic}")
        print(f"   Delay: {delay}s per transaction")
        print(f"   Shuffle: {shuffle}")
        print(f"   Limit: {limit if limit else 'All'}")
        print("=" * 70 + "\n")
        
        # Shuffle if requested
        if shuffle:
            df = df.sample(frac=1, random_state=42).reset_index(drop=True)
        
        # Limit if specified
        if limit:
            df = df.head(limit)
        
        total_transactions = len(df)
        fraud_count = 0
        normal_count = 0
        start_time = time.time()
        
        try:
            for idx, row in df.iterrows():
                # Convert row to dictionary
                transaction = row.to_dict()
                
                # Track fraud/normal
                if transaction.get('Class', 0) == 1:
                    fraud_count += 1
                else:
                    normal_count += 1
                
                # Send transaction
                success = self.send_transaction(transaction, idx)
                
                if success:
                    self.total_sent += 1
                    
                    # Progress update
                    if (idx + 1) % self.batch_size == 0:
                        elapsed = time.time() - start_time
                        rate = self.total_sent / elapsed
                        progress = (idx + 1) / total_transactions * 100
                        
                        print(f"📤 [{idx+1}/{total_transactions}] ({progress:.1f}%) | "
                              f"Fraud: {fraud_count} | Normal: {normal_count} | "
                              f"Rate: {rate:.1f} tx/s")
                
                # Delay to simulate real-time streaming
                if delay > 0:
                    time.sleep(delay)
            
            # Final summary
            elapsed = time.time() - start_time
            print("\n" + "=" * 70)
            print("✅ STREAMING COMPLETE!")
            print("=" * 70)
            print(f"   Total sent: {self.total_sent:,}")
            print(f"   Fraud: {fraud_count:,}")
            print(f"   Normal: {normal_count:,}")
            print(f"   Duration: {elapsed:.2f}s")
            print(f"   Average rate: {self.total_sent/elapsed:.1f} tx/s")
            print("=" * 70 + "\n")
            
        except KeyboardInterrupt:
            print("\n⚠️  Streaming interrupted by user")
        except Exception as e:
            print(f"\n❌ Streaming error: {e}")
        finally:
            self.close()
    
    def close(self):
        """Close Kafka producer"""
        if self.producer:
            print("\n🔌 Flushing and closing producer...")
            self.producer.flush()
            self.producer.close()
            print("✅ Producer closed")


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Kafka Producer for Fraud Detection')
    parser.add_argument('--data', type=str, default='data/creditcard.csv',
                       help='Path to transaction data CSV')
    parser.add_argument('--broker', type=str, default='localhost:9092',
                       help='Kafka broker address')
    parser.add_argument('--topic', type=str, default='fraud-transactions',
                       help='Kafka topic name')
    parser.add_argument('--delay', type=float, default=0.1,
                       help='Delay between transactions (seconds)')
    parser.add_argument('--batch-size', type=int, default=100,
                       help='Batch size for progress updates')
    parser.add_argument('--limit', type=int, default=None,
                       help='Limit number of transactions to send')
    parser.add_argument('--no-shuffle', action='store_true',
                       help='Do not shuffle transactions')
    
    args = parser.parse_args()
    
    # Create producer
    producer = FraudDataProducer(
        bootstrap_servers=args.broker,
        topic=args.topic,
        batch_size=args.batch_size
    )
    
    # Connect to Kafka
    if not producer.create_producer():
        print("❌ Failed to initialize producer. Exiting.")
        sys.exit(1)
    
    # Load data
    df = producer.load_data(args.data)
    if df is None:
        print("❌ Failed to load data. Exiting.")
        sys.exit(1)
    
    # Stream data
    producer.stream_data(
        df,
        delay=args.delay,
        shuffle=not args.no_shuffle,
        limit=args.limit
    )


if __name__ == "__main__":
    main()
