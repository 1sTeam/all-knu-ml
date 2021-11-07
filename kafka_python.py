from kafka import KafkaProducer
from kafka import KafkaConsumer 
from json import dumps
from json import loads
import time

def kafka_producer():
    producer = KafkaProducer(acks=0, compression_type='gzip', bootstrap_servers=['13.124.108.212:9092'], value_serializer=lambda x: dumps(x).encode('utf-8')) 
    start = time.time() 
    for i in range(10): 
        data = {'str' : 'Test:'+str(i)} 
        producer.send("mlRequest", value=data) 
        producer.flush()
    print("elapsed :", time.time() - start)

def kafka_consumer():
    consumer = KafkaConsumer('mlRequest', bootstrap_servers=['13.124.108.212:9092'], auto_offset_reset='earliest', enable_auto_commit=True, group_id='my-group', value_deserializer=lambda x: loads(x.decode('utf-8')), consumer_timeout_ms=1000 ) # consumer list를 가져온다 print('[begin] get consumer list') 
    for message in consumer:
         print("Topic: %s, Partition: %d, Offset: %d, Key: %s, Value: %s" % ( message.topic, message.partition, message.offset, message.key, message.value )) 
         print('[end] get consumer list')

kafka_consumer()