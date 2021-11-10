from kafka import KafkaProducer
from kafka import KafkaConsumer 
import json
import time
import requests
from flask import request
from bs4 import BeautifulSoup
import pandas as pd
from apscheduler.schedulers.background import BackgroundScheduler
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
import dataset as dt

def kafka_producer(pred,title,url):
    producer = KafkaProducer(acks=0, compression_type='gzip', bootstrap_servers=['13.124.108.212:9092'], value_serializer=lambda x: json.dumps(x).encode('utf-8')) 
    start = time.time()      
    producer.send("mlRequest", {
        'predict':pred,
        'title':title,
        'clickLink':url
    }) 
    producer.flush()
    print("elapsed :", time.time() - start)

def kafka_consumer():
    consumer = KafkaConsumer('mlRequest', bootstrap_servers=['13.124.108.212:9092'], auto_offset_reset='earliest', enable_auto_commit=True, group_id='my-group', value_deserializer=lambda x: json.loads(x.decode('utf-8')), consumer_timeout_ms=1000 ) # consumer list를 가져온다 print('[begin] get consumer list') 
    for message in consumer:
         print("Topic: %s, Partition: %d, Offset: %d, Key: %s, Value: %s" % ( message.topic, message.partition, message.offset, message.key, message.value )) 
         print('[end] get consumer list')

def crawling():
    headers = {'user-agent':'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/83.0.4103.62 Safari/537.36'}
    noticelist = list()
    url = "https://web.kangnam.ac.kr/menu/f19069e6134f8f8aa7f689a4a675e66f.do?paginationInfo.currentPageNo=1&searchMenuSeq=0&searchType=&searchValue="
    response = requests.get(url, headers=headers)
    soup=BeautifulSoup(response.content, 'html.parser')
    noticeArea=soup.find('div', class_='tbody')
    for item in noticeArea.find_all('ul')[3:]:
        noticedict = dict()
        if (item.find('li', class_='black05 ellipsis') != None):
            li_list = item.find_all('li')
            title = item.find('li', class_='black05 ellipsis').find("a").get("title")
            link = item.find('li', class_='black05 ellipsis').find("a").get("data-params")
            link = "https://web.kangnam.ac.kr/menu/board/info/f19069e6134f8f8aa7f689a4a675e66f.do?scrtWrtiYn=false&encMenuSeq=%s&encMenuBoardSeq=%s" %(link[34:66],link[87:119])
        else:
            continue
        noticedict['Title'] = title
        noticedict['Link'] =  link
        noticelist.append(noticedict)
    df = pd.DataFrame.from_records(noticelist)
    return df

def predict():
    model = load_model('best_model.h5')
    m_dataset = crawling()
    m_dt = dt.text_processing(m_dataset)
    m_dt.text_normalization()
    m_dt.text_tokenization()
    m_dt.text_integer()
    x_train = pad_sequences(m_dt.dataset['Title'], maxlen = 12)

    # for sentence in x_train:
    score = model.predict(x_train)

    print(score)

    for i, s in enumerate(score):
        s = float(s)
        if(s > 0.5):
            kafka_producer("{:.2f}% 확률로 비교과프로그램입니다.\n".format(s * 100), m_dataset['Title'][i], m_dataset['Link'][i])
        else:
            kafka_producer("{:.2f}% 확률로 비교과 프로그램이 아닙니다.\n".format((1 - s) * 100), m_dataset['Title'][i], m_dataset['Link'][i])

predict()
kafka_consumer()

# sched = BackgroundScheduler(daemon=True)
# for t in range(10,18,1):sched.add_job(crawling,'cron', week='1-53', day_of_week='0-4', hour=str(t))
# sched.start()
