from kafka import KafkaProducer
from kafka import KafkaConsumer 
from flask import request
from bs4 import BeautifulSoup
from apscheduler.schedulers.background import BackgroundScheduler
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
import json
import time
import requests
import pandas as pd
import dataset as dt

def subscribe_type(category):
    if category=='교수학습지원센터':
        return 'CTL'
    elif category=='마음나눔센터':
        return 'COUNSEL'
    elif category=='진로취창업센터':
        return 'CAREER'
    elif category=='대외교류센터':
        return 'GLOBAL'
    else:
        return 'EXCEPTION'

def kafka_producer(pred,title,url,category, tm):
    producer = KafkaProducer(acks=0, compression_type='gzip', bootstrap_servers=['13.124.108.212:9092'], value_serializer=lambda x: json.dumps(x).encode('euc-kr')) 
    start = time.time()      
    producer.send("mlRequest", {
        'subscribeTypes':[subscribe_type(category)],
        'predict':pred,
        'title':title,
        'clickLink':url,
        'time':tm
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
            category = li_list[4].text
            link = item.find('li', class_='black05 ellipsis').find("a").get("data-params")
            link = "https://web.kangnam.ac.kr/menu/board/info/f19069e6134f8f8aa7f689a4a675e66f.do?scrtWrtiYn=false&encMenuSeq=%s&encMenuBoardSeq=%s" %(link[34:66],link[87:119])
        else:
            continue
        noticedict['Title'] = title
        noticedict['Link'] =  link
        noticedict['Category'] =  category
        noticelist.append(noticedict)
    df = pd.DataFrame.from_records(noticelist)
    print("----------Crawling success----------")
    return df

def predict(dataframe):
    print("----------Model predict----------")
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
    tm = time.localtime(1575142526.500323)
    tm = time.strftime('%Y-%m-%d %I:%M:%S %p', tm)
    
    for i, s in enumerate(score):
        s = float(s)
        if(s > 0.5):
            # kafka_producer("{:.2f}% 확률로 비교과프로그램입니다.\n".format(s * 100), m_dataset['Title'][i], m_dataset['Link'][i],m_dataset['Category'][i],tm)
            kafka_producer(1, m_dataset['Title'][i], m_dataset['Link'][i],m_dataset['Category'][i],tm)
        else:
            # kafka_producer("{:.2f}% 확률로 비교과 프로그램이 아닙니다.\n".format((1 - s) * 100), m_dataset['Title'][i], m_dataset['Link'][i],m_dataset['Category'][i],tm)
            kafka_producer(0, m_dataset['Title'][i], m_dataset['Link'][i],m_dataset['Category'][i],tm)

origin_dataframe = crawling()
def schedule():
    print("----------Schedule start----------")
    global origin_dataframe
    dataframe = crawling()
    temp = dataframe.copy()
    for i, o in enumerate(origin_dataframe['Title']):
        if o == dataframe['Title'][i]:
            temp.drop(i)
    predict(temp)
    origin_dataframe = dataframe.copy()
    print("----------Schedule finished----------")
    

sched = BackgroundScheduler(timezone="UTC", daemon=True)
sched.add_job(schedule,'cron',day_of_week='0-4', hour='0-23')
sched.start()

while True:
    print("----------Running main process...----------")
    time.sleep(5)
