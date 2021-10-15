from urllib import response
import requests
from bs4 import BeautifulSoup
import pandas as pd

def all_pages_crawling(n = 1):
    headers = {'user-agent':'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/83.0.4103.62 Safari/537.36'}
    #1 부터 수집하면 오래걸림 공지사항이 494페이지까지 있어서 테스트용으로 490페이지부터 시작
    noticelist = list()

    while(1):
        url = "https://web.kangnam.ac.kr/menu/f19069e6134f8f8aa7f689a4a675e66f.do?paginationInfo.currentPageNo="+str(n)+"&searchMenuSeq=0&searchType=&searchValue="
        n+=1
        print(n)
        response = requests.get(url, headers=headers)
        soup=BeautifulSoup(response.content, 'html.parser')
        noticeArea=soup.find('div', class_='tbody')
        if (noticeArea):
            if(len(noticeArea.find_all('ul')) == 3 ):break

            for item in noticeArea.find_all('ul')[3:]:
                noticedict = dict()
                if (item.find('li', class_='black05 ellipsis') != None):
                    li_list = item.find_all('li')
                    title = item.find('li', class_='black05 ellipsis').find("a").get("title")
                    link = item.find('li', class_='black05 ellipsis').find("a").get("data-params")
                    link = "https://web.kangnam.ac.kr/menu/board/info/f19069e6134f8f8aa7f689a4a675e66f.do?scrtWrtiYn=false&encMenuSeq=%s&encMenuBoardSeq=%s" %(link[34:66],link[87:119])
                    postdate = li_list[5].text
                    division = li_list[2].text
                    #text = single_page_crawling(link,headers)

                else:
                    continue

                #print(title,  link, postdate, division,text)
                noticedict['Binary'] = 0
                noticedict['Title'] = title
                noticedict['Link'] =  link
                noticedict['PostDate'] = postdate
                noticedict['division'] = division
                noticedict['NoticeType'] = '공지사항'
                #noticedict['Text'] = text
                noticelist.append(noticedict)
     
    df = pd.DataFrame.from_records(noticelist)
    return df

def single_page_crawling(link, headers):
    response = requests.get(link, headers=headers)
    soup=BeautifulSoup(response.content, 'html.parser')
    noticeArea=soup.find('div', class_='tbody')
    text = ""
    if (noticeArea):
        item = noticeArea.find_all('ul')[1]
        text = item.find('div', class_= 'tbl_view cke_editable cke_editable_themed cke_contents_ltr').get_text()

    return text

def single_page_crawling_for_modeling():
    headers = {'user-agent':'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/83.0.4103.62 Safari/537.36'}
    noticelist = list()

    url = "https://web.kangnam.ac.kr/menu/f19069e6134f8f8aa7f689a4a675e66f.do?paginationInfo.currentPageNo=1&searchMenuSeq=0&searchType=&searchValue="
    response = requests.get(url, headers=headers)
    soup=BeautifulSoup(response.content, 'html.parser')
    noticeArea=soup.find('div', class_='tbody')
    if (noticeArea):
        for item in noticeArea.find_all('ul')[4:]:
            noticedict = dict()
            if (item.find('li', class_='black05 ellipsis') != None):
                li_list = item.find_all('li')
                title = item.find('li', class_='black05 ellipsis').find("a").get("title")
            else:
                continue

            #print(title,  link, postdate, division,text)
            noticedict['Title'] = title
            noticelist.append(noticedict)
     
    df = pd.DataFrame.from_records(noticelist)
    return df

#print(all_pages_crawling())
# all_pages_crawling().to_csv("dataframe.csv", mode='a', header=False, encoding='utf-8-sig')
