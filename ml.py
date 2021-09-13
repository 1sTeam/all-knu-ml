from urllib import response
import requests
from bs4 import BeautifulSoup
import pandas as pd

headers = {'user-agent':'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/83.0.4103.62 Safari/537.36'}

n = 1

while(1):
    url = "https://web.kangnam.ac.kr/menu/f19069e6134f8f8aa7f689a4a675e66f.do?paginationInfo.currentPageNo="+str(n)+"&searchMenuSeq=0&searchType=&searchValue="
    n+=1
    response = requests.get(url, headers=headers)

    soup=BeautifulSoup(response.content, 'html.parser')
    noticelist = list()
    noticeArea=soup.find('div', class_='tbody')
    if (noticeArea):
        if(len(noticeArea.find_all('ul')) == 4 ):break

        for item in noticeArea.find_all('ul')[4:]:
            noticedict = dict()
            if (item.find('li', class_='black05 ellipsis') != None):
                li_list = item.find_all('li')
                title = item.find('li', class_='black05 ellipsis').find("a").get("title")
                link = item.find('li', class_='black05 ellipsis').find("a").get("data-params")
                link = "https://web.kangnam.ac.kr/menu/board/info/f19069e6134f8f8aa7f689a4a675e66f.do?scrtWrtiYn=false&encMenuSeq=%s&encMenuBoardSeq=%s" %(link[34:66],link[87:119])
                postdate = li_list[5].text
                division = li_list[2].text

            else:
                continue

            #print(title,  link, postdate, division)
            noticedict['Title'] = title
            noticedict['Link'] =  link
            noticedict['PostDate'] = postdate
            noticedict['division'] = division
            noticedict['NoticeType'] = '공지사항'
            noticelist.append(noticedict)
        
        df = pd.DataFrame.from_records(noticelist)
        #print(df)