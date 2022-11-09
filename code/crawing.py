from selenium import webdriver
from bs4 import BeautifulSoup as bs
import urllib.request
import os
from tqdm import tqdm
import time

keyword = input('검색어 입력 : ')
url = 'https://www.google.com/search?q=' +keyword
url = url +'&rlz=1C1CHBD_koKR899KR899&source=lnms&tbm=isch&sa=X&ved=2ahUKEwj7lbzD2_L5AhWGrVYBHb_JAEQQ_AUoAXoECAEQAw&biw=941&bih=929&dpr=1'

driver = webdriver.Chrome('C:/Users/MIS1/Desktop/mediapipe/chromedriver.exe')
driver.get(url)
time.sleep(2)

for i in range(10):
    if i>4:
        if i%5 ==0:
            try:
                driver.find_element_by_css_selector('#islmp > div > div > div.tmS4cc.blLOvc.snjnxc > div.gBPM8 > div.qvfT1 > div.YstHxe > input').click()
            except:
                pass
    time.sleep(1)
    driver.execute_script('window.scrollTo(0,document.body.scrollHeight);')
    
html_source=driver.page_source
soup=bs(html_source, 'html.parser')

img_list = soup.find('div',class_='islrc').find_all('img')
img_url = soup.find('div',class_='islrc').find_all('img')[0]['src']
len(img_list)

# 저장폴더 생성
fDir = 'C:/Users/MIS1/Desktop/mediapipe/'
fName = os.listdir(fDir)
# 저장폴더 존재여부 확인
fName_dir = keyword+'0'
cnt = 0

while True:
    if fName_dir not in fName: # 새로 생성한 폴더가 현재 저장 위치에 있으면
        os.makedirs(fDir+fName_dir) # 없ㅇ면 현재 이름으로 생성
        break #생성후 while문 빠져나감
    cnt+=1
    fName_dir=keyword+str(cnt) # 새로운 폴더명 생성
    
print(fDir+fName_dir,'로폴더 생성')

cnt = 0
for img_url in tqdm(img_list, desc = '저장중...'):
    tmp_name = 'C:/Users/MIS1/Desktop/mediapipe/'+ fName_dir + '/' + keyword + str(cnt) + '.jpg'
    try:
        urllib.request.urlretrieve(img_url['src'],tmp_name)
    except:
        urllib.request.urlretrieve(img_url['data-src'],tmp_name)
    cnt+=1

driver.close()