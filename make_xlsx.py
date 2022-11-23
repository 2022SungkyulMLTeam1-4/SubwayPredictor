import os
import threading
import shutil
import json
import urllib.request
import datetime
import schedule
import time
from openpyxl import Workbook

url = """키 이슈로 생략"""

def dataStore():
    text_data = urllib.request.urlopen(url).read().decode('utf-8')
    subway_loc = json.loads(text_data)
    save_file = open('./subway_loc.json', 'w')
    json.dump(subway_loc, save_file)
    save_file.close()
    
    if not os.path.exists('./dataset'):
        os.makedirs('./dataset')
        
    filename = datetime.datetime.now().strftime("1호선 지하철 위치정보 %Y-%m-%d %H：%M")
    
    with open('./subway_loc.json') as file:
        data = json.load(file)
        wb = Workbook()
        ws = wb.active
        ws.append(['전체 지하철 수', 'n번째 지하철', '사소한 항목', '지하철호선ID', '지하철호선명', '지하철역ID', '지하철역명', '열차번호', '최종수신날짜', '최종수신시간', '상하행선구분', '종착지하철역ID', '종착지하철역명', '열차상태구분', '급행여부', '막차여부'])
        
        for data in data['realtimePositionList']:
            ws.append([data['totalCount'], data['rowNum'], data['selectedCount'], data['subwayId'], data['subwayNm'], data['statnId'], data['statnNm'], data['trainNo'], data['lastRecptnDt'], data['recptnDt'], data['updnLine'], data['statnTid'], data['statnTnm'], data['trainSttus'], data['directAt'], data['lstcarAt']])
        wb.save('./dataset/'+filename+'.xlsx')

record = schedule.every(1).minutes.do(dataStore)
count = 0

while True:
    schedule.run_pending()
    time.sleep(1)
    count += 1
    print(count)
    if count >= 60000:
        schedule.cancel_job(record)