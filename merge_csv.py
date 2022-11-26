import os

import pandas as pd

file_path = "파일 경로"
file_list = os.listdir(file_path)

for file_name in file_list:
    print(file_name)
    data = pd.read_excel(file_path + "/" + file_name)

    db = pd.DataFrame(data, columns=['전체 지하철 수', 'n번째 지하철', '사소한 항목', '지하철호선ID', '지하철호선명', '지하철역ID',
                                     '지하철역명', '열차번호', '최종수신날짜', '최종수신시간', '상하행선구분', '종착지하철역ID',
                                     '종착지하철역명', '열차상태구분', '급행여부', '막차여부'])

    db.to_csv('datacsv.csv', mode='a', header=False, index=True)
