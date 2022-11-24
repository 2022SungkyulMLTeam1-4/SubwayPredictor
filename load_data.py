import os
import json
import requests
import datetime
import time
from openpyxl import Workbook

# config 불러오기
with open("config.json", "rt", encoding="utf-8-sig") as config_file:
    CONFIG: dict = json.load(config_file)


def make_url(request_type: int, request_param: str, start: int, end: int) -> str:
    """
    받은 인자를 이용하여 지하철 실시간 API request를 위한 url을 구성합니다.

    :param request_type: 0: 위치정보, 1: 도착정보
    :param request_param: type에 따른 요청 파라미터 (0: 호선 이름, 1: 역 이름)
    :param start: 시작 페이지
    :param end: 끝 페이지
    :return: 구성된 url
    """
    request_types = ["realtimePosition", "realtimeStationArrival"]
    return (
        f"http://swopenapi.seoul.go.kr/api/subway/{CONFIG['key']}"
        f"/json/{request_types[request_type]}/{start}/{end}/{request_param}"
    )


def fetch(request_type: int, param: str, start: int, end: int) -> dict:
    """
    API에서 데이터를 가져와 반환합니다.

    :param request_type: 0: 위치정보, 1: 도착정보
    :param param: type에 따른 요청 파라미터 (0: 호선 이름, 1: 역 이름)
    :param start: 시작 페이지
    :param end: 끝 페이지
    :return: api 데이터 딕셔너리
    """
    url = make_url(request_type, param, start, end)
    req_data = requests.get(url)
    json_data = req_data.json()
    
    if json_data["errorMessage"].get("status", 200) == 200:
        print("load sucess")
    
    elif json_data["errorMessage"].get("status", 200) != 200:
        print(json_data["errorMessage"])
        
    return json_data

def data_save(json_data: any):
    """
    가져온 데이터를 바탕으로 xlsx 파일 (데이터셋)을 만듭니다.
    :json_data: json 데이터
    """
    
    if not os.path.exists('./dataset'):
        os.makedirs('./dataset')
        
    filename = datetime.datetime.now().strftime("1호선 지하철 위치정보 %Y-%m-%d %H：%M")
    
    wb = Workbook()
    ws = wb.active
    ws.append(
        ['전체 지하철 수', 'n번째 지하철', '사소한 항목', '지하철호선ID', '지하철호선명', '지하철역ID',
               '지하철역명', '열차번호', '최종수신날짜', '최종수신시간', '상하행선구분', '종착지하철역ID',
               '종착지하철역명', '열차상태구분', '급행여부', '막차여부']
        )
    if json_data['realtimePositionList'] != None:
        for data in json_data['realtimePositionList']:
            ws.append(
                [data['totalCount'], data['rowNum'], data['selectedCount'], data['subwayId'],
                    data['subwayNm'], data['statnId'], data['statnNm'], data['trainNo'],
                    data['lastRecptnDt'], data['recptnDt'], data['updnLine'], data['statnTid'],
                    data['statnTnm'], data['trainSttus'], data['directAt'], data['lstcarAt']]
                )
        wb.save('./dataset/' + filename + '.xlsx')

        print("dataset save sucess")
        
    elif json_data['realtimePositionList'] == None:
        pass
    
    # 60초마다 한번씩 실행합니다.
    time.sleep(60)

if __name__ == "__main__":
    
    while True:
        json_data = fetch(0, "1호선", 0, 1000)
        data_save(json_data)