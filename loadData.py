import json
import urllib.request

from openpyxl import Workbook

# config 불러오기
with open("config.json", "rt") as f:
    CONFIG: dict = json.load(f)


def make_url(station: str, start: int, end: int) -> str:
    """
    받은 인자를 이용하여 지하철 실시간 API request를 위한 url을 구성합니다.

    :param station: 역 이름
    :param start: 시작 페이지
    :param end: 끝 페이지
    :return: 구성된 url
    """
    return f"http://swopenapi.seoul.go.kr/api/subway/{CONFIG['key']}/xml/realtimeStationArrival/{start}/{end}/{station}"


def fetch(station: str, start: int, end: int) -> dict:
    """
    API에서 데이터를 가져와 반환합니다.

    :param station: 역 이름
    :param start: 시작 페이지
    :param end: 끝 페이지
    :return: api 데이터 딕셔너리
    """
    url = make_url(station, start, end)
    text_data = urllib.request.urlopen(url).read().decode("utf-8")
    subway_arrival = json.loads(text_data)
    return subway_arrival


def save_json(data: dict, file_name: str):
    wb = Workbook()
    ws = wb.active

    ws.append(
        [
            "전체 지하철 수",
            "n번째 지하철",
            "지하철호선ID",
            "지하철호선명",
            "지하철역ID",
            "지하철역명",
            "열차번호",
            "최종수신날짜",
            "최종수신시간",
            "상하행선구분",
            "종착지하철역ID",
            "종착지하철역명",
            "열차상태구분",
            "급행여부",
            "막차여부",
        ]
    )
    for row in data["realtimePositionList"]:
        ws.append(
            [
                row["totalCount"],
                row["rowNum"],
                row["subwayId"],
                row["subwayNm"],
                row["statnId"],
                row["statnNm"],
                row["trainNo"],
                row["lastRecptnDt"],
                row["recptnDt"],
                row["updnLine"],
                row["updnLine"],
                row["statnTid"],
                row["statnTnm"],
                row["trainSttus"],
                row["directAt"],
                row["lstcarAt"],
            ]
        )
    wb.save(file_name)
