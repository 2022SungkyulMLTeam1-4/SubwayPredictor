import json

import requests
from openpyxl import Workbook

# config 불러오기
with open("config.json", "rt", encoding="utf-8-sig") as config:
    CONFIG: dict = json.load(config)


def make_url(station: str, start: int, end: int) -> str:
    """
    받은 인자를 이용하여 지하철 실시간 API request를 위한 url을 구성합니다.

    :param station: 역 이름
    :param start: 시작 페이지
    :param end: 끝 페이지
    :return: 구성된 url
    """
    return f"http://swopenapi.seoul.go.kr/api/subway/{CONFIG['key']}/json/realtimeStationArrival/{start}/{end}/{station}"


def fetch(station: str, start: int, end: int) -> dict:
    """
    API에서 데이터를 가져와 반환합니다.
    만약 데이터가 받아와지지 않았을 경우, 빈 딕셔너리를 반환합니다.

    :param station: 역 이름
    :param start: 시작 페이지
    :param end: 끝 페이지
    :return: api 데이터 딕셔너리
    """
    url = make_url(station, start, end)
    req_data = requests.get(url)
    json_data = req_data.json()
    if json_data["status"] != 200:
        return {}
    return json_data


def save_data_to_excel(data: dict, excel_name: str):
    """
    딕셔너리 데이터를 엑셀로 저장합니다.
    미리 지정된 컬럼을 사용합니다.

    :param data: api에서 추출한 딕셔너리 데이터
    :param excel_name: 엑셀 파일에서 확장자를 제외한 이름
    """
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
    wb.save(f"{excel_name}.xlsx")


if __name__ == "__main__":

    def main():
        json_data = fetch("1호선", 0, 1000)
        if json_data == {}:
            return
        save_data_to_excel(json_data, "sample")

    main()
