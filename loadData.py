import json

import requests

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
    return json_data


if __name__ == "__main__":

    def main():
        json_data = fetch(0, "1호선", 0, 1000)
        print(json_data)
        if json_data.get("status", 200) != 200:
            print(json_data["status"], json_data["message"])
            return

    main()
