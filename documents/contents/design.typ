#import "../utils.typ": *
#let design = [
= 프로그램 구성
- `environment.py`: 에이전트가 투자할 종목의 차트 데이터를 관리합니다. 에이전트는 t 시점에서 t 이후의 시점을 알 수 없습니다.
- `agent.py`: 주식을 매수하거나 매도하는 투자자 역할로 초기 자본금, 현금 잔고, 주식 잔고를 관리합니다. 
- `networks`: t 시점의 주식 데이터가 제공됐을 때 매수할지, 또는 매도할지를 결정하는 역할을 합니다.
    - `__init__.py`: 어떤 네트워크를 사용할지 결정합니다.
    - `network.py`: 가치 신경망과 정책 신경망 학습을 위해 활용합니다.
- `learners.py`: 주어진 환경, 에이전트, 신경망을 가지고 강화학습을 수행합니다. 
- `data_manager.py`: 강화학습을 위한 차트 데이터를 불러오고 전처리를 진행합니다.
- `visualizer.py`: 학습 과정 시각화합니다.
- `main.py`: 여러 옵션을 가지고 강화학습을 실행합니다.
- `settings.py`: 기본 설정을 관리합니다.
- `utils.py`: 각종 유틸리티 함수를 관리합니다.
]