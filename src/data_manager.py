import os
import pandas as pd
import numpy as np

import settings

COLUMNS_CHART_DATA = ['date', 'open', 'high', 'low', 'close', 'volume']

COLUMNS_TRAINING_DATA = [
    'per', 'pbr', 'roe', 'open_lastclose_ratio', 'high_close_ratio', 'low_close_ratio',
    'diffratio', 'volume_lastvolume_ratio', 
    'close_ma5_ratio', 'volume_ma5_ratio', 
    'close_ma10_ratio', 'volume_ma10_ratio', 
    'close_ma20_ratio', 'volume_ma20_ratio', 
    'close_ma60_ratio', 'volume_ma60_ratio', 
    'close_ma120_ratio', 'volume_ma120_ratio',
    'market_kospi_ma5_ratio', 'market_kospi_ma20_ratio', 
    'market_kospi_ma60_ratio', 'market_kospi_ma120_ratio', 
    'bond_k3y_ma5_ratio', 'bond_k3y_ma20_ratio', 
    'bond_k3y_ma60_ratio', 'bond_k3y_ma120_ratio',
    'ind', 'ind_diff', 'ind_ma5', 'ind_ma10', 'ind_ma20', 'ind_ma60', 'ind_ma120',
    'inst', 'inst_diff', 'inst_ma5', 'inst_ma10', 'inst_ma20', 'inst_ma60', 'inst_ma120',
    'foreign', 'foreign_diff', 'foreign_ma5', 'foreign_ma10', 'foreign_ma20', 
    'foreign_ma60', 'foreign_ma120',
]

def load_data(code, date_from, date_to, ver):
    columns = COLUMNS_TRAINING_DATA

    # 시장 데이터
    df_marketfeatures = pd.read_csv(
        os.path.join(settings.BASE_DIR, 'data', ver, 'marketfeatures.csv'), 
        thousands=',', header=0, converters={'date': lambda x: str(x)})
    
    # 종목 데이터
    df_stockfeatures = None
    for filename in os.listdir(os.path.join(settings.BASE_DIR, 'data', ver)):
        if filename.startswith(code):
            df_stockfeatures = pd.read_csv(
                os.path.join(settings.BASE_DIR, 'data', ver, filename), 
                thousands=',', header=0, converters={'date': lambda x: str(x)})
            break

    # 시장 데이터와 종목 데이터 합치기
    df = pd.merge(df_stockfeatures, df_marketfeatures, on='date', how='left', suffixes=('', '_dup'))
    df = df.drop(df.filter(regex='_dup$').columns.tolist(), axis=1)

    # 날짜 오름차순 정렬
    df = df.sort_values(by='date').reset_index(drop=True)

    # 기간 필터링
    df['date'] = df['date'].str.replace('-', '')
    df = df[(df['date'] >= date_from) & (df['date'] <= date_to)]
    df = df.fillna(method='ffill').reset_index(drop=True)

    # 데이터 조정
    df.loc[:, ['per', 'pbr', 'roe']] = df[['per', 'pbr', 'roe']].apply(lambda x: x / 100)

    # 차트 데이터 분리
    chart_data = df[COLUMNS_CHART_DATA]

    # 학습 데이터 분리
    training_data = df[columns]

    return chart_data, training_data