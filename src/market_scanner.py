"""
全市场股票扫描器 - 稳定版
"""
import pandas as pd
import numpy as np
from datetime import datetime
from typing import List, Dict
import logging
import time

logger = logging.getLogger(__name__)


class MarketScanner:

    def __init__(self):
        self.all_stocks = None
        self.filtered_stocks = None

    def fetch_all_stocks(self) -> pd.DataFrame:
        try:
            import akshare as ak
            logger.info("获取全A股行情...")
            for attempt in range(3):
                try:
                    df = ak.stock_zh_a_spot()
                    if df is not None and not df.empty:
                        logger.info(f"成功: {len(df)} 只")
                        self.all_stocks = df
                        return df
                except Exception as e:
                    logger.warning(f"第{attempt+1}次: {e}")
                    time.sleep(5)
            return pd.DataFrame()
        except Exception as e:
            logger.error(f"失败: {e}")
            return pd.DataFrame()

    def quick_filter(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return df

        initial = len(df)
        logger.info(f"筛选前: {initial} 只")

        # 列名映射
        df = df.rename(columns={
            '最新价': 'price', '涨跌幅': 'change_pct',
            '成交量': 'volume', '成交额': 'amount',
            '代码': 'code', '名称': 'name',
        })

        # 填充缺失列
        for col in ['code', 'name', 'price', 'change_pct', 'volume']:
            if col not in df.columns:
                df[col] = 0 if col != 'name' else ''

        # 转数值
        df['price'] = pd.to_numeric(df['price'], errors='coerce').fillna(0)
        df['change_pct'] = pd.to_numeric(df['change_pct'], errors='coerce').fillna(0)
        df['volume'] = pd.to_numeric(df['volume'], errors='coerce').fillna(0)

        # 筛选
        df = df[~df['name'].astype(str).str.contains('ST|退市|N |C ', na=False)]
        df = df[df['volume'] > 0]
        df = df[(df['change_pct'] > -3) & (df['change_pct'] < 10)]
        df = df[df['price'] >= 5]

        # 成交额筛选
        if 'amount' in df.columns:
            df['amount'] = pd.to_numeric(df['amount'], errors='coerce').fillna(0)
            df = df[df['amount'] >= 10000000]

        # 限制数量
        if len(df) > 300:
            if 'amount' in df.columns:
                df = df.nlargest(300, 'amount')
            else:
                df = df.nlargest(300, 'volume')

        df['scan_time'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        self.filtered_stocks = df
        logger.info(f"筛选后: {len(df)} 只")
        return df

    def get_stock_list(self) -> List[Dict]:
        if self.filtered_stocks is None or self.filtered_stocks.empty:
            return []
        stocks = []
        for _, row in self.filtered_stocks.iterrows():
            code = str(row.get('code', '')).strip()
            if not code:
                continue
            stocks.append({
                'code': code,
                'name': str(row.get('name', ''))[:8],
                'price': round(float(row.get('price', 0)), 2),
                'change_pct': round(float(row.get('change_pct', 0)), 2),
                'volume': float(row.get('volume', 0)),
                'amount': float(row.get('amount', 0)),
                'turnover': 0,
                'market_cap': 0,
                'volume_ratio': 1.0,
            })
        return stocks
