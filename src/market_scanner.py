"""
全市场股票扫描器 - 优化版
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

        # 重命名
        df = df.rename(columns={
            'changepercent': 'change_pct', 'trade': 'price',
            'turnoverratio': 'turnover', 'per': 'pe', 'mktcap': 'market_cap',
            '代码': 'code', '名称': 'name', '最新价': 'price',
            '涨跌幅': 'change_pct', '成交量': 'volume', '换手率': 'turnover',
        })

        # 确保列存在
        for col in ['code', 'name', 'price', 'change_pct']:
            if col not in df.columns:
                df[col] = 0
        if 'volume' not in df.columns:
            df['volume'] = 0
        if 'turnover' not in df.columns:
            df['turnover'] = 0

        # 排除 ST
        df = df[~df['name'].astype(str).str.contains('ST|退市|N |C ', na=False)]

        # 排除停牌
        df['volume'] = pd.to_numeric(df['volume'], errors='coerce')
        df = df[df['volume'] > 0]

        # 涨跌幅放宽到 -5% ~ +15%
        df['change_pct'] = pd.to_numeric(df['change_pct'], errors='coerce')
        df = df[(df['change_pct'] > -5) & (df['change_pct'] < 15)]

        # 换手率放宽到 0.3% ~ 35%
        df['turnover'] = pd.to_numeric(df['turnover'], errors='coerce')
        df = df[(df['turnover'] >= 0.3) & (df['turnover'] <= 35)]

        # 价格 > 2 元
        df['price'] = pd.to_numeric(df['price'], errors='coerce')
        df = df[df['price'] >= 2]

        df['volume_ratio'] = 1.0
        df['scan_time'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        self.filtered_stocks = df
        logger.info(f"筛选后: {len(df)} 只 (剔除{initial-len(df)}只)")
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
                'turnover': round(float(row.get('turnover', 0)), 2),
                'market_cap': float(row.get('market_cap', 0)),
                'volume_ratio': 1.0,
            })
        return stocks
