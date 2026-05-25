"""
全市场股票扫描器
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
        """获取全A股实时行情 - 腾讯接口"""
        try:
            import akshare as ak

            logger.info("正在获取全A股实时行情（腾讯接口）...")

            for attempt in range(3):
                try:
                    df = ak.stock_zh_a_spot()
                    if df is not None and not df.empty:
                        logger.info(f"成功: {len(df)} 只")
                        self.all_stocks = df
                        return df
                except Exception as e:
                    logger.warning(f"第{attempt+1}次失败: {e}")
                    time.sleep(5)

            logger.error("所有尝试均失败")
            return pd.DataFrame()

        except Exception as e:
            logger.error(f"获取失败: {e}")
            return pd.DataFrame()

    def quick_filter(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return df

        initial = len(df)
        logger.info(f"快速筛选，原始: {initial} 只")

        col_map = {
            '代码': 'code', '名称': 'name', '最新价': 'price',
            '涨跌幅': 'change_pct', '成交量': 'volume', '成交额': 'amount',
            '换手率': 'turnover', '市盈率-动态': 'pe', '总市值': 'market_cap',
            '量比': 'volume_ratio',
        }
        df = df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})

        for col in ['code', 'name', 'price', 'change_pct', 'turnover']:
            if col not in df.columns:
                logger.error(f"缺少列: {col}")
                return pd.DataFrame()

        df = df[~df['name'].str.contains('ST|退市|N |C ', na=False)]
        df['volume'] = pd.to_numeric(df['volume'], errors='coerce')
        df = df[df['volume'] > 0]
        df['change_pct'] = pd.to_numeric(df['change_pct'], errors='coerce')
        df = df[(df['change_pct'] > -3) & (df['change_pct'] < 10)]
        df['turnover'] = pd.to_numeric(df['turnover'], errors='coerce')
        df = df[(df['turnover'] >= 0.5) & (df['turnover'] <= 30)]
        df['price'] = pd.to_numeric(df['price'], errors='coerce')
        df = df[df['price'] >= 3]

        if 'market_cap' in df.columns:
            df['market_cap'] = pd.to_numeric(df['market_cap'], errors='coerce')
            df = df[df['market_cap'] >= 2e9]

        if 'volume_ratio' not in df.columns:
            df['volume_ratio'] = 1.0

        df['scan_time'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        self.filtered_stocks = df
        logger.info(f"筛选完成: {initial} -> {len(df)} 只")
        return df

    def get_stock_list(self) -> List[Dict]:
        if self.filtered_stocks is None or self.filtered_stocks.empty:
            return []
        stocks = []
        for _, row in self.filtered_stocks.iterrows():
            stocks.append({
                'code': str(row.get('code', '')).strip(),
                'name': str(row.get('name', '')).strip(),
                'price': round(float(row.get('price', 0)), 2),
                'change_pct': round(float(row.get('change_pct', 0)), 2),
                'volume': float(row.get('volume', 0)),
                'amount': float(row.get('amount', 0)),
                'turnover': round(float(row.get('turnover', 0)), 2),
                'pe': round(float(row.get('pe', 0)), 2) if row.get('pe') and float(row.get('pe', 0)) > 0 else 0,
                'market_cap': float(row.get('market_cap', 0)),
                'volume_ratio': round(float(row.get('volume_ratio', 1)), 2),
            })
        return stocks
