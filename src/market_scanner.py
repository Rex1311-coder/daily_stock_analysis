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
        try:
            import akshare as ak
            logger.info("正在获取全A股实时行情（腾讯接口）...")
            for attempt in range(3):
                try:
                    df = ak.stock_zh_a_spot()
                    if df is not None and not df.empty:
                        logger.info(f"成功: {len(df)} 只, 列: {list(df.columns)[:10]}")
                        self.all_stocks = df
                        return df
                except Exception as e:
                    logger.warning(f"第{attempt+1}次失败: {e}")
                    time.sleep(5)
            return pd.DataFrame()
        except Exception as e:
            logger.error(f"获取失败: {e}")
            return pd.DataFrame()

    def quick_filter(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return df

        initial = len(df)
        logger.info(f"快速筛选，原始: {initial} 只")
        logger.info(f"实际列名: {list(df.columns)}")

        # 腾讯接口列名（可能是英文）
        col_map = {
            'code': 'code', 'name': 'name', 'price': 'price',
            'changepercent': 'change_pct', 'trade': 'price',
            'volume': 'volume', 'amount': 'amount',
            'turnoverratio': 'turnover', 'per': 'pe',
            'mktcap': 'market_cap',
        }

        # 先重命名
        df = df.rename(columns={
            'changepercent': 'change_pct',
            'trade': 'price',
            'turnoverratio': 'turnover',
            'per': 'pe',
            'mktcap': 'market_cap',
        })

        # 检查并补充缺失列
        if 'code' not in df.columns and '代码' in df.columns:
            df['code'] = df['代码']
        if 'name' not in df.columns and '名称' in df.columns:
            df['name'] = df['名称']
        if 'price' not in df.columns and '最新价' in df.columns:
            df['price'] = df['最新价']
        if 'change_pct' not in df.columns and '涨跌幅' in df.columns:
            df['change_pct'] = df['涨跌幅']
        if 'volume' not in df.columns and '成交量' in df.columns:
            df['volume'] = df['成交量']
        if 'turnover' not in df.columns and '换手率' in df.columns:
            df['turnover'] = df['换手率']

        # 没有换手率就跳过这个筛选条件
        has_turnover = 'turnover' in df.columns

        # 筛选
        if 'name' in df.columns:
            df = df[~df['name'].str.contains('ST|退市|N |C ', na=False)]

        if 'volume' in df.columns:
            df['volume'] = pd.to_numeric(df['volume'], errors='coerce')
            df = df[df['volume'] > 0]

        if 'change_pct' in df.columns:
            df['change_pct'] = pd.to_numeric(df['change_pct'], errors='coerce')
            df = df[(df['change_pct'] > -3) & (df['change_pct'] < 10)]

        if has_turnover:
            df['turnover'] = pd.to_numeric(df['turnover'], errors='coerce')
            df = df[(df['turnover'] >= 0.5) & (df['turnover'] <= 30)]
        else:
            df['turnover'] = 0

        if 'price' in df.columns:
            df['price'] = pd.to_numeric(df['price'], errors='coerce')
            df = df[df['price'] >= 3]

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
