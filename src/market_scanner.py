"""
全市场股票扫描器 - 调试版
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
                        logger.info(f"列名: {list(df.columns)}")
                        logger.info(f"前2行:\n{df.head(2).to_string()}")
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
        logger.info(f"原始列: {list(df.columns)[:15]}")

        # 腾讯接口实际列名处理
        # 先看实际有哪些列，然后适配
        col_map = {}
        
        for col in df.columns:
            col_lower = col.lower()
            if col_lower in ['代码', 'code', 'symbol']:
                col_map[col] = 'code'
            elif col_lower in ['名称', 'name']:
                col_map[col] = 'name'
            elif col_lower in ['最新价', 'trade', 'price', '现价']:
                col_map[col] = 'price'
            elif col_lower in ['涨跌幅', 'changepercent', 'pctchange', '涨跌']:
                col_map[col] = 'change_pct'
            elif col_lower in ['成交量', 'volume']:
                col_map[col] = 'volume'
            elif col_lower in ['成交额', 'amount']:
                col_map[col] = 'amount'
            elif col_lower in ['换手率', 'turnoverratio', 'turnover']:
                col_map[col] = 'turnover'
            elif col_lower in ['量比', 'volratio']:
                col_map[col] = 'volume_ratio'

        df = df.rename(columns=col_map)
        logger.info(f"重命名后列: {list(df.columns)[:15]}")

        # 填充缺失列
        if 'code' not in df.columns:
            logger.error("缺少 code 列")
            return pd.DataFrame()
        if 'name' not in df.columns:
            df['name'] = ''
        if 'price' not in df.columns:
            df['price'] = 0
        if 'change_pct' not in df.columns:
            df['change_pct'] = 0
        if 'volume' not in df.columns:
            df['volume'] = 1
        if 'turnover' not in df.columns:
            df['turnover'] = 1

        # 转数值
        df['price'] = pd.to_numeric(df['price'], errors='coerce').fillna(0)
        df['change_pct'] = pd.to_numeric(df['change_pct'], errors='coerce').fillna(0)
        df['volume'] = pd.to_numeric(df['volume'], errors='coerce').fillna(1)
        df['turnover'] = pd.to_numeric(df['turnover'], errors='coerce').fillna(0)

        logger.info(f"price范围: {df['price'].min()}-{df['price'].max()}")
        logger.info(f"change_pct范围: {df['change_pct'].min()}-{df['change_pct'].max()}")
        logger.info(f"turnover范围: {df['turnover'].min()}-{df['turnover'].max()}")

        # 逐步筛选，打印每步剩余
        # 1. 排除 ST
        if 'name' in df.columns:
            before = len(df)
            df = df[~df['name'].astype(str).str.contains('ST|退市|N |C ', na=False)]
            logger.info(f"排除ST: {before} -> {len(df)}")

        # 2. 排除停牌
        before = len(df)
        df = df[df['volume'] > 0]
        logger.info(f"排除停牌: {before} -> {len(df)}")

        # 3. 涨跌幅
        before = len(df)
        df = df[(df['change_pct'] > -5) & (df['change_pct'] < 15)]
        logger.info(f"涨跌幅筛选: {before} -> {len(df)}")

        # 4. 换手率
        before = len(df)
        df = df[(df['turnover'] >= 0.1) & (df['turnover'] <= 50)]
        logger.info(f"换手率筛选: {before} -> {len(df)}")

        # 5. 价格
        before = len(df)
        df = df[df['price'] >= 2]
        logger.info(f"价格筛选: {before} -> {len(df)}")

        df['volume_ratio'] = 1.0
        df['scan_time'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        self.filtered_stocks = df
        logger.info(f"最终: {len(df)} 只")
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
