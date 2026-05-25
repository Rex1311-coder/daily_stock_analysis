"""
全市场股票扫描器 - 获取全部A股实时行情并快速筛选
"""
import pandas as pd
import numpy as np
from datetime import datetime
from typing import List, Dict
import logging
import time

logger = logging.getLogger(__name__)


class MarketScanner:
    """全市场股票扫描器"""

    def __init__(self):
        self.all_stocks = None
        self.filtered_stocks = None

    def fetch_all_stocks(self) -> pd.DataFrame:
        """获取全A股实时行情"""
        try:
            import akshare as ak

            logger.info("正在获取全A股实时行情...")
            df = None

            # 新浪接口
            try:
                logger.info("尝试新浪接口...")
                df = ak.stock_zh_a_spot()
                if df is not None and not df.empty:
                    logger.info(f"新浪接口成功: {len(df)} 只")
            except Exception as e:
                logger.warning(f"新浪接口失败: {e}")

            # 东财接口备用
            if df is None or df.empty:
                logger.info("尝试东财接口...")
                for attempt in range(2):
                    try:
                        df = ak.stock_zh_a_spot_em()
                        if df is not None and not df.empty:
                            logger.info(f"东财接口成功: {len(df)} 只")
                            break
                    except Exception as e:
                        logger.warning(f"东财第{attempt+1}次失败: {e}")
                        time.sleep(3)

            if df is None or df.empty:
                logger.error("所有接口均失败")
                return pd.DataFrame()

            self.all_stocks = df
            return df

        except Exception as e:
            logger.error(f"获取全市场股票失败: {e}")
            return pd.DataFrame()

    def quick_filter(self, df: pd.DataFrame) -> pd.DataFrame:
        """快速筛选"""
        if df.empty:
            return df

        initial_count = len(df)
        logger.info(f"开始快速筛选，原始: {initial_count} 只")

        col_map = {
            '代码': 'code', '名称': 'name', '最新价': 'price',
            '涨跌幅': 'change_pct', '成交量': 'volume', '成交额': 'amount',
            '换手率': 'turnover', '市盈率-动态': 'pe', '总市值': 'market_cap',
            '量比': 'volume_ratio',
        }
        df = df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})

        if 'name' in df.columns:
            df = df[~df['name'].str.contains('ST|退市|N |C ', na=False)]

        if 'volume' in df.columns:
            df['volume'] = pd.to_numeric(df['volume'], errors='coerce')
            df = df[df['volume'] > 0]

        if 'change_pct' in df.columns:
            df['change_pct'] = pd.to_numeric(df['change_pct'], errors='coerce')
            df = df[(df['change_pct'] > -3) & (df['change_pct'] < 10)]

        if 'turnover' in df.columns:
            df['turnover'] = pd.to_numeric(df['turnover'], errors='coerce')
            df = df[(df['turnover'] >= 0.5) & (df['turnover'] <= 30)]

        if 'price' in df.columns:
            df['price'] = pd.to_numeric(df['price'], errors='coerce')
            df = df[df['price'] >= 3]

        if 'market_cap' in df.columns:
            df['market_cap'] = pd.to_numeric(df['market_cap'], errors='coerce')
            df = df[df['market_cap'] >= 2e9]

        if 'volume_ratio' not in df.columns:
            df['volume_ratio'] = 1.0

        df['scan_time'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        self.filtered_stocks = df
        logger.info(f"快速筛选完成: {initial_count} -> {len(df)} 只")
        return df

    def get_stock_list(self) -> List[Dict]:
        """获取筛选后的股票列表"""
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
