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
                        # 诊断：打印列名和数据样例
                        logger.info(f"列名: {list(df.columns)}")
                        logger.info(f"数据样例:\n{df.head(3).to_string()}")
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
        """快速筛选 - 修复版"""
        if df.empty:
            return df

        initial = len(df)
        logger.info(f"筛选前: {initial} 只")
        logger.info(f"DataFrame列名: {list(df.columns)}")

        # ============================================
        # 第1步：智能列名映射
        # ============================================
        rename_map = {}
        
        # akshare真实返回的中文列名
        if '代码' in df.columns:
            rename_map['代码'] = 'code'
        if '名称' in df.columns:
            rename_map['名称'] = 'name'
        if '最新价' in df.columns:
            rename_map['最新价'] = 'price'
        if '涨跌幅' in df.columns:
            rename_map['涨跌幅'] = 'change_pct'
        if '成交量' in df.columns:
            rename_map['成交量'] = 'volume'
        if '成交额' in df.columns:
            rename_map['成交额'] = 'amount'
        if '换手率' in df.columns:
            rename_map['换手率'] = 'turnover'
        if '总市值' in df.columns:
            rename_map['总市值'] = 'market_cap'
        if '市盈率-动态' in df.columns:
            rename_map['市盈率-动态'] = 'pe'
        
        # 兼容可能的英文列名
        if 'changepercent' in df.columns:
            rename_map['changepercent'] = 'change_pct'
        if 'trade' in df.columns:
            rename_map['trade'] = 'price'
        if 'turnoverratio' in df.columns:
            rename_map['turnoverratio'] = 'turnover'
        if 'mktcap' in df.columns:
            rename_map['mktcap'] = 'market_cap'
        
        if rename_map:
            df = df.rename(columns=rename_map)
            logger.info(f"重命名后列名: {list(df.columns)}")

        # ============================================
        # 第2步：清洗code列
        # ============================================
        if 'code' in df.columns:
            df['code'] = df['code'].astype(str).str.strip()
            before = len(df)
            df = df[df['code'].str.match(r'^\d{6}$', na=False)]
            logger.info(f"代码清洗: {before} → {len(df)} 只")
        else:
            logger.error("缺少'code'列，无法继续")
            return pd.DataFrame()

        # ============================================
        # 第3步：排除ST、退市、新股
        # ============================================
        if 'name' in df.columns:
            before = len(df)
            df['name'] = df['name'].astype(str).str.strip()
            mask_st = ~df['name'].str.contains('ST|退', na=False)
            mask_new = ~df['name'].str.match(r'^[NC]\s', na=False)
            df = df[mask_st & mask_new]
            logger.info(f"排除ST/新股: {before} → {len(df)} 只")

        # ============================================
        # 第4步：排除停牌
        # ============================================
        if 'volume' in df.columns:
            before = len(df)
            df['volume'] = pd.to_numeric(df['volume'], errors='coerce')
            nan_vol = df['volume'].isna().sum()
            df['volume'] = df['volume'].fillna(0)
            df = df[df['volume'] > 0]
            logger.info(f"排除停牌: {before} → {len(df)} 只 (NaN: {nan_vol})")
        else:
            logger.warning("缺少'volume'列，跳过停牌过滤")

        # ============================================
        # 第5步：涨跌幅过滤
        # ============================================
        if 'change_pct' in df.columns:
            before = len(df)
            df['change_pct'] = pd.to_numeric(df['change_pct'], errors='coerce')
            nan_pct = df['change_pct'].isna().sum()
            
            if nan_pct > before * 0.5:
                logger.warning(f"change_pct NaN占比过高({nan_pct}/{before})，跳过")
            else:
                df = df.dropna(subset=['change_pct'])
                df = df[(df['change_pct'] > -5) & (df['change_pct'] < 15)]
            
            logger.info(f"涨跌幅过滤: {before} → {len(df)} 只")
        else:
            logger.warning("缺少'change_pct'列，跳过涨跌幅过滤")

        # ============================================
        # 第6步：换手率过滤（关键修复）
        # ============================================
        if 'turnover' in df.columns:
            before = len(df)
            df['turnover'] = pd.to_numeric(df['turnover'], errors='coerce')
            nan_turn = df['turnover'].isna().sum()
            
            if nan_turn > before * 0.5:
                logger.warning(f"turnover NaN占比过高({nan_turn}/{before})，跳过")
            else:
                # 保留NaN，只过滤有值且不在范围内的
                valid_mask = df['turnover'].notna()
                in_range = (df['turnover'] >= 0.3) & (df['turnover'] <= 35)
                df = df[~valid_mask | in_range]
            
            logger.info(f"换手率过滤: {before} → {len(df)} 只 (NaN: {nan_turn})")
        else:
            logger.warning("缺少'turnover'列，跳过换手率过滤")

        # ============================================
        # 第7步：价格过滤
        # ============================================
        if 'price' in df.columns:
            before = len(df)
            df['price'] = pd.to_numeric(df['price'], errors='coerce')
            df = df.dropna(subset=['price'])
            df = df[df['price'] >= 2]
            logger.info(f"价格过滤: {before} → {len(df)} 只")
        else:
            logger.warning("缺少'price'列，跳过价格过滤")

        # ============================================
        # 第8步：补充字段
        # ============================================
        if 'volume_ratio' not in df.columns:
            df['volume_ratio'] = 1.0
        
        df['scan_time'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        self.filtered_stocks = df
        
        logger.info(f"✅ 最终: {len(df)} 只 (剔除{initial-len(df)}只)")
        
        if len(df) == 0:
            logger.error("❌ 筛选结果为0！检查上面的过滤步骤日志")
        
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
