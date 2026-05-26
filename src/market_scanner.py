"""
全市场股票扫描器 - 优化版（兼容akshare各种版本）
"""
import pandas as pd
import numpy as np
from datetime import datetime
from typing import List, Dict, Optional
import logging
import time

logger = logging.getLogger(__name__)


class MarketScanner:

    def __init__(self):
        self.all_stocks: Optional[pd.DataFrame] = None
        self.filtered_stocks: Optional[pd.DataFrame] = None

    def fetch_all_stocks(self) -> pd.DataFrame:
        """
        获取全A股实时行情数据
        
        Returns:
            pd.DataFrame: 包含所有A股实时行情的DataFrame
        """
        try:
            import akshare as ak
            logger.info("获取全A股行情...")
            
            for attempt in range(3):
                try:
                    df = ak.stock_zh_a_spot()
                    if df is not None and not df.empty:
                        # 诊断信息：记录列名和数据样例
                        logger.info(f"akshare返回列名: {list(df.columns)}")
                        logger.info(f"数据样例:\n{df.head(3).to_string()}")
                        logger.info(f"成功获取: {len(df)} 只股票")
                        self.all_stocks = df
                        return df
                except Exception as e:
                    logger.warning(f"第{attempt+1}次获取失败: {e}")
                    if attempt < 2:
                        time.sleep(5)
            
            logger.error("3次重试均失败，返回空DataFrame")
            return pd.DataFrame()
            
        except ImportError:
            logger.error("未安装akshare库，请执行: pip install akshare")
            return pd.DataFrame()
        except Exception as e:
            logger.error(f"获取全市场股票失败: {e}")
            return pd.DataFrame()

    def quick_filter(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        快速筛选股票（多级过滤）
        
        过滤条件：
        1. 代码标准化（提取6位数字）
        2. 排除ST/退市/新股
        3. 排除停牌（成交量=0）
        4. 涨跌幅区间过滤
        5. 价格下限过滤
        
        Args:
            df: 原始行情DataFrame
            
        Returns:
            pd.DataFrame: 筛选后的DataFrame
        """
        if df.empty:
            logger.warning("输入DataFrame为空，跳过筛选")
            return df

        initial = len(df)
        logger.info(f"筛选前: {initial} 只")
        
        # ============================================
        # 第1步：智能列名映射
        # ============================================
        df = self._rename_columns(df)
        logger.info(f"可用列: {list(df.columns)}")
        
        # ============================================
        # 第2步：代码标准化
        # ============================================
        df = self._clean_stock_code(df)
        if df.empty:
            return df
        
        # ============================================
        # 第3步：排除ST/退市/新股
        # ============================================
        df = self._filter_special_stocks(df)
        if df.empty:
            return df
        
        # ============================================
        # 第4步：排除停牌
        # ============================================
        df = self._filter_suspended(df)
        if df.empty:
            return df
        
        # ============================================
        # 第5步：涨跌幅过滤
        # ============================================
        df = self._filter_change_pct(df)
        if df.empty:
            return df
        
        # ============================================
        # 第6步：价格过滤
        # ============================================
        df = self._filter_price(df)
        if df.empty:
            return df
        
        # ============================================
        # 第7步：补充缺失字段
        # ============================================
        df = self._fill_missing_fields(df)
        
        # ============================================
        # 最终结果
        # ============================================
        df['scan_time'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        self.filtered_stocks = df
        
        removed = initial - len(df)
        retention = len(df) / initial * 100 if initial > 0 else 0
        
        logger.info(f"✅ 筛选完成: {len(df)} 只 (保留率 {retention:.1f}%, 剔除 {removed} 只)")
        
        if len(df) == 0:
            logger.error("❌ 筛选结果为0！可能原因：")
            logger.error("  1. 所有股票均不满足筛选条件")
            logger.error("  2. 当前为非交易时段（数据可能异常）")
            logger.error("  3. akshare返回数据格式发生变化")
        
        return df

    def get_stock_list(self) -> List[Dict]:
        """
        将筛选后的DataFrame转换为字典列表
        
        Returns:
            List[Dict]: 股票信息列表
        """
        if self.filtered_stocks is None or self.filtered_stocks.empty:
            return []
        
        stocks = []
        for _, row in self.filtered_stocks.iterrows():
            try:
                stock = {
                    'code': str(row.get('code', '')).strip(),
                    'name': str(row.get('name', ''))[:8],
                    'price': round(float(row.get('price', 0)), 2),
                    'change_pct': round(float(row.get('change_pct', 0)), 2),
                    'volume': float(row.get('volume', 0)),
                    'amount': float(row.get('amount', 0)),
                    'turnover': round(float(row.get('turnover', 0)), 2),
                    'market_cap': float(row.get('market_cap', 0)),
                    'volume_ratio': float(row.get('volume_ratio', 1.0)),
                }
                
                # 过滤无效代码
                if not stock['code']:
                    continue
                    
                stocks.append(stock)
                
            except Exception as e:
                logger.warning(f"转换股票数据失败: {e}")
                continue
        
        logger.info(f"生成股票列表: {len(stocks)} 只")
        return stocks

    # ============================================
    # 私有方法：各过滤步骤
    # ============================================

    def _rename_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        智能列名映射（兼容akshare不同版本的列名）
        """
        rename_map = {}
        
        # akshare常见列名映射表
        column_mapping = {
            '代码': 'code',
            '名称': 'name',
            '最新价': 'price',
            '涨跌幅': 'change_pct',
            '涨跌额': 'change_amount',
            '成交量': 'volume',
            '成交额': 'amount',
            '换手率': 'turnover',
            '总市值': 'market_cap',
            '流通市值': 'float_market_cap',
            '市盈率-动态': 'pe_dynamic',
            '市盈率-静态': 'pe_static',
            '市净率': 'pb',
            '量比': 'volume_ratio',
            '今开': 'open',
            '昨收': 'pre_close',
            '最高': 'high',
            '最低': 'low',
            '买入': 'bid',
            '卖出': 'ask',
            '60日涨跌幅': 'change_pct_60d',
            '年初至今涨跌幅': 'change_pct_ytd',
        }
        
        for original, target in column_mapping.items():
            if original in df.columns:
                rename_map[original] = target
        
        if rename_map:
            df = df.rename(columns=rename_map)
            logger.info(f"列名映射完成: {len(rename_map)} 个字段")
        
        return df

    def _clean_stock_code(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        清洗股票代码（提取6位纯数字）
        
        处理格式：
        - sh600519 → 600519
        - bj920000 → 920000
        - 600519 → 600519
        """
        if 'code' not in df.columns:
            logger.error("❌ 缺少'code'列，无法继续筛选")
            return pd.DataFrame()
        
        before = len(df)
        
        # 转换为字符串并去除空格
        df['code'] = df['code'].astype(str).str.strip()
        
        # 提取6位数字代码
        df['code'] = df['code'].str.extract(r'(\d{6})', expand=False)
        
        # 删除无法提取代码的行
        invalid_codes = df['code'].isna().sum()
        df = df.dropna(subset=['code'])
        
        logger.info(f"代码标准化: {before} → {len(df)} 只 (无效代码: {invalid_codes})")
        return df

    def _filter_special_stocks(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        排除ST、*ST、退市、新股(N)、次新股(C)
        """
        if 'name' not in df.columns:
            logger.warning("缺少'name'列，跳过特殊股票过滤")
            return df
        
        before = len(df)
        df['name'] = df['name'].astype(str).str.strip()
        
        # 排除ST和退市
        mask_st = ~df['name'].str.contains('ST|退', na=False, case=False)
        
        # 排除新股和次新股（以N或C开头）
        mask_new = ~df['name'].str.match(r'^[NC]\s', na=False)
        
        df = df[mask_st & mask_new]
        
        removed = before - len(df)
        logger.info(f"排除ST/新股: {before} → {len(df)} 只 (剔除 {removed})")
        return df

    def _filter_suspended(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        排除停牌股票（成交量=0或NaN）
        """
        if 'volume' not in df.columns:
            logger.warning("缺少'volume'列，跳过停牌过滤")
            return df
        
        before = len(df)
        
        # 转换为数值
        df['volume'] = pd.to_numeric(df['volume'], errors='coerce')
        
        # 统计NaN数量
        nan_volume = df['volume'].isna().sum()
        
        # NaN填充为0后过滤
        df['volume'] = df['volume'].fillna(0)
        df = df[df['volume'] > 0]
        
        removed = before - len(df)
        logger.info(f"排除停牌: {before} → {len(df)} 只 (停牌/NaN: {removed})")
        return df

    def _filter_change_pct(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        涨跌幅过滤（-5% ~ +15%）
        
        排除：
        - 跌幅超过5%的（短期风险较大）
        - 涨幅超过15%的（科创板/创业板涨停板，或异常波动）
        """
        if 'change_pct' not in df.columns:
            logger.warning("缺少'change_pct'列，跳过涨跌幅过滤")
            return df
        
        before = len(df)
        
        df['change_pct'] = pd.to_numeric(df['change_pct'], errors='coerce')
        nan_pct = df['change_pct'].isna().sum()
        
        # 删除NaN
        df = df.dropna(subset=['change_pct'])
        
        # 过滤涨跌幅区间
        df = df[(df['change_pct'] > -5) & (df['change_pct'] < 15)]
        
        removed = before - len(df)
        logger.info(f"涨跌幅过滤: {before} → {len(df)} 只 (剔除 {removed}, NaN: {nan_pct})")
        return df

    def _filter_price(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        价格过滤（>= 2元）
        
        排除低价股（流动性差、退市风险高）
        """
        if 'price' not in df.columns:
            logger.warning("缺少'price'列，跳过价格过滤")
            return df
        
        before = len(df)
        
        df['price'] = pd.to_numeric(df['price'], errors='coerce')
        nan_price = df['price'].isna().sum()
        
        # 删除NaN
        df = df.dropna(subset=['price'])
        
        # 过滤低价股
        df = df[df['price'] >= 2]
        
        removed = before - len(df)
        logger.info(f"价格过滤: {before} → {len(df)} 只 (剔除 {removed}, NaN: {nan_price})")
        return df

    def _fill_missing_fields(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        补充缺失字段（某些akshare版本不返回的字段）
        """
        default_values = {
            'turnover': 0.0,
            'market_cap': 0.0,
            'float_market_cap': 0.0,
            'pe_dynamic': 0.0,
            'pe_static': 0.0,
            'pb': 0.0,
            'volume_ratio': 1.0,
            'open': 0.0,
            'high': 0.0,
            'low': 0.0,
        }
        
        for field, default in default_values.items():
            if field not in df.columns:
                df[field] = default
        
        # 如果有今开/昨收，计算量比（可选）
        if 'volume_ratio' in df.columns and df['volume_ratio'].eq(1.0).all():
            if 'volume' in df.columns and len(df) > 0:
                # 简化处理：量比默认1.0（盘中无法准确计算）
                pass
        
        return df


# ============================================
# 使用示例
# ============================================
if __name__ == "__main__":
    # 配置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # 创建扫描器
    scanner = MarketScanner()
    
    # 获取全市场股票
    df = scanner.fetch_all_stocks()
    
    if not df.empty:
        # 快速筛选
        filtered_df = scanner.quick_filter(df)
        
        # 获取股票列表
        stock_list = scanner.get_stock_list()
        
        print(f"\n{'='*60}")
        print(f"筛选结果: {len(stock_list)} 只")
        
        if stock_list:
            # 打印前5只
            print(f"\n前5只股票:")
            print(f"{'代码':<10} {'名称':<10} {'价格':>8} {'涨跌幅':>8}")
            print("-" * 40)
            for stock in stock_list[:5]:
                print(
                    f"{stock['code']:<10} "
                    f"{stock['name']:<10} "
                    f"{stock['price']:>8.2f} "
                    f"{stock['change_pct']:>+7.2f}%"
                )
    else:
        print("获取数据失败")
