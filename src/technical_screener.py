"""
技术指标批量计算器 - 优化版（批量K线 + 缓存）
"""
import pandas as pd
import numpy as np
import sqlite3
from datetime import datetime, timedelta
from typing import List, Dict
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
import os

logger = logging.getLogger(__name__)


class TechnicalScreener:

    def __init__(self, max_workers: int = 10):
        self.max_workers = max_workers
        self._kline_cache = {}
        self._db_path = "data/kline_cache.db"
        self._init_db()

    def _init_db(self):
        """初始化K线缓存数据库"""
        os.makedirs("data", exist_ok=True)
        with sqlite3.connect(self._db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS kline (
                    code TEXT,
                    date TEXT,
                    open REAL,
                    high REAL,
                    low REAL,
                    close REAL,
                    volume REAL,
                    PRIMARY KEY (code, date)
                )
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_code ON kline(code)")

    def _load_cache_batch(self, codes: List[str]):
        """批量从缓存加载K线"""
        if not codes:
            return
        placeholders = ','.join(['?' for _ in codes])
        with sqlite3.connect(self._db_path) as conn:
            df = pd.read_sql_query(
                f"SELECT * FROM kline WHERE code IN ({placeholders}) AND date >= ?",
                conn,
                params=codes + [(datetime.now() - timedelta(days=180)).strftime('%Y-%m-%d')]
            )
        if not df.empty:
            for code in df['code'].unique():
                self._kline_cache[code] = df[df['code'] == code].sort_values('date')

    def _save_cache_batch(self, data: dict):
        """批量保存K线到缓存"""
        rows = []
        for code, df in data.items():
            if df is None or df.empty:
                continue
            for _, row in df.iterrows():
                rows.append((
                    code,
                    str(row['date'])[:10],
                    float(row['open']),
                    float(row['high']),
                    float(row['low']),
                    float(row['close']),
                    float(row['volume']),
                ))
        if rows:
            with sqlite3.connect(self._db_path) as conn:
                conn.executemany(
                    "INSERT OR REPLACE INTO kline VALUES (?,?,?,?,?,?,?)",
                    rows
                )

    def batch_calculate(self, stock_list: List[Dict]) -> List[Dict]:
        if not stock_list:
            return []
        
        logger.info(f"批量技术指标: {len(stock_list)} 只")
        codes = [s['code'].zfill(6) for s in stock_list]
        
        # 1. 从缓存加载
        logger.info("从缓存加载K线...")
        self._load_cache_batch(codes)
        cached = len(self._kline_cache)
        logger.info(f"缓存命中: {cached}/{len(codes)} 只")
        
        # 2. 获取未缓存的K线
        uncached = [c for c in codes if c not in self._kline_cache]
        if uncached:
            logger.info(f"获取 {len(uncached)} 只K线数据...")
            new_data = self._fetch_kline_batch(uncached)
            self._kline_cache.update(new_data)
            self._save_cache_batch(new_data)
        
        # 3. 并行计算指标
        logger.info("计算技术指标...")
        results = []
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {executor.submit(self._calc, s): s for s in stock_list}
            for i, f in enumerate(as_completed(futures)):
                try:
                    r = f.result(timeout=10)
                    if r:
                        results.append(r)
                except:
                    pass
                if (i + 1) % 200 == 0:
                    logger.info(f"进度: {i+1}/{len(stock_list)}")
        
        logger.info(f"完成: {len(results)} 只")
        return results

    def _fetch_kline_batch(self, codes: List[str]) -> dict:
        """批量获取K线数据"""
        result = {}
        batch_size = 50
        for i in range(0, len(codes), batch_size):
            batch = codes[i:i+batch_size]
            try:
                import akshare as ak
                end = datetime.now().strftime('%Y%m%d')
                start = (datetime.now() - timedelta(days=180)).strftime('%Y%m%d')
                
                for code in batch:
                    try:
                        df = ak.stock_zh_a_hist(
                            symbol=code, period="daily",
                            start_date=start, end_date=end, adjust="qfq"
                        )
                        if df is not None and not df.empty:
                            df = df.rename(columns={
                                '日期': 'date', '开盘': 'open',
                                '收盘': 'close', '最高': 'high',
                                '最低': 'low', '成交量': 'volume'
                            })
                            result[code] = df
                    except:
                        pass
                
                logger.info(f"批量获取: {i+1}-{min(i+batch_size, len(codes))}/{len(codes)}")
            except:
                pass
        
        return result

        def _calc(self, stock: Dict) -> Dict:
        code = stock['code'].zfill(6)
        df = self._kline_cache.get(code)
        
        if df is None or len(df) < 20:
            return None
        
        # 确保列存在
        if 'close' not in df.columns:
            if '收盘' in df.columns:
                df = df.rename(columns={'收盘': 'close', '开盘': 'open', '最高': 'high', '最低': 'low', '成交量': 'volume'})
            else:
                return None
        
        # 确保是数值
        for col in ['close', 'high', 'low', 'volume']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        close = df['close'].values
        high = df['high'].values
        low = df['low'].values
        volume = df['volume'].values
        latest_price = float(stock['price'])
        
        # 均线
        ma5 = self._ma(close, 5)[-1]
        ma10 = self._ma(close, 10)[-1]
        ma20 = self._ma(close, 20)[-1]
        ma60 = self._ma(close, 60)[-1] if len(close) >= 60 else ma20
        
        is_bullish = ma5 > ma10 > ma20
        ma_score = 30 if (ma5 > ma10 > ma20 > ma60) else (20 if is_bullish else (10 if latest_price > ma20 else 0))
        dist_ma20 = (latest_price - ma20) / ma20 * 100 if ma20 > 0 else 0
        
        # MACD
        macd, signal, hist = self._macd(close)
        macd_score = 25 if (macd[-1] > 0 and macd[-1] > signal[-1] and hist[-1] > hist[-2]) else \
                     (15 if macd[-1] > signal[-1] else (8 if hist[-1] > hist[-2] else 0))
        
        # RSI
        rsi = self._rsi(close, 14)[-1]
        rsi_score = 20 if 25 <= rsi < 30 else (15 if 30 <= rsi <= 70 else (10 if rsi < 25 else 5))
        
        # 量价
        vol_ma5 = self._ma(volume, 5)[-1]
        vol_ratio = volume[-1] / vol_ma5 if vol_ma5 > 0 else 1
        vol_score = 15 if 1.2 <= vol_ratio <= 2.5 else (8 if 0.8 <= vol_ratio <= 1.2 else 3)
        
        total = ma_score + macd_score + rsi_score + vol_score
        
        stock.update({
            'ma5': round(ma5, 2), 'ma10': round(ma10, 2),
            'ma20': round(ma20, 2), 'ma60': round(ma60, 2),
            'rsi': round(rsi, 1),
            'technical_score': round(total, 1),
            'dist_from_ma20': round(dist_ma20, 1),
            'vol_ratio': round(vol_ratio, 2),
            'ideal_buy_price': round(ma20 * 0.97, 2) if ma20 > 0 else round(latest_price * 0.95, 2),
            'stop_loss_price': round(ma60 * 0.95, 2) if ma60 > 0 else round(latest_price * 0.90, 2),
            'is_bullish': is_bullish,
        })
        return stock

    def _ma(self, data, p):
        r = np.full_like(data, np.nan, dtype=float)
        for i in range(p-1, len(data)):
            r[i] = np.mean(data[i-p+1:i+1])
        return r

    def _macd(self, c, f=12, s=26, sig=9):
        ef = pd.Series(c).ewm(span=f, adjust=False).mean().values
        es = pd.Series(c).ewm(span=s, adjust=False).mean().values
        m = ef - es
        sg = pd.Series(m).ewm(span=sig, adjust=False).mean().values
        return m, sg, m - sg

    def _rsi(self, c, p=14):
        d = np.diff(c, prepend=c[0])
        g = np.where(d > 0, d, 0)
        l = np.where(d < 0, -d, 0)
        r = np.full_like(c, np.nan, dtype=float)
        for i in range(p, len(c)):
            ag = np.mean(g[i-p+1:i+1])
            al = np.mean(l[i-p+1:i+1])
            r[i] = 100 - (100/(1+ag/al)) if al > 0 else 100
        return r

    def filter_top_stocks(self, stocks, top_n=100):
        valid = [s for s in stocks if s and 'technical_score' in s]
        valid.sort(key=lambda x: x['technical_score'], reverse=True)
        return valid[:top_n]
