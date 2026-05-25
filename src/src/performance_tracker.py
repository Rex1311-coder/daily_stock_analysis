"""
推荐绩效追踪器 - 跟踪历史推荐的涨跌幅
"""
import sqlite3
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Dict
import logging
import os

logger = logging.getLogger(__name__)


class PerformanceTracker:

    def __init__(self, db_path: str = "data/recommendations.db"):
        self.db_path = db_path
        self._init()

    def _init(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS performance (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    code TEXT NOT NULL,
                    name TEXT,
                    rec_date TEXT NOT NULL,
                    rec_price REAL,
                    price_1d REAL, return_1d REAL,
                    price_3d REAL, return_3d REAL,
                    price_5d REAL, return_5d REAL,
                    price_10d REAL, return_10d REAL,
                    price_20d REAL, return_20d REAL,
                    is_win_5d INTEGER DEFAULT 0,
                    updated_at TEXT
                )
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_perf_code ON performance(code, rec_date)")

    def record_recommendations(self, recommendations: List[Dict], scan_time: str):
        """记录推荐时的价格"""
        with sqlite3.connect(self.db_path) as conn:
            for s in recommendations:
                conn.execute("""
                    INSERT OR REPLACE INTO performance 
                    (code, name, rec_date, rec_price, updated_at)
                    VALUES (?, ?, ?, ?, datetime('now','localtime'))
                """, (s['code'], s.get('name', ''), scan_time[:10], s.get('price', 0)))
            conn.commit()
        logger.info(f"记录 {len(recommendations)} 条推荐")

    def update_performance(self):
        """更新所有未结算的推荐表现"""
        with sqlite3.connect(self.db_path) as conn:
            # 获取未更新的记录
            df = pd.read_sql_query(
                "SELECT * FROM performance WHERE price_5d IS NULL",
                conn
            )
        
        if df.empty:
            logger.info("无待更新记录")
            return
        
        logger.info(f"更新 {len(df)} 条记录...")
        
        try:
            import akshare as ak
            
            for _, row in df.iterrows():
                code = row['code']
                rec_date = row['rec_date']
                rec_price = row['rec_price']
                
                try:
                    # 获取K线
                    kline = ak.stock_zh_a_hist(
                        symbol=code.zfill(6),
                        period="daily",
                        start_date=rec_date.replace('-', ''),
                        end_date=datetime.now().strftime('%Y%m%d'),
                        adjust="qfq"
                    )
                    
                    if kline is None or kline.empty:
                        continue
                    
                    kline = kline.rename(columns={'日期': 'date', '收盘': 'close'})
                    kline['date'] = pd.to_datetime(kline['date'])
                    kline = kline.sort_values('date')
                    
                    updates = {}
                    for days, col in [(1, 'price_1d'), (3, 'price_3d'), (5, 'price_5d'), (10, 'price_10d'), (20, 'price_20d')]:
                        if len(kline) > days:
                            price = kline['close'].values[days]
                            ret = round((price - rec_price) / rec_price * 100, 2)
                            updates[col] = price
                            updates[f'return_{days}d'] = ret
                            if days == 5:
                                updates['is_win_5d'] = 1 if ret > 0 else 0
                    
                    if updates:
                        updates['updated_at'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                        set_clause = ', '.join([f"{k}=?" for k in updates.keys()])
                        values = list(updates.values()) + [row['id']]
                        with sqlite3.connect(self.db_path) as conn:
                            conn.execute(f"UPDATE performance SET {set_clause} WHERE id=?", values)
                            conn.commit()
                            
                except Exception as e:
                    logger.debug(f"更新 {code} 失败: {e}")
            
        except Exception as e:
            logger.error(f"批量更新失败: {e}")

    def get_stats(self, days: int = 30) -> Dict:
        """获取统计"""
        with sqlite3.connect(self.db_path) as conn:
            total = conn.execute(
                "SELECT COUNT(*) FROM performance WHERE rec_date >= date('now', ?)",
                (f'-{days} days',)
            ).fetchone()[0]
            
            wins = conn.execute(
                "SELECT COUNT(*) FROM performance WHERE is_win_5d=1 AND rec_date >= date('now', ?)",
                (f'-{days} days',)
            ).fetchone()[0]
            
            avg_return = conn.execute(
                "SELECT AVG(return_5d) FROM performance WHERE return_5d IS NOT NULL AND rec_date >= date('now', ?)",
                (f'-{days} days',)
            ).fetchone()[0]
        
        win_rate = round(wins / total * 100, 1) if total > 0 else 0
        avg_return = round(avg_return, 2) if avg_return else 0
        
        return {
            'total': total,
            'wins': wins,
            'win_rate': win_rate,
            'avg_return_5d': avg_return,
        }

    def get_top_performers(self, limit: int = 10) -> pd.DataFrame:
        """历史表现最好的股票"""
        with sqlite3.connect(self.db_path) as conn:
            return pd.read_sql_query(
                f"""SELECT code, name, COUNT(*) as times, 
                    AVG(return_5d) as avg_return, 
                    SUM(is_win_5d)*100.0/COUNT(*) as win_rate
                    FROM performance 
                    WHERE return_5d IS NOT NULL
                    GROUP BY code 
                    HAVING times >= 2
                    ORDER BY win_rate DESC 
                    LIMIT {limit}""",
                conn
            )
