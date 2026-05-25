"""
历史推荐存储
"""
import sqlite3
import pandas as pd
from datetime import datetime
from typing import List, Dict
import logging
import os

logger = logging.getLogger(__name__)


class RecommendationStore:
    
    def __init__(self, db_path: str = "data/recommendations.db"):
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self.db_path = db_path
        self._init()
    
    def _init(self):
        with sqlite3.connect(self.db_path) as c:
            c.execute("""CREATE TABLE IF NOT EXISTS recommendations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                scan_time TEXT, rank INTEGER, code TEXT, name TEXT,
                price REAL, change_pct REAL, turnover REAL,
                ma5 REAL, ma20 REAL, ma60 REAL, rsi REAL,
                technical_score REAL, ai_score REAL, final_score REAL,
                ideal_buy_price REAL, stop_loss_price REAL,
                ai_action TEXT, ai_confidence INTEGER, ai_reason TEXT,
                risk_level TEXT, is_bullish INTEGER,
                created_at TEXT DEFAULT (datetime('now','localtime'))
            )""")
            c.execute("CREATE INDEX IF NOT EXISTS idx_time ON recommendations(scan_time)")
            c.execute("CREATE INDEX IF NOT EXISTS idx_code ON recommendations(code, scan_time)")
    
    def save_batch(self, recs: List[Dict], scan_time: str = None):
        if not scan_time:
            scan_time = datetime.now().strftime('%Y-%m-%d %H:%M')
        if not recs:
            return
        with sqlite3.connect(self.db_path) as c:
            for i, s in enumerate(recs):
                c.execute("""INSERT INTO recommendations 
                    (scan_time,rank,code,name,price,change_pct,turnover,
                     ma5,ma20,ma60,rsi,technical_score,ai_score,final_score,
                     ideal_buy_price,stop_loss_price,ai_action,ai_confidence,
                     ai_reason,risk_level,is_bullish)
                    VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                    (scan_time, i+1, str(s.get('code','')), str(s.get('name','')),
                     s.get('price',0), s.get('change_pct',0), s.get('turnover',0),
                     s.get('ma5',0), s.get('ma20',0), s.get('ma60',0), s.get('rsi',0),
                     s.get('technical_score',0), s.get('ai_score',0), s.get('final_score',0),
                     s.get('ideal_buy_price',0), s.get('stop_loss_price',0),
                     str(s.get('ai_action','hold')), s.get('ai_confidence',0),
                     str(s.get('ai_reason','')), str(s.get('risk_level','中')),
                     1 if s.get('is_bullish') else 0))
            c.commit()
        logger.info(f"保存 {len(recs)} 条推荐")
    
    def get_latest(self, limit=20):
        with sqlite3.connect(self.db_path) as c:
            t = c.execute("SELECT MAX(scan_time) FROM recommendations").fetchone()[0]
            if t:
                return pd.read_sql_query(f"SELECT * FROM recommendations WHERE scan_time=? ORDER BY rank LIMIT {limit}", c, params=(t,))
        return pd.DataFrame()
    
    def get_history(self, days=7):
        with sqlite3.connect(self.db_path) as c:
            return pd.read_sql_query(f"SELECT * FROM recommendations WHERE scan_time>=datetime('now','-{days} days','localtime') ORDER BY scan_time DESC, rank", c)
    
    def export_csv(self, path=None):
        df = self.get_latest()
        if df.empty:
            return ""
        if not path:
            path = f"reports/recommend_{datetime.now().strftime('%Y%m%d_%H%M')}.csv"
        os.makedirs(os.path.dirname(path), exist_ok=True)
        df.to_csv(path, index=False, encoding='utf-8-sig')
        return path
