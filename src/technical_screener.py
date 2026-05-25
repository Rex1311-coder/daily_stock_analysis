"""
技术指标计算 - 极简版
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

logger = logging.getLogger(__name__)


class TechnicalScreener:

    def __init__(self, max_workers: int = 10):
        self.max_workers = max_workers

    def batch_calculate(self, stock_list: List[Dict]) -> List[Dict]:
        if not stock_list:
            return []
        logger.info(f"技术指标: {len(stock_list)} 只")
        results = []
        success = 0
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {executor.submit(self._calc, s): s for s in stock_list}
            for i, f in enumerate(as_completed(futures)):
                try:
                    r = f.result(timeout=15)
                    if r:
                        results.append(r)
                        success += 1
                except:
                    pass
                if (i + 1) % 50 == 0:
                    logger.info(f"进度: {i+1}/{len(stock_list)} (成功{success})")
        logger.info(f"完成: {len(results)} 只")
        return results

    def _calc(self, stock: Dict) -> Dict:
        code = stock['code'].zfill(6)
        try:
            import akshare as ak
            end = datetime.now().strftime('%Y%m%d')
            start = (datetime.now() - timedelta(days=180)).strftime('%Y%m%d')
            df = ak.stock_zh_a_hist(symbol=code, period="daily", start_date=start, end_date=end, adjust="qfq")
            if df is None or df.empty or len(df) < 20:
                return None

            close = df['收盘'].values
            high = df['最高'].values
            low = df['最低'].values
            volume = df['成交量'].values
            latest_price = float(stock['price'])

            ma5 = np.mean(close[-5:])
            ma10 = np.mean(close[-10:])
            ma20 = np.mean(close[-20:])
            ma60 = np.mean(close[-60:]) if len(close) >= 60 else ma20

            is_bullish = ma5 > ma10 > ma20
            dist_ma20 = (latest_price - ma20) / ma20 * 100 if ma20 > 0 else 0

            # RSI
            delta = np.diff(close[-15:], prepend=close[-15])
            gain = np.where(delta > 0, delta, 0)
            loss = np.where(delta < 0, -delta, 0)
            avg_gain = np.mean(gain[-14:]) if len(gain) >= 14 else np.mean(gain)
            avg_loss = np.mean(loss[-14:]) if len(loss) >= 14 else np.mean(loss)
            rsi = 100 - (100 / (1 + avg_gain / avg_loss)) if avg_loss > 0 else 50

            # 量比
            vol_ma5 = np.mean(volume[-6:-1])
            vol_ratio = volume[-1] / vol_ma5 if vol_ma5 > 0 else 1

            score = 0
            if is_bullish: score += 30
            if 30 <= rsi <= 70: score += 20
            if 0.8 <= vol_ratio <= 3: score += 15
            if dist_ma20 < 10: score += 10

            stock.update({
                'ma5': round(ma5, 2), 'ma10': round(ma10, 2),
                'ma20': round(ma20, 2), 'ma60': round(ma60, 2),
                'rsi': round(rsi, 1),
                'technical_score': round(score, 1),
                'dist_from_ma20': round(dist_ma20, 1),
                'vol_ratio': round(vol_ratio, 2),
                'ideal_buy_price': round(ma20 * 0.97, 2),
                'stop_loss_price': round(ma60 * 0.95, 2),
                'is_bullish': is_bullish,
            })
            return stock
        except:
            return None

    def filter_top_stocks(self, stocks, top_n=50):
        valid = [s for s in stocks if s and 'technical_score' in s]
        valid.sort(key=lambda x: x['technical_score'], reverse=True)
        return valid[:top_n]
