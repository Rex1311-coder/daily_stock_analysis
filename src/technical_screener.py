"""
技术指标计算
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

logger = logging.getLogger(__name__)


class TechnicalScreener:

    def __init__(self, max_workers: int = 10):
        self.max_workers = max_workers
        self._err_count = 0
        self._err_printed = 0

    def batch_calculate(self, stock_list: List[Dict]) -> List[Dict]:
        if not stock_list:
            return []
        logger.info(f"技术指标: {len(stock_list)} 只")
        results = []
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {executor.submit(self._calc, s): s for s in stock_list}
            for i, f in enumerate(as_completed(futures)):
                try:
                    r = f.result(timeout=20)
                    if r:
                        results.append(r)
                except:
                    pass
                if (i + 1) % 50 == 0:
                    logger.info(f"进度: {i+1}/{len(stock_list)} (成功{len(results)})")
        logger.info(f"完成: {len(results)} 只, 错误: {self._err_count}")
        return results

    def _calc(self, stock: Dict) -> Dict:
        code = stock['code'].zfill(6)
        try:
            import akshare as ak
            end = datetime.now().strftime('%Y%m%d')
            start = (datetime.now() - timedelta(days=180)).strftime('%Y%m%d')

            df = None
            for attempt in range(2):
                try:
                    df = ak.stock_zh_a_hist(symbol=code, period="daily", start_date=start, end_date=end, adjust="qfq")
                    if df is not None and not df.empty:
                        break
                except Exception as e:
                    if attempt == 0:
                        time.sleep(1)

            if df is None or df.empty or len(df) < 20:
                return None

            close = df['收盘'].values
            volume = df['成交量'].values
            latest_price = float(stock['price'])

            ma5 = round(np.mean(close[-5:]), 2)
            ma10 = round(np.mean(close[-10:]), 2)
            ma20 = round(np.mean(close[-20:]), 2)
            ma60 = round(np.mean(close[-60:]), 2) if len(close) >= 60 else ma20

            is_bullish = ma5 > ma10 > ma20
            dist_ma20 = round((latest_price - ma20) / ma20 * 100, 1) if ma20 > 0 else 0

            delta = np.diff(close[-15:], prepend=close[-15])
            gain = np.where(delta > 0, delta, 0)
            loss = np.where(delta < 0, -delta, 0)
            avg_gain = np.mean(gain[-14:]) if len(gain) >= 14 else np.mean(gain)
            avg_loss = np.mean(loss[-14:]) if len(loss) >= 14 else np.mean(loss)
            rsi = round(100 - (100 / (1 + avg_gain / avg_loss)), 1) if avg_loss > 0 else 50

            vol_ma5 = np.mean(volume[-6:-1])
            vol_ratio = round(volume[-1] / vol_ma5, 2) if vol_ma5 > 0 else 1

            score = 0
            if is_bullish: score += 30
            if 30 <= rsi <= 70: score += 20
            if 0.8 <= vol_ratio <= 3: score += 15
            if dist_ma20 < 10: score += 10
            if latest_price > ma20: score += 10

            stock.update({
                'ma5': ma5, 'ma10': ma10, 'ma20': ma20, 'ma60': ma60,
                'rsi': rsi, 'technical_score': score,
                'dist_from_ma20': dist_ma20, 'vol_ratio': vol_ratio,
                'ideal_buy_price': round(ma20 * 0.97, 2) if ma20 > 0 else round(latest_price * 0.95, 2),
                'stop_loss_price': round(ma60 * 0.95, 2) if ma60 > 0 else round(latest_price * 0.90, 2),
                'is_bullish': is_bullish,
            })
            return stock
        except Exception as e:
            self._err_count += 1
            if self._err_printed < 5:
                self._err_printed += 1
                logger.warning(f"计算失败 {code}: {e}")
            return None

    def filter_top_stocks(self, stocks, top_n=50):
        valid = [s for s in stocks if s and 'technical_score' in s]
        valid.sort(key=lambda x: x['technical_score'], reverse=True)
        return valid[:top_n]
