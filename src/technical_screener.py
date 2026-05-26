"""
技术指标计算 - 增强版（修复symbol前缀 + 详细错误日志）
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
import random

logger = logging.getLogger(__name__)


class TechnicalScreener:
    """
    技术指标计算器
    
    功能：
    1. 批量计算股票技术指标（均线、RSI、MACD、KDJ、布林带等）
    2. 量价配合确认（7种状态识别）
    3. 综合技术面评分（6维度）
    4. 智能买卖价格计算
    5. API限流保护 + 详细错误日志
    """

    def __init__(self, max_workers: int = 5):
        """
        初始化技术筛选器
        
        Args:
            max_workers: 并发线程数（降低到5，避免API限流）
        """
        self.max_workers = max_workers
        self._err_count = 0
        self._err_printed = 0
        self._first_fail_logged = False

    # ============================================================
    # 公共方法
    # ============================================================

    def batch_calculate(self, stock_list: List[Dict]) -> List[Dict]:
        """
        批量计算技术指标（带限流保护）
        
        Args:
            stock_list: 股票基础信息列表
            
        Returns:
            增加技术指标字段的股票列表
        """
        if not stock_list:
            return []

        total = len(stock_list)
        logger.info(f"技术指标计算: {total} 只 (并发: {self.max_workers})")

        results = []
        self._first_fail_logged = False
        self._err_count = 0
        self._err_printed = 0

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {}
            for i, stock in enumerate(stock_list):
                if i > 0 and i % 10 == 0:
                    time.sleep(0.3)
                futures[executor.submit(self._calc_one, stock)] = stock

            for i, future in enumerate(as_completed(futures)):
                failed_stock = futures[future]
                
                try:
                    result = future.result(timeout=30)
                    if result:
                        results.append(result)
                except Exception as e:
                    self._err_count += 1
                    if self._err_printed < 3:
                        self._err_printed += 1
                        logger.warning(
                            f"线程异常: {failed_stock.get('code')} {failed_stock.get('name')} - {e}"
                        )

                processed = i + 1
                if processed % 50 == 0 or processed >= total:
                    logger.info(
                        f"技术进度: {processed}/{total} "
                        f"(成功{len(results)} 失败{self._err_count})"
                    )

        logger.info(f"技术指标完成: {len(results)}只 失败{self._err_count}只")
        return results

    def filter_top_stocks(
        self,
        stocks: List[Dict],
        top_n: int = 50,
        min_score: int = 40
    ) -> List[Dict]:
        """筛选技术面Top股票"""
        valid = [
            s for s in stocks
            if s and 'technical_score' in s and s['technical_score'] >= min_score
        ]

        valid.sort(key=lambda x: x['technical_score'], reverse=True)

        top = valid[:top_n]
        if top:
            logger.info(
                f"技术Top{top_n}: {len(top)}只 "
                f"(最高{top[0]['technical_score']}分 最低{top[-1]['technical_score']}分)"
            )
        else:
            logger.warning(f"技术Top: 无 (有效股票{len(valid)}只, 阈值{min_score}分)")

        return top

    # ============================================================
    # 私有方法：单只股票计算（带详细错误日志）
    # ============================================================

    def _calc_one(self, stock: Dict) -> Optional[Dict]:
        """
        计算单只股票的技术指标
        
        Args:
            stock: 股票基础信息
            
        Returns:
            增加技术指标后的股票字典，失败返回None
        """
        code = str(stock.get('code', '')).zfill(6)
        name = stock.get('name', '未知')

        try:
            time.sleep(random.uniform(0.05, 0.3))

            # 获取K线数据
            df = self._fetch_kline(code)
            
            if df is None:
                if not self._first_fail_logged:
                    logger.warning(f"❌ {code} {name}: _fetch_kline 返回 None")
                    self._first_fail_logged = True
                return None
            
            if len(df) < 20:
                if not self._first_fail_logged:
                    logger.warning(f"❌ {code} {name}: K线不足20条 (实际{len(df)}条)")
                    self._first_fail_logged = True
                return None

            if '收盘' not in df.columns:
                if not self._first_fail_logged:
                    logger.warning(f"❌ {code} {name}: 缺少'收盘'列, 列名={list(df.columns)}")
                    self._first_fail_logged = True
                return None

            # 提取数据
            close = df['收盘'].values
            high = df['最高'].values if '最高' in df.columns else close
            low = df['最低'].values if '最低' in df.columns else close
            volume = df['成交量'].values if '成交量' in df.columns else np.zeros(len(close))

            if len(close) == 0:
                return None

            latest_price = float(stock.get('price', close[-1]))
            change_pct = float(stock.get('change_pct', 0))
            turnover = float(stock.get('turnover', 0))

            # 计算各项指标
            indicators = {}
            indicators['change_pct'] = change_pct
            indicators['turnover'] = turnover
            indicators.update(self._calc_ma(close))
            indicators.update(self._calc_rsi(close))
            indicators.update(self._calc_macd(close))
            indicators.update(self._calc_kdj(high, low, close))
            indicators.update(self._calc_bollinger(close))
            indicators.update(self._calc_volume_metrics(volume))
            indicators.update(self._calc_price_position(latest_price, close))

            # 综合技术评分
            technical_score = self._calc_technical_score(indicators, latest_price)

            # 买卖价格
            buy_price, stop_loss, target1, target2, target3 = self._calc_trade_prices(
                indicators, latest_price
            )

            # 量价状态
            vol_price_status = self._get_volume_price_status(indicators)

            # 更新股票字段
            stock.update({
                'ma5': indicators.get('ma5', latest_price),
                'ma10': indicators.get('ma10', latest_price),
                'ma20': indicators.get('ma20', latest_price),
                'ma60': indicators.get('ma60', latest_price),
                'rsi': indicators.get('rsi', 50),
                'macd': indicators.get('macd', 0),
                'macd_signal': indicators.get('macd_signal', 0),
                'macd_hist': indicators.get('macd_hist', 0),
                'kdj_k': indicators.get('kdj_k', 50),
                'kdj_d': indicators.get('kdj_d', 50),
                'kdj_j': indicators.get('kdj_j', 50),
                'boll_upper': indicators.get('boll_upper', latest_price * 1.1),
                'boll_mid': indicators.get('boll_mid', latest_price),
                'boll_lower': indicators.get('boll_lower', latest_price * 0.9),
                'volume': float(stock.get('volume', volume[-1])),
                'avg_volume': indicators.get('avg_volume', volume[-1]),
                'vol_ratio': indicators.get('vol_ratio', 1.0),
                'dist_from_ma5': indicators.get('dist_ma5', 0),
                'dist_from_ma20': indicators.get('dist_ma20', 0),
                'is_bullish': indicators.get('is_bullish', False),
                'vol_price_status': vol_price_status,
                'technical_score': technical_score,
                'ideal_buy_price': buy_price,
                'stop_loss_price': stop_loss,
                'target1': target1,
                'target2': target2,
                'target3': target3,
            })

            return stock

        except Exception as e:
            self._err_count += 1
            if self._err_printed < 10:
                self._err_printed += 1
                logger.warning(f"💥 计算异常 {code} {name}: {type(e).__name__}: {e}")
            return None

    # ============================================================
    # 私有方法：数据获取（带前缀 + 详细日志）
    # ============================================================

    def _get_symbol(self, code: str) -> str:
        """
        根据股票代码自动判断市场并添加前缀
        
        Args:
            code: 6位纯数字代码
            
        Returns:
            带前缀的symbol，如 sh600519
        """
        code = str(code).zfill(6)
        
        if code.startswith('6'):
            return f"sh{code}"
        elif code.startswith('9'):
            return f"sh{code}"
        elif code.startswith('0'):
            return f"sz{code}"
        elif code.startswith('3'):
            return f"sz{code}"
        elif code.startswith('2'):
            return f"sz{code}"
        elif code.startswith('4'):
            return f"bj{code}"
        elif code.startswith('8'):
            return f"sh{code}"
        else:
            return f"sh{code}"

    def _fetch_kline(self, code: str) -> Optional[pd.DataFrame]:
        """
        获取历史K线数据（带详细错误日志）
        
        Args:
            code: 6位纯数字代码
            
        Returns:
            K线DataFrame，失败返回None
        """
        try:
            import akshare as ak

            symbol = self._get_symbol(code)
            end = datetime.now().strftime('%Y%m%d')
            start = (datetime.now() - timedelta(days=180)).strftime('%Y%m%d')

            for attempt in range(2):
                try:
                    df = ak.stock_zh_a_hist(
                        symbol=symbol,
                        period="daily",
                        start_date=start,
                        end_date=end,
                        adjust="qfq"
                    )

                    if df is not None and not df.empty:
                        if len(df) >= 20:
                            return df
                        else:
                            if not self._first_fail_logged:
                                logger.warning(f"⚠ {symbol}: K线不足20条({len(df)}条)")
                                self._first_fail_logged = True
                            return None
                    else:
                        if not self._first_fail_logged:
                            logger.warning(f"⚠ {symbol}: 返回空数据(None或empty)")
                            self._first_fail_logged = True
                        return None

                except Exception as e:
                    if attempt == 0:
                        time.sleep(2)
                    else:
                        if not self._first_fail_logged:
                            logger.warning(f"⚠ {symbol}: 获取失败 - {type(e).__name__}: {str(e)[:100]}")
                            self._first_fail_logged = True

            return None

        except ImportError:
            logger.error("❌ 未安装akshare库")
            return None
        except Exception as e:
            logger.warning(f"⚠ {code}: _fetch_kline异常 - {e}")
            return None

    # ============================================================
    # 私有方法：指标计算
    # ============================================================

    def _calc_ma(self, close: np.ndarray) -> Dict:
        result = {}
        n = len(close)
        if n >= 5:
            result['ma5'] = round(np.mean(close[-5:]), 2)
        if n >= 10:
            result['ma10'] = round(np.mean(close[-10:]), 2)
        if n >= 20:
            result['ma20'] = round(np.mean(close[-20:]), 2)
        if n >= 60:
            result['ma60'] = round(np.mean(close[-60:]), 2)
        else:
            result['ma60'] = result.get('ma20', close[-1])
        ma5 = result.get('ma5', close[-1])
        ma10 = result.get('ma10', close[-1])
        ma20 = result.get('ma20', close[-1])
        result['is_bullish'] = ma5 > ma10 > ma20
        return result

    def _calc_rsi(self, close: np.ndarray, period: int = 14) -> Dict:
        result = {'rsi': 50}
        n = len(close)
        if n < period + 1:
            return result
        delta = np.diff(close[-(period+1):])
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        avg_gain = np.mean(gain)
        avg_loss = np.mean(loss)
        if avg_loss > 0:
            rsi = 100 - (100 / (1 + avg_gain / avg_loss))
        else:
            rsi = 100 if avg_gain > 0 else 50
        result['rsi'] = round(rsi, 1)
        return result

    def _calc_macd(self, close: np.ndarray) -> Dict:
        result = {'macd': 0, 'macd_signal': 0, 'macd_hist': 0}
        n = len(close)
        if n < 35:
            return result
        ema12 = self._ema(close, 12)
        ema26 = self._ema(close, 26)
        dif = ema12 - ema26
        dea = self._ema(pd.Series(dif).values, 9)
        hist = 2 * (dif - dea)
        result['macd'] = round(dif[-1], 4)
        result['macd_signal'] = round(dea[-1], 4)
        result['macd_hist'] = round(hist[-1], 4)
        return result

    def _calc_kdj(self, high: np.ndarray, low: np.ndarray, close: np.ndarray, n: int = 9) -> Dict:
        result = {'kdj_k': 50, 'kdj_d': 50, 'kdj_j': 50}
        length = len(close)
        if length < n:
            return result
        recent_close = close[-n:]
        recent_high = high[-n:]
        recent_low = low[-n:]
        highest = np.max(recent_high)
        lowest = np.min(recent_low)
        if highest == lowest:
            return result
        rsv = (recent_close[-1] - lowest) / (highest - lowest) * 100
        k = 2/3 * 50 + 1/3 * rsv
        d = 2/3 * 50 + 1/3 * k
        j = 3 * k - 2 * d
        result['kdj_k'] = round(k, 1)
        result['kdj_d'] = round(d, 1)
        result['kdj_j'] = round(j, 1)
        return result

    def _calc_bollinger(self, close: np.ndarray, period: int = 20) -> Dict:
        result = {}
        n = len(close)
        if n < period:
            result['boll_mid'] = close[-1]
            result['boll_upper'] = close[-1] * 1.1
            result['boll_lower'] = close[-1] * 0.9
            return result
        mid = np.mean(close[-period:])
        std = np.std(close[-period:])
        result['boll_mid'] = round(mid, 2)
        result['boll_upper'] = round(mid + 2 * std, 2)
        result['boll_lower'] = round(mid - 2 * std, 2)
        return result

    def _calc_volume_metrics(self, volume: np.ndarray) -> Dict:
        result = {}
        n = len(volume)
        if n >= 6:
            avg_vol = np.mean(volume[-6:-1])
        elif n >= 2:
            avg_vol = np.mean(volume[:-1])
        else:
            avg_vol = volume[-1]
        latest_vol = volume[-1]
        result['avg_volume'] = round(float(avg_vol), 0)
        result['vol_ratio'] = round(latest_vol / avg_vol, 2) if avg_vol > 0 else 1.0
        return result

    def _calc_price_position(self, latest_price: float, close: np.ndarray) -> Dict:
        result = {}
        n = len(close)
        if n >= 5:
            ma5 = np.mean(close[-5:])
            result['dist_ma5'] = round((latest_price - ma5) / ma5 * 100, 1) if ma5 > 0 else 0
        else:
            result['dist_ma5'] = 0
        if n >= 20:
            ma20 = np.mean(close[-20:])
            result['dist_ma20'] = round((latest_price - ma20) / ma20 * 100, 1) if ma20 > 0 else 0
        else:
            result['dist_ma20'] = 0
        if n >= 60:
            high_60 = np.max(close[-60:])
            low_60 = np.min(close[-60:])
            if high_60 > low_60:
                result['position_60d'] = round(
                    (latest_price - low_60) / (high_60 - low_60) * 100, 1
                )
        return result

    # ============================================================
    # 量价状态
    # ============================================================

    def _get_volume_price_status(self, indicators: Dict) -> str:
        change_pct = indicators.get('change_pct', 0)
        vol_ratio = indicators.get('vol_ratio', 1)
        if change_pct > 1 and vol_ratio > 1.5:
            return "放量上涨 ✅"
        elif change_pct > 0.5 and vol_ratio > 1.2:
            return "温和放量上涨"
        elif change_pct > 1 and vol_ratio < 0.8:
            return "缩量上涨 ⚠诱多"
        elif change_pct < -1 and vol_ratio > 1.5:
            return "放量下跌 ⚠危险"
        elif change_pct < 0 and vol_ratio < 0.8:
            return "缩量下跌 正常调整"
        elif abs(change_pct) < 0.5 and vol_ratio > 2:
            return "放量滞涨 ⚠出货"
        elif abs(change_pct) < 0.5 and vol_ratio < 0.5:
            return "缩量横盘"
        else:
            return "量价正常"

    # ============================================================
    # 评分与价格
    # ============================================================

    def _calc_technical_score(self, indicators: Dict, latest_price: float) -> int:
        score = 0

        ma5 = indicators.get('ma5', latest_price)
        ma10 = indicators.get('ma10', latest_price)
        ma20 = indicators.get('ma20', latest_price)
        ma60 = indicators.get('ma60', latest_price)

        if ma5 > ma10 > ma20 > ma60:
            score += 25
        elif ma5 > ma10 > ma20:
            score += 20
        elif ma5 > ma20 and ma10 > ma20:
            score += 14
        elif ma5 < ma10 < ma20:
            score += 0
        else:
            score += 8

        rsi = indicators.get('rsi', 50)
        if 40 <= rsi <= 60:
            score += 15
        elif 30 <= rsi <= 70:
            score += 12
        elif rsi > 70:
            score += 6
        elif rsi < 30:
            score += 8
        else:
            score += 8

        macd = indicators.get('macd', 0)
        macd_signal = indicators.get('macd_signal', 0)
        macd_hist = indicators.get('macd_hist', 0)
        if macd > 0 and macd > macd_signal and macd_hist > 0:
            score += 15
        elif macd > 0 and macd > macd_signal:
            score += 12
        elif macd > macd_signal:
            score += 10
        elif macd > 0:
            score += 7
        else:
            score += 3

        change_pct = indicators.get('change_pct', 0)
        vol_ratio = indicators.get('vol_ratio', 1)
        turnover = indicators.get('turnover', 0)

        if change_pct > 1 and vol_ratio > 1.5:
            score += 20
        elif change_pct > 0.5 and vol_ratio > 1.2:
            score += 16
        elif change_pct > 1 and vol_ratio < 0.8:
            score += 6
        elif change_pct < -1 and vol_ratio > 1.5:
            score += 2
        elif change_pct < 0 and vol_ratio < 0.8:
            score += 12
        elif abs(change_pct) < 0.5 and vol_ratio > 2:
            score += 3
        else:
            score += 12

        if 1 <= turnover <= 8:
            pass
        elif turnover > 15:
            score -= 5
        elif 0 < turnover < 0.5:
            score -= 3

        dist_ma20 = indicators.get('dist_ma20', 0)
        if 0 <= dist_ma20 <= 3:
            score += 15
        elif 3 < dist_ma20 <= 5:
            score += 12
        elif -3 <= dist_ma20 < 0:
            score += 10
        elif dist_ma20 < -5:
            score += 5
        else:
            score += 6

        boll_upper = indicators.get('boll_upper', latest_price * 1.1)
        upside_potential = (boll_upper - latest_price) / latest_price * 100
        if upside_potential > 15:
            score += 10
        elif upside_potential > 10:
            score += 8
        elif upside_potential > 5:
            score += 6
        elif upside_potential > 3:
            score += 4
        else:
            score += 2

        return min(100, max(0, score))

    def _calc_trade_prices(self, indicators: Dict, latest_price: float) -> tuple:
        ma20 = indicators.get('ma20', latest_price)
        ma60 = indicators.get('ma60', latest_price)
        boll_upper = indicators.get('boll_upper', latest_price * 1.1)
        boll_lower = indicators.get('boll_lower', latest_price * 0.95)

        buy_price = round((ma20 * 0.98 + boll_lower) / 2, 2)
        if buy_price > latest_price:
            buy_price = round(latest_price * 0.98, 2)
        buy_price = max(buy_price, round(latest_price * 0.95, 2))

        stop1 = round(ma60 * 0.98, 2) if ma60 > 0 else round(latest_price * 0.93, 2)
        stop2 = round(boll_lower * 0.99, 2)
        stop3 = round(latest_price * 0.93, 2)
        stop_loss = min(stop1, stop2, stop3)
        stop_loss = min(stop_loss, round(latest_price * 0.95, 2))

        target1 = round(boll_upper, 2)
        if target1 < latest_price * 1.03:
            target1 = round(latest_price * 1.05, 2)

        band_width = boll_upper - boll_lower if boll_lower > 0 else latest_price * 0.1
        target2 = round(boll_upper + band_width * 0.3, 2)
        if target2 < latest_price * 1.06:
            target2 = round(latest_price * 1.08, 2)

        target3 = round(latest_price * 1.12, 2)

        return buy_price, stop_loss, target1, target2, target3

    @staticmethod
    def _ema(data: np.ndarray, period: int) -> np.ndarray:
        if len(data) < period:
            return np.full_like(data, np.mean(data))
        result = np.zeros_like(data)
        result[0] = data[0]
        multiplier = 2 / (period + 1)
        for i in range(1, len(data)):
            result[i] = (data[i] - result[i-1]) * multiplier + result[i-1]
        return result
