"""
技术指标计算 - 增强版
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

logger = logging.getLogger(__name__)


class TechnicalScreener:
    """
    技术指标计算器
    
    功能：
    1. 批量计算股票技术指标（均线、RSI、MACD、KDJ等）
    2. 综合技术面评分
    3. 智能买入/止损价格计算
    4. 过滤Top股票
    """

    def __init__(self, max_workers: int = 10):
        """
        初始化技术筛选器
        
        Args:
            max_workers: 并发线程数
        """
        self.max_workers = max_workers
        self._err_count = 0
        self._err_printed = 0

    # ============================================================
    # 公共方法
    # ============================================================

    def batch_calculate(self, stock_list: List[Dict]) -> List[Dict]:
        """
        批量计算技术指标
        
        Args:
            stock_list: 股票基础信息列表
            
        Returns:
            增加技术指标字段的股票列表
        """
        if not stock_list:
            return []

        total = len(stock_list)
        logger.info(f"技术指标计算: {total} 只")

        results = []

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {executor.submit(self._calc_one, s): s for s in stock_list}

            for i, future in enumerate(as_completed(futures)):
                try:
                    result = future.result(timeout=20)
                    if result:
                        results.append(result)
                except Exception:
                    pass

                # 进度报告
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
        """
        筛选技术面Top股票
        
        Args:
            stocks: 包含技术指标的股票列表
            top_n: 返回数量
            min_score: 最低技术评分
            
        Returns:
            Top股票列表
        """
        # 过滤有效且有评分的
        valid = [
            s for s in stocks
            if s and 'technical_score' in s and s['technical_score'] >= min_score
        ]

        # 按技术评分降序
        valid.sort(key=lambda x: x['technical_score'], reverse=True)

        top = valid[:top_n]
        logger.info(
            f"技术Top{top_n}: {len(top)}只 "
            f"(最高{top[0]['technical_score']}分 最低{top[-1]['technical_score']}分)"
            if top else "技术Top: 无"
        )

        return top

    # ============================================================
    # 私有方法：单只股票计算
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

        try:
            # 获取历史K线数据
            df = self._fetch_kline(code)
            if df is None or len(df) < 20:
                logger.debug(f"{code} 历史数据不足")
                return None

            # 提取价格和成交量序列
            close = df['收盘'].values
            high = df['最高'].values
            low = df['最低'].values
            volume = df['成交量'].values

            latest_price = float(stock.get('price', close[-1]))

            # 计算各项指标
            indicators = {}
            indicators.update(self._calc_ma(close))
            indicators.update(self._calc_rsi(close))
            indicators.update(self._calc_macd(close))
            indicators.update(self._calc_kdj(high, low, close))
            indicators.update(self._calc_bollinger(close))
            indicators.update(self._calc_volume_metrics(volume, latest_price))
            indicators.update(self._calc_price_position(latest_price, close))

            # 综合技术评分
            technical_score = self._calc_technical_score(indicators, latest_price)

            # 计算买卖价格
            buy_price, stop_loss = self._calc_trade_prices(
                indicators, latest_price
            )

            # 补充所有必需字段（AI分析需要）
            stock.update({
                # 均线
                'ma5': indicators.get('ma5', latest_price),
                'ma10': indicators.get('ma10', latest_price),
                'ma20': indicators.get('ma20', latest_price),
                'ma60': indicators.get('ma60', latest_price),

                # 技术指标
                'rsi': indicators.get('rsi', 50),
                'macd': indicators.get('macd', 0),
                'macd_signal': indicators.get('macd_signal', 0),
                'macd_hist': indicators.get('macd_hist', 0),
                'kdj_k': indicators.get('kdj_k', 50),
                'kdj_d': indicators.get('kdj_d', 50),
                'kdj_j': indicators.get('kdj_j', 50),

                # 布林带
                'boll_upper': indicators.get('boll_upper', latest_price * 1.1),
                'boll_mid': indicators.get('boll_mid', latest_price),
                'boll_lower': indicators.get('boll_lower', latest_price * 0.9),

                # 成交量
                'volume': float(stock.get('volume', volume[-1])),
                'avg_volume': indicators.get('avg_volume', volume[-1]),
                'vol_ratio': indicators.get('vol_ratio', 1.0),

                # 价格位置
                'dist_from_ma5': indicators.get('dist_ma5', 0),
                'dist_from_ma20': indicators.get('dist_ma20', 0),
                'is_bullish': indicators.get('is_bullish', False),

                # 评分
                'technical_score': technical_score,

                # 交易参考（会被AI覆盖，这里作为默认值）
                'ideal_buy_price': buy_price,
                'stop_loss_price': stop_loss,
                'target_price': round(latest_price * 1.1, 2),
            })

            return stock

        except Exception as e:
            self._err_count += 1
            if self._err_printed < 5:
                self._err_printed += 1
                logger.warning(f"计算失败 {code}: {e}")
            return None

    # ============================================================
    # 私有方法：数据获取
    # ============================================================

    def _fetch_kline(self, code: str) -> Optional[pd.DataFrame]:
        """
        获取历史K线数据
        
        Args:
            code: 6位股票代码
            
        Returns:
            K线DataFrame，失败返回None
        """
        try:
            import akshare as ak

            end = datetime.now().strftime('%Y%m%d')
            start = (datetime.now() - timedelta(days=180)).strftime('%Y%m%d')

            for attempt in range(3):
                try:
                    df = ak.stock_zh_a_hist(
                        symbol=code,
                        period="daily",
                        start_date=start,
                        end_date=end,
                        adjust="qfq"
                    )

                    if df is not None and not df.empty and len(df) >= 20:
                        return df

                except Exception as e:
                    if attempt < 2:
                        time.sleep(1 * (attempt + 1))
                    else:
                        logger.debug(f"{code} 获取K线失败: {e}")

            return None

        except ImportError:
            logger.error("未安装akshare")
            return None

    # ============================================================
    # 私有方法：指标计算
    # ============================================================

    def _calc_ma(self, close: np.ndarray) -> Dict:
        """计算移动平均线"""
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

        # 均线排列
        ma5 = result.get('ma5', close[-1])
        ma10 = result.get('ma10', close[-1])
        ma20 = result.get('ma20', close[-1])
        result['is_bullish'] = ma5 > ma10 > ma20

        return result

    def _calc_rsi(self, close: np.ndarray, period: int = 14) -> Dict:
        """计算RSI指标"""
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
        """计算MACD指标"""
        result = {'macd': 0, 'macd_signal': 0, 'macd_hist': 0}
        n = len(close)

        if n < 35:
            return result

        # 计算EMA
        ema12 = self._ema(close, 12)
        ema26 = self._ema(close, 26)

        dif = ema12 - ema26
        dea = self._ema(pd.Series(dif).values, 9)
        hist = 2 * (dif - dea)

        result['macd'] = round(dif[-1], 4)
        result['macd_signal'] = round(dea[-1], 4)
        result['macd_hist'] = round(hist[-1], 4)

        return result

    def _calc_kdj(self, high: np.ndarray, low: np.ndarray, close: np.ndarray,
                  n: int = 9) -> Dict:
        """计算KDJ指标"""
        result = {'kdj_k': 50, 'kdj_d': 50, 'kdj_j': 50}
        length = len(close)

        if length < n:
            return result

        # 取最近n天
        recent_close = close[-n:]
        recent_high = high[-n:]
        recent_low = low[-n:]

        highest = np.max(recent_high)
        lowest = np.min(recent_low)

        if highest == lowest:
            return result

        rsv = (recent_close[-1] - lowest) / (highest - lowest) * 100

        # 简化计算：使用默认前值50
        k = 2/3 * 50 + 1/3 * rsv
        d = 2/3 * 50 + 1/3 * k
        j = 3 * k - 2 * d

        result['kdj_k'] = round(k, 1)
        result['kdj_d'] = round(d, 1)
        result['kdj_j'] = round(j, 1)

        return result

    def _calc_bollinger(self, close: np.ndarray, period: int = 20) -> Dict:
        """计算布林带"""
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

    def _calc_volume_metrics(self, volume: np.ndarray,
                             latest_price: float) -> Dict:
        """计算成交量相关指标"""
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

    def _calc_price_position(self, latest_price: float,
                             close: np.ndarray) -> Dict:
        """计算价格相对位置"""
        result = {}

        n = len(close)
        if n >= 60:
            high_60 = np.max(close[-60:])
            low_60 = np.min(close[-60:])
            if high_60 > low_60:
                result['position_60d'] = round(
                    (latest_price - low_60) / (high_60 - low_60) * 100, 1
                )
            else:
                result['position_60d'] = 50

        return result

    # ============================================================
    # 私有方法：评分与价格
    # ============================================================

    def _calc_technical_score(self, indicators: Dict,
                              latest_price: float) -> int:
        """
        综合技术面评分 (0-100)
        
        评分维度：
        - 均线形态 (0-30分)
        - RSI指标 (0-20分)
        - MACD指标 (0-20分)
        - 成交量 (0-15分)
        - 价格位置 (0-15分)
        """
        score = 0

        # 1. 均线形态 (30分)
        ma5 = indicators.get('ma5', latest_price)
        ma10 = indicators.get('ma10', latest_price)
        ma20 = indicators.get('ma20', latest_price)
        ma60 = indicators.get('ma60', latest_price)

        if ma5 > ma10 > ma20:
            score += 20  # 多头排列
            if ma20 > ma60:
                score += 10  # 中长期也多头
        elif ma5 > ma20 and ma10 > ma20:
            score += 15  # 短期偏多
        elif ma5 < ma10 < ma20:
            score += 0   # 空头排列
        else:
            score += 8   # 震荡

        # 2. RSI (20分)
        rsi = indicators.get('rsi', 50)
        if 30 <= rsi <= 70:
            score += 20  # 健康区间
        elif 40 <= rsi <= 60:
            score += 15  # 中性偏强
        elif rsi > 70:
            score += 8   # 超买
        elif rsi < 30:
            score += 5   # 超卖
        else:
            score += 10

        # 3. MACD (20分)
        macd = indicators.get('macd', 0)
        macd_signal = indicators.get('macd_signal', 0)
        macd_hist = indicators.get('macd_hist', 0)

        if macd > macd_signal and macd > 0:
            score += 20  # 金叉且在零轴上方
        elif macd > macd_signal:
            score += 15  # 金叉
        elif macd > 0:
            score += 10  # 零轴上方
        else:
            score += 5   # 弱势

        # 4. 成交量 (15分)
        vol_ratio = indicators.get('vol_ratio', 1)
        if 0.8 <= vol_ratio <= 2.0:
            score += 15  # 量能正常
        elif 2.0 < vol_ratio <= 3.0:
            score += 10  # 温和放量
        elif vol_ratio > 3.0:
            score += 5   # 异常放量
        else:
            score += 8   # 缩量

        # 5. 价格位置 (15分)
        dist_ma20 = indicators.get('dist_ma20', 0)
        if 0 <= dist_ma20 <= 5:
            score += 15  # 贴近MA20，安全
        elif 5 < dist_ma20 <= 10:
            score += 10  # 略高但可接受
        elif dist_ma20 < 0:
            score += 8   # 低于MA20
        else:
            score += 5   # 乖离过大

        return min(100, score)

    def _calc_trade_prices(self, indicators: Dict,
                           latest_price: float) -> tuple:
        """
        计算理想买入价和止损价
        
        Returns:
            (买入价, 止损价)
        """
        ma20 = indicators.get('ma20', latest_price)
        ma60 = indicators.get('ma60', latest_price)
        boll_lower = indicators.get('boll_lower', latest_price * 0.95)

        # 理想买入价：MA20和布林下轨之间
        buy_price = round((ma20 + boll_lower) / 2, 2)
        if buy_price > latest_price:
            buy_price = round(latest_price * 0.98, 2)

        # 止损价：MA60下方3%或布林下轨
        stop_loss = round(min(ma60 * 0.97, boll_lower * 0.98), 2)
        if stop_loss >= latest_price * 0.95:
            stop_loss = round(latest_price * 0.93, 2)

        return buy_price, stop_loss

    # ============================================================
    # 工具方法
    # ============================================================

    @staticmethod
    def _ema(data: np.ndarray, period: int) -> np.ndarray:
        """计算指数移动平均"""
        if len(data) < period:
            return np.full_like(data, np.mean(data))

        result = np.zeros_like(data)
        result[0] = data[0]

        multiplier = 2 / (period + 1)
        for i in range(1, len(data)):
            result[i] = (data[i] - result[i-1]) * multiplier + result[i-1]

        return result
