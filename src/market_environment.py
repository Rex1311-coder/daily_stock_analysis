"""
大盘环境评估模块
判断当前市场整体状况，用于调整个股评分和仓位建议
"""
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class MarketEnvironment:
    """
    大盘环境评估器
    
    评估维度：
    1. 指数涨跌幅
    2. 指数趋势（相对MA20）
    3. 市场宽度（涨跌家数比）
    4. 市场情绪（涨停/跌停比）
    """

    INDEX_MAP = {
        'sh000001': '上证指数',
        'sz399001': '深证成指',
        'sz399006': '创业板指',
    }

    def __init__(self):
        self.assessment = {}

    def assess(self) -> Dict:
        """综合评估大盘环境"""
        logger.info("=" * 40)
        logger.info("📊 大盘环境评估")
        logger.info("=" * 40)

        index_score, index_detail = self._assess_indexes()
        breadth_score, breadth_detail = self._assess_breadth()
        sentiment_score, sentiment_detail = self._assess_sentiment()

        total_score = round(
            index_score * 0.5 + breadth_score * 0.3 + sentiment_score * 0.2, 1
        )

        level, suggestion, position_advice, adjust_factor = self._map_score(total_score)

        self.assessment = {
            'score': total_score,
            'level': level,
            'suggestion': suggestion,
            'position_advice': position_advice,
            'adjust_factor': adjust_factor,
            'index_detail': index_detail,
            'market_breadth': breadth_detail,
            'sentiment': sentiment_detail,
        }

        self._log_assessment()
        return self.assessment

    def get_score_adjustment(self) -> float:
        """获取评分调整系数"""
        if not self.assessment:
            self.assess()
        return self.assessment.get('adjust_factor', 0)

    def get_position_advice(self) -> float:
        """获取建议仓位比例"""
        if not self.assessment:
            self.assess()
        return self.assessment.get('position_advice', 0.5)

    def is_tradable(self) -> bool:
        """是否适合交易"""
        if not self.assessment:
            self.assess()
        return self.assessment.get('score', 50) >= 30

    def _assess_indexes(self) -> Tuple[float, Dict]:
        """评估主要指数"""
        detail = {}
        scores = []

        for code, name in self.INDEX_MAP.items():
            try:
                change_pct, ma20_bias = self._get_index_data(code)
                if change_pct is None:
                    continue

                idx_score = 50
                if change_pct > 1.5:
                    idx_score += 15
                elif change_pct > 0.5:
                    idx_score += 8
                elif change_pct > 0:
                    idx_score += 3
                elif change_pct > -1:
                    idx_score -= 5
                elif change_pct > -2:
                    idx_score -= 12
                else:
                    idx_score -= 20

                if ma20_bias is not None:
                    if ma20_bias > 2:
                        idx_score += 10
                    elif ma20_bias > 0:
                        idx_score += 5
                    elif ma20_bias > -2:
                        idx_score -= 5
                    else:
                        idx_score -= 10

                idx_score = max(0, min(100, idx_score))
                scores.append(idx_score)
                detail[name] = {
                    'change_pct': change_pct,
                    'ma20_bias': ma20_bias,
                    'score': idx_score,
                }
            except Exception as e:
                logger.debug(f"评估{name}失败: {e}")

        avg_score = round(sum(scores) / len(scores), 1) if scores else 50
        return avg_score, detail

    def _get_index_data(self, code: str) -> Tuple[Optional[float], Optional[float]]:
        """获取指数数据"""
        try:
            import akshare as ak
            import pandas as pd

            df = ak.stock_zh_index_daily(symbol=code)
            if df is None or df.empty:
                return None, None

            latest = df.iloc[-1]
            change_pct = float(latest.get('涨跌幅', 0))

            if len(df) >= 20:
                ma20 = df['收盘'].rolling(20).mean().iloc[-1]
                close = float(latest['收盘'])
                ma20_bias = round((close - ma20) / ma20 * 100, 1)
            else:
                ma20_bias = None

            return change_pct, ma20_bias
        except Exception as e:
            logger.debug(f"获取指数{code}失败: {e}")
            return None, None

    def _assess_breadth(self) -> Tuple[float, Dict]:
        """评估市场宽度"""
        try:
            import akshare as ak
            import pandas as pd

            df = ak.stock_zh_a_spot()
            if df is None or df.empty:
                return 50, {'status': '无法获取'}

            change_pct = pd.to_numeric(df['涨跌幅'], errors='coerce')
            up_count = (change_pct > 0).sum()
            down_count = (change_pct < 0).sum()
            total = len(df)
            up_ratio = up_count / total * 100 if total > 0 else 50

            if up_ratio > 70:
                score, status = 90, '普涨'
            elif up_ratio > 55:
                score, status = 70, '多数上涨'
            elif up_ratio > 45:
                score, status = 50, '涨跌互现'
            elif up_ratio > 30:
                score, status = 30, '多数下跌'
            else:
                score, status = 10, '普跌'

            return score, {
                'status': status,
                'up': int(up_count),
                'down': int(down_count),
                'up_ratio': round(up_ratio, 1),
            }
        except Exception as e:
            logger.debug(f"市场宽度评估失败: {e}")
            return 50, {'status': '评估失败'}

    def _assess_sentiment(self) -> Tuple[float, Dict]:
        """评估市场情绪"""
        try:
            import akshare as ak
            import pandas as pd

            df = ak.stock_zh_a_spot()
            if df is None or df.empty:
                return 50, {'status': '无法获取'}

            change_pct = pd.to_numeric(df['涨跌幅'], errors='coerce')
            limit_up = (change_pct >= 9.8).sum()
            limit_down = (change_pct <= -9.8).sum()

            if limit_up > 100 and limit_down < 10:
                score, sentiment = 90, '极度乐观'
            elif limit_up > 50 and limit_down < 20:
                score, sentiment = 70, '偏乐观'
            elif limit_up < 30 and limit_down < 30:
                score, sentiment = 50, '中性'
            elif limit_down > 50:
                score, sentiment = 20, '恐慌'
            elif limit_down > 100:
                score, sentiment = 5, '极度恐慌'
            else:
                score, sentiment = 40, '偏谨慎'

            return score, {
                'sentiment': sentiment,
                'limit_up': int(limit_up),
                'limit_down': int(limit_down),
            }
        except Exception as e:
            logger.debug(f"市场情绪评估失败: {e}")
            return 50, {'status': '评估失败'}

    def _map_score(self, score: float) -> Tuple[str, str, float, float]:
        """评分映射"""
        if score >= 80:
            return '强势', '积极做多', 1.0, 8
        elif score >= 65:
            return '偏强', '正常操作', 0.8, 3
        elif score >= 50:
            return '中性', '谨慎操作', 0.6, 0
        elif score >= 35:
            return '偏弱', '减仓为主', 0.3, -8
        else:
            return '弱势', '空仓等待', 0.1, -15

    def _log_assessment(self):
        """打印评估结果"""
        a = self.assessment
        logger.info(f"大盘: {a['score']}分 | {a['level']} | {a['suggestion']}")
        logger.info(f"仓位建议: {a['position_advice']*100:.0f}% | 评分调整: {a['adjust_factor']:+.0f}分")

        for name, d in a.get('index_detail', {}).items():
            logger.info(f"  {name}: {d['change_pct']:+.2f}% | MA20: {d.get('ma20_bias', 'N/A')}% | {d['score']}分")

        b = a.get('market_breadth', {})
        logger.info(f"宽度: {b.get('status','')} | 涨{int(b.get('up',0))} 跌{int(b.get('down',0))}")

        s = a.get('sentiment', {})
        logger.info(f"情绪: {s.get('sentiment','')} | 涨停{int(s.get('limit_up',0))} 跌停{int(s.get('limit_down',0))}")


def assess_market() -> Dict:
    """快速评估大盘"""
    env = MarketEnvironment()
    return env.assess()
