"""AI深度分析排名器 - 增强版"""
import json
import re
import time
from typing import List, Dict, Optional
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError

logger = logging.getLogger(__name__)


class AIRanker:
    """
    AI深度分析排名器
    
    功能：
    1. 批量调用AI模型分析股票
    2. 多维度综合评分（技术面 + AI判断 + 置信度）
    3. 智能推荐排序（趋势/风险调整）
    4. 降级方案（AI不可用时使用技术指标兜底）
    """

    def __init__(self, analyzer=None, max_workers: int = 5):
        """
        初始化AI排名器
        
        Args:
            analyzer: 分析器实例（预留，当前未使用）
            max_workers: 并发线程数
        """
        self.analyzer = analyzer
        self.max_workers = max_workers
        self.model = "deepseek/deepseek-chat"
        self.analysis_timeout = 90

    # ============================================================
    # 公共方法
    # ============================================================

    def batch_analyze(self, stocks: List[Dict]) -> List[Dict]:
        """
        批量AI分析股票
        
        Args:
            stocks: 包含技术指标的股票列表
            
        Returns:
            增加AI分析字段的股票列表
        """
        if not stocks:
            return []

        total = len(stocks)
        logger.info(f"AI批量分析: {total} 只")

        batch_size = 3
        results = []
        completed = 0
        failed = 0

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {}
            for i in range(0, total, batch_size):
                batch = stocks[i:i + batch_size]
                futures[executor.submit(self._analyze_batch, batch)] = i

            for future in as_completed(futures):
                batch_start = futures[future]
                batch_len = min(batch_size, total - batch_start)

                try:
                    r = future.result(timeout=self.analysis_timeout)
                    if r:
                        results.extend(r)
                        completed += len(r)
                    else:
                        failed += batch_len
                except TimeoutError:
                    logger.warning(f"批次超时 (起始 {batch_start})")
                    failed += batch_len
                except Exception as e:
                    logger.warning(f"批次异常 (起始 {batch_start}): {e}")
                    failed += batch_len

                processed = completed + failed
                if processed % 15 == 0 or processed >= total:
                    logger.info(f"AI进度: {processed}/{total} (成功{completed} 失败{failed})")

        logger.info(f"AI分析完成: 成功{completed}只 失败{failed}只")
        return results

    def get_top_recommendations(
        self,
        stocks: List[Dict],
        top_n: int = 20,
        min_score: int = 55
    ) -> List[Dict]:
        """
        获取Top推荐股票
        
        策略：
        1. 优先AI建议买入 + 评分达标
        2. 不足时补充高评分持有
        3. 排除明确卖出
        4. 评分不足时降低阈值兜底
        
        Args:
            stocks: 分析后的股票列表
            top_n: 返回数量
            min_score: 最低评分
            
        Returns:
            推荐股票列表
        """
        if not stocks:
            return []

        # 过滤有效结果，排除卖出
        valid = [
            s for s in stocks
            if s and 'final_score' in s and s.get('ai_action') != 'sell'
        ]

        if not valid:
            logger.warning("无有效推荐股票")
            return []

        # 分类
        buy_list = [
            s for s in valid
            if s.get('ai_action') == 'buy' and s.get('final_score', 0) >= min_score
        ]
        hold_list = [
            s for s in valid
            if s.get('ai_action') == 'hold' and s.get('final_score', 0) >= min_score
        ]

        # 按评分降序
        buy_list.sort(key=lambda x: x['final_score'], reverse=True)
        hold_list.sort(key=lambda x: x['final_score'], reverse=True)

        # 组合推荐
        recommendations = buy_list[:top_n]

        if len(recommendations) < top_n:
            remaining = [s for s in hold_list if s not in recommendations]
            recommendations.extend(remaining[:top_n - len(recommendations)])

        # 仍不足则降低阈值
        if len(recommendations) < top_n:
            logger.info(f"推荐不足{top_n}只，放宽评分阈值")
            extra = [
                s for s in valid
                if s not in recommendations and s.get('final_score', 0) >= 40
            ]
            extra.sort(key=lambda x: x['final_score'], reverse=True)
            recommendations.extend(extra[:top_n - len(recommendations)])

        # 简单去重（避免同行业过度集中）
        seen_prefix = set()
        diverse = []
        for s in recommendations:
            prefix = s.get('name', '')[:2]
            if prefix not in seen_prefix or len(diverse) < 5:
                diverse.append(s)
                seen_prefix.add(prefix)

        final = diverse[:top_n]

        buy_count = sum(1 for s in final if s.get('ai_action') == 'buy')
        logger.info(f"Top{top_n}推荐: 买入{buy_count}只 持有{len(final)-buy_count}只")
        return final

    # ============================================================
    # 私有方法：分析流程
    # ============================================================

    def _analyze_batch(self, stocks: List[Dict]) -> List[Dict]:
        """
        分析一批股票
        
        Args:
            stocks: 3只股票
            
        Returns:
            分析后的股票列表
        """
        if not stocks:
            return []

        try:
            context = self._build_context(stocks)
            prompt = self._build_prompt(context)
            resp_text = self._call_model(prompt)

            if resp_text:
                return self._parse_response(resp_text, stocks)
            else:
                logger.debug("AI返回空，使用降级方案")
                return self._fallback_score(stocks)

        except Exception as e:
            logger.warning(f"批次分析失败: {e}")
            return self._fallback_score(stocks)

    def _build_context(self, stocks: List[Dict]) -> List[str]:
        """构建股票分析上下文"""
        details = []

        for i, s in enumerate(stocks):
            price = s.get('price', 0)
            change_pct = s.get('change_pct', 0)
            turnover = s.get('turnover', 0)
            volume = s.get('volume', 0)

            ma5 = s.get('ma5', 0)
            ma10 = s.get('ma10', 0)
            ma20 = s.get('ma20', 0)
            ma60 = s.get('ma60', 0)

            rsi = s.get('rsi', 50)
            macd = s.get('macd', 0)
            macd_signal = s.get('macd_signal', 0)
            kdj_k = s.get('kdj_k', 50)
            kdj_d = s.get('kdj_d', 50)

            # 乖离率
            bias_ma5 = ((price - ma5) / ma5 * 100) if ma5 > 0 else 0
            bias_ma20 = ((price - ma20) / ma20 * 100) if ma20 > 0 else 0

            # 均线形态
            if ma5 > ma10 > ma20:
                ma_pattern = "多头排列↑"
            elif ma5 < ma10 < ma20:
                ma_pattern = "空头排列↓"
            else:
                ma_pattern = "均线缠绕→"

            # MACD状态
            if macd > macd_signal:
                macd_status = "金叉(多头)"
            elif macd < macd_signal:
                macd_status = "死叉(空头)"
            else:
                macd_status = "粘合"

            # RSI状态
            if rsi > 80:
                rsi_status = "超买"
            elif rsi < 20:
                rsi_status = "超卖"
            elif rsi > 50:
                rsi_status = "偏强"
            else:
                rsi_status = "偏弱"

            # 量能判断
            avg_vol = s.get('avg_volume', 0)
            if avg_vol > 0 and volume > 0:
                vol_ratio = volume / avg_vol
                if vol_ratio > 2:
                    vol_status = f"放量({vol_ratio:.1f}倍)"
                elif vol_ratio < 0.5:
                    vol_status = f"缩量({vol_ratio:.1f}倍)"
                else:
                    vol_status = "正常"
            else:
                vol_status = "未知"

            detail = (
                f"【股票{i+1}】{s['name']}({s['code']})\n"
                f"  价格: {price:.2f} | 涨跌: {change_pct:+.2f}% | 换手: {turnover:.1f}%\n"
                f"  均线: MA5={ma5:.2f} MA10={ma10:.2f} MA20={ma20:.2f} MA60={ma60:.2f}\n"
                f"  形态: {ma_pattern} | 乖离: MA5={bias_ma5:+.1f}% MA20={bias_ma20:+.1f}%\n"
                f"  RSI: {rsi:.0f}({rsi_status}) | MACD: {macd:.3f}({macd_status})\n"
                f"  KDJ: K={kdj_k:.0f} D={kdj_d:.0f} | 量能: {vol_status}"
            )

            details.append(detail)

        return details

    def _build_prompt(self, details: List[str]) -> str:
        """构建AI分析提示词"""
        context = "\n\n".join(details)

        prompt = f"""你是A股短线交易分析专家。请对以下{len(details)}只股票逐一分析。

{context}

## 分析框架

### 1. 趋势判断
- 多头排列(MA5>MA10>MA20)：看涨趋势
- 空头排列(MA5<MA10<MA20)：看跌趋势
- 均线缠绕：震荡趋势

### 2. 买卖信号
- **强烈买入**：多头排列 + MACD金叉 + RSI 40-60 + 放量突破
- **可以买入**：多头趋势 + 回调至MA10获支撑 + RSI不超买
- **观望持有**：趋势不明朗 + 指标中性
- **建议减仓**：死叉出现 + RSI超买 + 乖离率过大(>8%)
- **建议卖出**：空头排列 + MACD死叉 + 破位下跌

### 3. 评分标准 (0-100)
- 80-100：多头强势 + 指标共振看多 + 量价配合好
- 65-79：趋势偏多 + 有买入信号 + 风险可控
- 50-64：方向不明 + 观望为主
- 35-49：趋势偏空 + 建议减仓
- 0-34：空头明确 + 建议离场

### 4. 风险等级
- **低**：多头排列 + RSI 30-70 + 换手率1-5% + 乖离率<5%
- **中**：均线缠绕 或 RSI偏高/低 或 换手异常
- **高**：空头排列 或 RSI>80/<20 或 乖离率>8%

## 返回格式

严格返回JSON数组，不要Markdown代码块：

```json
[
  {{
    "id": 1,
    "trend": "多头",
    "action": "buy",
    "score": 78,
    "confidence": 85,
    "risk": "低",
    "reason": "多头排列+MACD金叉，回踩MA10支撑，可介入",
    "buy_price": 15.50,
    "stop_loss": 14.80,
    "target": 17.00
  }}
]
