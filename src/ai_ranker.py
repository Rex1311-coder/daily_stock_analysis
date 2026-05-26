"""AI深度分析排名器 - 增强版（含多档止盈策略）"""
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
    2. 多维度综合评分（技术面 + AI判断 + 置信度 + 上涨空间）
    3. 三档目标价（保守/中性/激进）
    4. 智能推荐排序
    5. 降级方案（AI不可用时使用技术指标兜底）
    """

    def __init__(self, analyzer=None, max_workers: int = 5):
        """
        初始化AI排名器
        
        Args:
            analyzer: 分析器实例（预留）
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
                    logger.info(
                        f"AI进度: {processed}/{total} (成功{completed} 失败{failed})"
                    )

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
        2. 按上涨空间排序（同分时空间大的优先）
        3. 不足时补充高评分持有
        4. 排除明确卖出
        
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

        # 排序：先按评分，同分按上涨空间
        buy_list.sort(
            key=lambda x: (x['final_score'], x.get('upside_pct', 0)),
            reverse=True
        )
        hold_list.sort(
            key=lambda x: (x['final_score'], x.get('upside_pct', 0)),
            reverse=True
        )

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
            extra.sort(
                key=lambda x: (x['final_score'], x.get('upside_pct', 0)),
                reverse=True
            )
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
        avg_upside = sum(s.get('upside_pct', 0) for s in final) / max(len(final), 1)
        logger.info(
            f"Top{top_n}推荐: 买入{buy_count}只 持有{len(final)-buy_count}只 "
            f"平均上涨空间{avg_upside:.1f}%"
        )
        return final

    # ============================================================
    # 私有方法：分析流程
    # ============================================================

    def _analyze_batch(self, stocks: List[Dict]) -> List[Dict]:
        """
        分析一批股票（3只）
        
        Args:
            stocks: 3只股票数据
            
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
        """构建详细的股票分析上下文"""
        details = []

        for i, s in enumerate(stocks):
            price = s.get('price', 0)
            change_pct = s.get('change_pct', 0)
            turnover = s.get('turnover', 0)
            volume = s.get('volume', 0)

            # 均线
            ma5 = s.get('ma5', 0)
            ma10 = s.get('ma10', 0)
            ma20 = s.get('ma20', 0)
            ma60 = s.get('ma60', 0)

            # 技术指标
            rsi = s.get('rsi', 50)
            macd = s.get('macd', 0)
            macd_signal = s.get('macd_signal', 0)
            macd_hist = s.get('macd_hist', 0)
            kdj_k = s.get('kdj_k', 50)
            kdj_d = s.get('kdj_d', 50)

            # 布林带
            boll_upper = s.get('boll_upper', price * 1.1)
            boll_mid = s.get('boll_mid', price)
            boll_lower = s.get('boll_lower', price * 0.9)

            # 乖离率
            bias_ma5 = ((price - ma5) / ma5 * 100) if ma5 > 0 else 0
            bias_ma20 = ((price - ma20) / ma20 * 100) if ma20 > 0 else 0

            # 上涨空间（到布林上轨）
            upside_to_boll = ((boll_upper - price) / price * 100) if price > 0 else 0

            # 均线形态
            if ma5 > ma10 > ma20 > ma60:
                ma_pattern = "完全多头排列↑↑"
            elif ma5 > ma10 > ma20:
                ma_pattern = "多头排列↑"
            elif ma5 < ma10 < ma20:
                ma_pattern = "空头排列↓"
            else:
                ma_pattern = "均线缠绕→"

            # MACD状态
            if macd > 0 and macd > macd_signal:
                macd_status = "零轴上金叉(强势)"
            elif macd > macd_signal:
                macd_status = "金叉(转多)"
            elif macd > 0 and macd < macd_signal:
                macd_status = "零轴上死叉(回调)"
            elif macd < macd_signal:
                macd_status = "死叉(偏空)"
            else:
                macd_status = "粘合"

            # MACD柱变化
            if macd_hist > 0:
                hist_status = "红柱(多头动能)"
            elif macd_hist < 0:
                hist_status = "绿柱(空头动能)"
            else:
                hist_status = "零轴"

            # RSI状态
            if rsi > 80:
                rsi_status = "超买⚠"
            elif rsi > 70:
                rsi_status = "偏强↑"
            elif rsi > 50:
                rsi_status = "中性偏强"
            elif rsi > 30:
                rsi_status = "中性偏弱"
            elif rsi > 20:
                rsi_status = "偏弱↓"
            else:
                rsi_status = "超卖⚡"

            # 量能判断
            avg_vol = s.get('avg_volume', 0)
            if avg_vol > 0 and volume > 0:
                vol_ratio = volume / avg_vol
                if vol_ratio > 3:
                    vol_status = f"异常放量({vol_ratio:.1f}倍)⚠"
                elif vol_ratio > 2:
                    vol_status = f"明显放量({vol_ratio:.1f}倍)"
                elif vol_ratio > 1.2:
                    vol_status = f"温和放量({vol_ratio:.1f}倍)"
                elif vol_ratio > 0.8:
                    vol_status = "正常"
                elif vol_ratio > 0.5:
                    vol_status = f"缩量({vol_ratio:.1f}倍)"
                else:
                    vol_status = f"地量({vol_ratio:.1f}倍)"
            else:
                vol_status = "未知"

            detail = (
                f"【股票{i+1}】{s['name']}({s['code']})\n"
                f"  现价: {price:.2f} | 涨跌: {change_pct:+.2f}% | 换手: {turnover:.1f}%\n"
                f"  均线: MA5={ma5:.2f} MA10={ma10:.2f} MA20={ma20:.2f} MA60={ma60:.2f}\n"
                f"  形态: {ma_pattern} | 乖离: MA5={bias_ma5:+.1f}% MA20={bias_ma20:+.1f}%\n"
                f"  RSI: {rsi:.0f}({rsi_status}) | MACD: {macd:.4f}({macd_status}) | {hist_status}\n"
                f"  KDJ: K={kdj_k:.0f} D={kdj_d:.0f} | 量能: {vol_status}\n"
                f"  布林: 上轨{boll_upper:.2f} 中轨{boll_mid:.2f} 下轨{boll_lower:.2f}\n"
                f"  上涨空间(至布林上轨): {upside_to_boll:.1f}%"
            )

            details.append(detail)

        return details

    def _build_prompt(self, details: List[str]) -> str:
        """构建AI分析提示词（含多档止盈策略）"""
        context = "\n\n".join(details)
        n = len(details)

        prompt = f"""你是A股短线交易专家。请对以下{n}只股票逐一分析，**重点关注盈利空间和多档止盈策略**。

{context}

## 分析框架

### 1. 趋势与买卖信号
- **强烈买入**：完全多头排列 + MACD零轴上金叉 + RSI 40-60 + 放量突破
- **可以买入**：多头趋势 + 回调至MA10/MA20获支撑 + RSI不超买 + 缩量企稳
- **观望持有**：趋势不明朗 + 指标中性 + 等待方向选择
- **建议减仓**：死叉出现 + RSI超买(>75) + 乖离率过大(>8%) + 放量滞涨
- **建议卖出**：空头排列 + MACD死叉 + 跌破MA20 + 放量下跌

### 2. 盈利空间评估（核心）
根据以下因素判断上涨空间：
- **压力位分析**：布林上轨、前期高点、整数关口、MA60位置
- **布林带状态**：带宽大小（大=波动大=空间大）、价格在带中的位置
- **成交量配合**：放量突破压力位成功率高，缩量靠近压力位易回落
- **均线支撑**：下方均线密集区支撑力度
- **指标共振**：多指标同时看多时空间更大

### 3. 三档止盈策略（必须给出三个目标价）
- **target1（保守止盈）**：第一压力位（如布林上轨/前高/整数关），达到后卖出30%
- **target2（中性止盈）**：突破第一压力后的第二目标，达到后再卖40%
- **target3（激进止盈）**：趋势加速后的终极目标，清仓剩余30%

### 4. 评分标准 (0-100)
- 80-100：多头强势 + 上涨空间>15% + 多指标共振 + 放量配合
- 65-79：趋势偏多 + 上涨空间8-15% + 有买入信号
- 50-64：方向不明或空间有限(3-8%)
- 35-49：趋势偏空或空间<3%
- 0-34：明确看跌或已到顶部

### 5. 风险等级
- **低**：完全多头排列 + RSI 30-70 + 上涨空间>10% + 换手正常
- **中**：均线缠绕 或 RSI偏高(70-80) 或 换手略高
- **高**：空头排列 或 RSI超买(>80)/超卖(<20) 或 乖离率>8% 或 异常放量

## 返回格式（严格JSON数组，无其他内容）

[{{
  "id":1,
  "trend":"多头",
  "action":"buy",
  "score":78,
  "confidence":85,
  "risk":"低",
  "reason":"完全多头排列+MACD零轴上金叉+回踩MA10获支撑，布林开口向上，上方空间充足",
  "buy_price":15.20,
  "stop_loss":14.50,
  "target1":16.50,
  "target2":17.80,
  "target3":19.50,
  "upside_pct":28.3
}}]

字段说明：
- upside_pct: target3相对现价的预期最大涨幅百分比
- target1/2/3: 价格依次递增，分别对应保守/中性/激进
- 只返回JSON数组，不要Markdown代码块"""

        return prompt

    def _call_model(self, prompt: str, max_retries: int = 2) -> Optional[str]:
        """调用AI模型（带重试机制）"""
        import litellm

        for attempt in range(max_retries + 1):
            try:
                resp = litellm.completion(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=800,
                    temperature=0.3,
                    timeout=60,
                )

                if resp and resp.choices:
                    return resp.choices[0].message.content

            except Exception as e:
                if attempt < max_retries:
                    wait = 2 ** attempt
                    logger.debug(
                        f"AI调用重试({attempt+1}/{max_retries+1})，等待{wait}s: {e}"
                    )
                    time.sleep(wait)
                else:
                    logger.warning(f"AI调用失败: {e}")

        return None

    def _parse_response(self, resp_text: str, stocks: List[Dict]) -> List[Dict]:
        """解析AI返回的JSON结果"""
        # 清理文本
        resp_text = resp_text.strip()

        # 去掉Markdown代码块
        if resp_text.startswith('```'):
            lines = resp_text.split('\n')
            if len(lines) > 1:
                resp_text = '\n'.join(lines[1:])
            else:
                resp_text = resp_text[3:]
        if resp_text.endswith('```'):
            resp_text = resp_text[:-3]

        # 提取JSON数组
        match = re.search(r'\[.*\]', resp_text, re.DOTALL)
        if not match:
            logger.debug("未找到JSON数组，使用降级方案")
            return self._fallback_score(stocks)

        try:
            ai_results = json.loads(match.group())
        except json.JSONDecodeError as e:
            logger.warning(f"JSON解析失败: {e}")
            return self._fallback_score(stocks)

        # 映射结果到股票
        for item in ai_results:
            idx = item.get('id', 0) - 1
            if 0 <= idx < len(stocks):
                self._apply_ai_result(stocks[idx], item)

        return stocks

    def _apply_ai_result(self, stock: Dict, ai: Dict) -> None:
        """将AI分析结果应用到股票数据（含三档止盈）"""
        ai_score = ai.get('score', 50)
        ai_confidence = ai.get('confidence', 50)
        tech_score = stock.get('technical_score', 50)
        trend = ai.get('trend', '震荡')
        risk = ai.get('risk', '中')
        upside_pct = ai.get('upside_pct', 5)
        price = stock.get('price', 0)

        # ========== 综合评分 ==========
        # 技术30% + AI评分50% + 置信度20%
        final_score = round(
            tech_score * 0.3 + ai_score * 0.5 + ai_confidence * 0.2, 1
        )

        # ========== 趋势调整 ==========
        if trend == '多头':
            final_score = min(100, final_score + 5)
        elif trend == '空头':
            final_score = max(0, final_score - 10)

        # ========== 风险调整 ==========
        if risk == '高':
            final_score = max(0, final_score - 5)
        elif risk == '低':
            final_score = min(100, final_score + 3)

        # ========== 上涨空间加分 ==========
        if upside_pct > 20:
            final_score = min(100, final_score + 8)  # 大空间
        elif upside_pct > 15:
            final_score = min(100, final_score + 5)
        elif upside_pct > 8:
            final_score = min(100, final_score + 3)
        elif upside_pct < 3:
            final_score = max(0, final_score - 5)  # 空间太小

        # ========== 处理目标价 ==========
        target1 = ai.get('target1', round(price * 1.05, 2))
        target2 = ai.get('target2', round(price * 1.08, 2))
        target3 = ai.get('target3', round(price * 1.12, 2))

        # 确保 target1 < target2 < target3
        if target1 >= target2:
            target2 = round(target1 * 1.05, 2)
        if target2 >= target3:
            target3 = round(target2 * 1.05, 2)

        # 更新股票字段
        stock.update({
            # 基础分析
            'ai_trend': trend,
            'ai_action': ai.get('action', 'hold'),
            'ai_score': ai_score,
            'ai_confidence': ai_confidence,
            'ai_reason': ai.get('reason', ''),
            'risk_level': risk,
            'final_score': final_score,

            # 盈利空间
            'upside_pct': upside_pct,

            # 交易价格
            'ideal_buy_price': ai.get('buy_price', price),
            'stop_loss_price': ai.get('stop_loss', round(price * 0.95, 2)),

            # 三档止盈
            'target1': target1,
            'target2': target2,
            'target3': target3,

            # 盈亏比（target1相对止损）
            'profit_loss_ratio': round(
                (target1 - price) / max(price - ai.get('stop_loss', price * 0.95), 0.01), 1
            ) if price > 0 else 0,
        })

    def _fallback_score(self, stocks: List[Dict]) -> List[Dict]:
        """
        降级方案：纯技术指标评分
        
        当AI不可用时使用，确保系统仍能产出结果。
        """
        logger.info("使用技术指标降级评分")

        for s in stocks:
            tech_score = s.get('technical_score', 50)
            price = s.get('price', 0)
            boll_upper = s.get('boll_upper', price * 1.1)

            # 基于布林带估算上涨空间
            upside_pct = round((boll_upper - price) / price * 100, 1) if price > 0 else 5

            s.update({
                'ai_trend': '震荡',
                'ai_action': 'hold',
                'ai_score': 50,
                'ai_confidence': 30,
                'ai_reason': 'AI暂不可用，参考技术指标',
                'risk_level': '中',
                'final_score': tech_score,
                'upside_pct': upside_pct,
                'ideal_buy_price': price,
                'stop_loss_price': round(price * 0.95, 2),
                'target1': round(price * 1.05, 2),
                'target2': round(price * 1.08, 2),
                'target3': round(price * 1.12, 2),
                'profit_loss_ratio': 1.0,
            })

        return stocks
