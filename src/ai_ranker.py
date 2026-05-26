"""AI深度分析排名器 - 增强版（含大盘环境 + 量价配合 + 多档止盈）"""
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
    2. 大盘环境评估 + 量价配合确认
    3. 多维度综合评分（技术面 + AI判断 + 置信度 + 上涨空间 + 大盘调整）
    4. 三档目标价（保守/中性/激进）+ 仓位分配
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
        self.market_adjust = 0
        self.position_advice = 0.5
        self.market_assessment = {}

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

        # ========== 大盘环境评估 ==========
        self._assess_market()

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

    def _assess_market(self):
        """评估大盘环境"""
        try:
            from src.market_environment import MarketEnvironment
            env = MarketEnvironment()
            market = env.assess()
            self.market_adjust = market.get('adjust_factor', 0)
            self.position_advice = market.get('position_advice', 0.5)
            self.market_assessment = market

            if not env.is_tradable():
                logger.warning("⚠️ 大盘环境恶劣，建议空仓或轻仓")
        except ImportError:
            logger.warning("market_environment模块未就绪")
            self.market_adjust = 0
            self.position_advice = 0.5
        except Exception as e:
            logger.warning(f"大盘评估失败: {e}")
            self.market_adjust = 0
            self.position_advice = 0.5

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
        2. 按评分和上涨空间综合排序
        3. 不足时补充高评分持有
        4. 排除明确卖出
        5. 简单去重 + 仓位分配
        
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

        # ========== 仓位分配 ==========
        if final:
            scores = [s.get('final_score', 50) for s in final]
            total_score = sum(scores)
            for s in final:
                s['position_pct'] = round(
                    (s.get('final_score', 50) / total_score) * self.position_advice * 100, 1
                ) if total_score > 0 else round(self.position_advice / len(final) * 100, 1)

        buy_count = sum(1 for s in final if s.get('ai_action') == 'buy')
        avg_upside = sum(s.get('upside_pct', 0) for s in final) / max(len(final), 1)
        logger.info(
            f"Top{top_n}推荐: 买入{buy_count}只 持有{len(final)-buy_count}只 "
            f"平均空间{avg_upside:.1f}% | 建议总仓位{self.position_advice*100:.0f}%"
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
        """构建详细的股票分析上下文（含量价配合状态）"""
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
            if macd > 0 and macd > macd_signal and macd_hist > 0:
                macd_status = "零轴上金叉+红柱(强势)"
            elif macd > 0 and macd > macd_signal:
                macd_status = "零轴上金叉(转多)"
            elif macd > macd_signal:
                macd_status = "金叉(偏多)"
            elif macd > 0 and macd < macd_signal:
                macd_status = "零轴上死叉(回调)"
            elif macd < macd_signal:
                macd_status = "死叉(偏空)"
            else:
                macd_status = "粘合"

            # MACD柱状态
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

            # 量价配合判断（增强版）
            avg_vol = s.get('avg_volume', 0)
            vol_ratio = s.get('vol_ratio', 1)

            if avg_vol > 0 and volume > 0:
                if change_pct > 1 and vol_ratio > 1.5:
                    vol_status = f"🔺放量上涨({vol_ratio:.1f}倍) ✅"
                elif change_pct > 0.5 and vol_ratio > 1.2:
                    vol_status = f"🔺温和放量上涨({vol_ratio:.1f}倍)"
                elif change_pct > 1 and vol_ratio < 0.8:
                    vol_status = f"⚠缩量上涨({vol_ratio:.1f}倍) 警惕诱多"
                elif change_pct < -1 and vol_ratio > 1.5:
                    vol_status = f"🔻放量下跌({vol_ratio:.1f}倍) ⚠危险"
                elif change_pct < 0 and vol_ratio < 0.8:
                    vol_status = f"缩量下跌({vol_ratio:.1f}倍) 正常调整"
                elif abs(change_pct) < 0.5 and vol_ratio > 2:
                    vol_status = f"⚠放量滞涨({vol_ratio:.1f}倍) 出货信号"
                elif abs(change_pct) < 0.5 and vol_ratio < 0.5:
                    vol_status = f"缩量横盘({vol_ratio:.1f}倍)"
                else:
                    vol_status = f"正常({vol_ratio:.1f}倍)"
            else:
                vol_status = "未知"

            # 量价状态（来自技术筛选）
            vp_status = s.get('vol_price_status', '')

            detail = (
                f"【股票{i+1}】{s['name']}({s['code']})\n"
                f"  现价: {price:.2f} | 涨跌: {change_pct:+.2f}% | 换手: {turnover:.1f}%\n"
                f"  均线: MA5={ma5:.2f} MA10={ma10:.2f} MA20={ma20:.2f} MA60={ma60:.2f}\n"
                f"  形态: {ma_pattern} | 乖离: MA5={bias_ma5:+.1f}% MA20={bias_ma20:+.1f}%\n"
                f"  RSI: {rsi:.0f}({rsi_status}) | MACD: {macd:.4f}({macd_status}) | {hist_status}\n"
                f"  KDJ: K={kdj_k:.0f} D={kdj_d:.0f} | 量价: {vol_status}\n"
                f"  量价状态: {vp_status}\n"
                f"  布林: 上轨{boll_upper:.2f} 中轨{boll_mid:.2f} 下轨{boll_lower:.2f}\n"
                f"  空间(至布林上轨): {upside_to_boll:.1f}%"
            )

            details.append(detail)

        return details

    def _build_prompt(self, details: List[str]) -> str:
        """构建AI分析提示词（含大盘环境 + 量价配合 + 多档止盈）"""
        context = "\n\n".join(details)
        n = len(details)

        # 大盘环境提示
        market_hint = ""
        if self.market_assessment:
            m = self.market_assessment
            market_hint = (
                f"\n## 当前大盘环境\n"
                f"- 状态: {m.get('level', '未知')} | 评分: {m.get('score', 50)}分\n"
                f"- 建议: {m.get('suggestion', '正常操作')} | 仓位: {m.get('position_advice', 0.5)*100:.0f}%\n"
                f"- 评分调整: {m.get('adjust_factor', 0):+.0f}分\n"
            )

        prompt = f"""你是A股短线交易专家。请对以下{n}只股票逐一分析，重点关注量价配合、盈利空间和多档止盈。
{market_hint}
{context}

## 分析框架

### 1. 趋势与买卖信号
- **强烈买入**：完全多头排列 + MACD零轴上金叉 + RSI 40-60 + 放量上涨 ✅
- **可以买入**：多头趋势 + 回调至MA10/MA20获支撑 + RSI不超买 + 缩量企稳
- **观望持有**：趋势不明朗 + 指标中性 + 等待方向选择
- **建议减仓**：死叉出现 + RSI超买(>75) + 乖离率过大(>8%) + 放量滞涨
- **建议卖出**：空头排列 + MACD死叉 + 跌破MA20 + 放量下跌 ❌

### 2. 量价配合分析（非常重要！）
- **放量上涨**：主力资金进场，上涨可信度高，空间大 ✅
- **缩量上涨**：量价背离，可能是诱多，上涨空间有限 ⚠
- **放量下跌**：主力出货，坚决回避 ❌
- **缩量下跌**：正常调整洗盘，可能是买点
- **放量滞涨**：典型出货信号，坚决回避 ❌
- **缩量横盘**：交投清淡，等待方向选择

### 3. 盈利空间评估
- 压力位：布林上轨、前高、整数关口、MA60
- 布林带宽度（宽=波动大=空间大）
- 量价配合好 + 布林开口向上 = 空间大
- 缩量靠近压力位 = 突破概率低，空间有限

### 4. 三档止盈策略（必须给出三个目标价）
- **target1（保守止盈）**：第一压力位，达到后卖出30%，锁定基础利润
- **target2（中性止盈）**：突破第一压力后的第二目标，达到后再卖40%
- **target3（激进止盈）**：趋势加速后的终极目标，清仓剩余30%

### 5. 评分标准 (0-100)
- 80-100：多头强势 + 上涨空间>15% + 放量上涨 + 多指标共振
- 65-79：趋势偏多 + 上涨空间8-15% + 有买入信号 + 量价正常
- 50-64：方向不明或空间有限(3-8%)或量价背离
- 35-49：趋势偏空或空间<3%或放量滞涨/下跌
- 0-34：明确看跌或已到顶部

### 6. 风险等级
- **低**：完全多头排列 + RSI 30-70 + 上涨空间>10% + 放量上涨 + 换手正常
- **中**：均线缠绕 或 RSI偏高(70-80) 或 量价背离
- **高**：空头排列 或 RSI超买(>80)/超卖(<20) 或 乖离>8% 或 放量滞涨/下跌

## 返回格式（严格JSON数组，无其他内容）

[{{"id":1,"trend":"多头","action":"buy","score":78,"confidence":85,"risk":"低","reason":"完全多头+MACD零轴上金叉+放量上涨，布林开口向上，空间充足","buy_price":15.20,"stop_loss":14.50,"target1":16.50,"target2":17.80,"target3":19.50,"upside_pct":28.3}}]

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
        """将AI分析结果应用到股票数据（含大盘调整 + 量价调整）"""
        ai_score = ai.get('score', 50)
        ai_confidence = ai.get('confidence', 50)
        tech_score = stock.get('technical_score', 50)
        trend = ai.get('trend', '震荡')
        risk = ai.get('risk', '中')
        upside_pct = ai.get('upside_pct', 5)
        price = stock.get('price', 0)
        vol_price_status = stock.get('vol_price_status', '')

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

        # ========== 量价配合调整 ==========
        if '放量上涨' in vol_price_status:
            final_score = min(100, final_score + 3)
        elif '放量下跌' in vol_price_status or '放量滞涨' in vol_price_status:
            final_score = max(0, final_score - 8)
        elif '缩量上涨' in vol_price_status:
            final_score = max(0, final_score - 3)

        # ========== 上涨空间加分 ==========
        if upside_pct > 20:
            final_score = min(100, final_score + 8)
        elif upside_pct > 15:
            final_score = min(100, final_score + 5)
        elif upside_pct > 8:
            final_score = min(100, final_score + 3)
        elif upside_pct < 3:
            final_score = max(0, final_score - 5)

        # ========== 大盘环境调整 ==========
        final_score = max(0, min(100, final_score + self.market_adjust))

        # ========== 处理目标价 ==========
        target1 = ai.get('target1', round(price * 1.05, 2))
        target2 = ai.get('target2', round(price * 1.08, 2))
        target3 = ai.get('target3', round(price * 1.12, 2))

        # 确保 target1 < target2 < target3
        if target1 >= target2:
            target2 = round(target1 * 1.05, 2)
        if target2 >= target3:
            target3 = round(target2 * 1.05, 2)

        stop_loss = ai.get('stop_loss', round(price * 0.95, 2))

        # 盈亏比
        profit_loss_ratio = round(
            (target1 - price) / max(price - stop_loss, 0.01), 1
        ) if price > 0 else 0

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
            'stop_loss_price': stop_loss,

            # 三档止盈
            'target1': target1,
            'target2': target2,
            'target3': target3,

            # 盈亏比
            'profit_loss_ratio': profit_loss_ratio,

            # 大盘调整记录
            'market_adjust': self.market_adjust,
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
            upside_pct = round((boll_upper - price) / price * 100, 1) if price > 0 else 5
            final_score = max(0, min(100, tech_score + self.market_adjust))

            s.update({
                'ai_trend': '震荡',
                'ai_action': 'hold',
                'ai_score': 50,
                'ai_confidence': 30,
                'ai_reason': 'AI暂不可用，参考技术指标',
                'risk_level': '中',
                'final_score': final_score,
                'upside_pct': upside_pct,
                'ideal_buy_price': price,
                'stop_loss_price': round(price * 0.95, 2),
                'target1': round(price * 1.05, 2),
                'target2': round(price * 1.08, 2),
                'target3': round(price * 1.12, 2),
                'profit_loss_ratio': 1.0,
                'market_adjust': self.market_adjust,
            })

        return stocks
