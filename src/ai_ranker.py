"""AI深度分析排名器"""
import json
import re
from typing import List, Dict, Optional
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

logger = logging.getLogger(__name__)


class AIRanker:

    def __init__(self, analyzer, max_workers: int = 3):
        self.analyzer = analyzer
        self.max_workers = max_workers

    def batch_analyze(self, stocks: List[Dict]) -> List[Dict]:
        if not stocks:
            return []
        logger.info(f"AI深度分析: {len(stocks)} 只")
        results = []
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {executor.submit(self._analyze, s): s for s in stocks}
            for i, f in enumerate(as_completed(futures)):
                try:
                    r = f.result(timeout=45)
                    if r:
                        results.append(r)
                except:
                    pass
                if (i + 1) % 20 == 0:
                    logger.info(f"AI进度: {i+1}/{len(stocks)}")
        logger.info(f"AI分析完成: {len(results)} 只")
        return results

    def _analyze(self, stock: Dict) -> Optional[Dict]:
        try:
            prompt = f"""分析此A股，返回严格JSON（不要Markdown）：
{stock['name']}({stock['code']})
现价:{stock['price']} 涨跌:{stock.get('change_pct',0)}% 换手:{stock.get('turnover',0)}%
MA5/20/60:{stock.get('ma5')}/{stock.get('ma20')}/{stock.get('ma60')} RSI:{stock.get('rsi')}
返回:{{"action":"buy/hold/sell","score":0-100,"confidence":0-100,"reason":"15字内","risk":"低/中/高"}}"""

            import litellm
            resp = litellm.completion(
                model="deepseek/deepseek-chat",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=200,
            )
            resp_text = resp.choices[0].message.content if resp else ""

            if not resp_text:
                return None

            resp_text = resp_text.strip()
            if resp_text.startswith('```'):
                resp_text = resp_text.split('\n', 1)[1] if '\n' in resp_text else resp_text[3:]
            if resp_text.endswith('```'):
                resp_text = resp_text[:-3]

            m = re.search(r'\{.*\}', resp_text, re.DOTALL)
            if not m:
                return None

            ai = json.loads(m.group())
            ts = stock.get('technical_score', 50)
            fs = round(ts * 0.4 + ai.get('score', 50) * 0.6, 1)

            stock.update({
                'ai_action': ai.get('action', 'hold'),
                'ai_score': ai.get('score', 50),
                'ai_confidence': ai.get('confidence', 50),
                'ai_reason': ai.get('reason', ''),
                'risk_level': ai.get('risk', '中'),
                'final_score': fs,
            })
            return stock
        except Exception as e:
            logger.debug(f"AI分析失败: {e}")
            return None

    def get_top_recommendations(self, stocks, top_n=20):
        valid = [s for s in stocks if s and 'final_score' in s]
        valid.sort(key=lambda x: x['final_score'], reverse=True)

        buy = [s for s in valid if s.get('ai_action') == 'buy' and s.get('final_score', 0) > 55]
        if len(buy) < top_n:
            remain = [s for s in valid if s not in buy]
            remain.sort(key=lambda x: x['final_score'], reverse=True)
            buy.extend(remain[:top_n - len(buy)])

        return buy[:top_n]
