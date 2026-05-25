"""AI深度分析排名器 - 批量版"""
import json
import re
from typing import List, Dict, Optional
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

logger = logging.getLogger(__name__)


class AIRanker:

    def __init__(self, analyzer, max_workers: int = 10):
        self.analyzer = analyzer
        self.max_workers = max_workers

    def batch_analyze(self, stocks: List[Dict]) -> List[Dict]:
        if not stocks:
            return []
        
        logger.info(f"AI批量分析: {len(stocks)} 只")
        
        # 每 5 只一组
        batch_size = 5
        results = []
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = []
            for i in range(0, len(stocks), batch_size):
                batch = stocks[i:i+batch_size]
                futures.append(executor.submit(self._analyze_batch, batch))
            
            for i, f in enumerate(as_completed(futures)):
                try:
                    r = f.result(timeout=60)
                    if r:
                        results.extend(r)
                except:
                    pass
                if (i + 1) % 5 == 0:
                    logger.info(f"AI进度: {min((i+1)*batch_size, len(stocks))}/{len(stocks)}")
        
        logger.info(f"AI完成: {len(results)} 只")
        return results

    def _analyze_batch(self, stocks: List[Dict]) -> List[Dict]:
        """批量分析 5 只股票"""
        try:
            # 构建批量 Prompt
            stock_list = []
            for i, s in enumerate(stocks):
                stock_list.append(
                    f"{i+1}. {s['name']}({s['code']}) 现价{s['price']} "
                    f"涨跌{s.get('change_pct',0)}% 换手{s.get('turnover',0)}% "
                    f"MA5/20:{s.get('ma5')}/{s.get('ma20')} RSI:{s.get('rsi')}"
                )
            
            prompt = f"""分析以下{len(stocks)}只A股，返回严格JSON数组（不要Markdown）：

{chr(10).join(stock_list)}

返回格式：[{{"id":1,"action":"buy/hold/sell","score":0-100,"confidence":0-100,"reason":"10字内","risk":"低/中/高"}}, ...]
只返回JSON数组，不要其他内容。"""

            import litellm
            resp = litellm.completion(
                model="deepseek/deepseek-chat",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=500,
                temperature=0.3,
            )
            resp_text = resp.choices[0].message.content if resp else ""

            # 解析 JSON
            resp_text = resp_text.strip()
            if resp_text.startswith('```'):
                resp_text = resp_text.split('\n', 1)[1] if '\n' in resp_text else resp_text[3:]
            if resp_text.endswith('```'):
                resp_text = resp_text[:-3]
            
            m = re.search(r'\[.*\]', resp_text, re.DOTALL)
            if not m:
                return []
            
            results = json.loads(m.group())
            
            for item in results:
                idx = item.get('id', 0) - 1
                if 0 <= idx < len(stocks):
                    s = stocks[idx]
                    ts = s.get('technical_score', 50)
                    fs = round(ts * 0.4 + item.get('score', 50) * 0.6, 1)
                    s.update({
                        'ai_action': item.get('action', 'hold'),
                        'ai_score': item.get('score', 50),
                        'ai_confidence': item.get('confidence', 50),
                        'ai_reason': item.get('reason', ''),
                        'risk_level': item.get('risk', '中'),
                        'final_score': fs,
                    })
            
            return stocks
            
        except Exception as e:
            logger.debug(f"批量AI失败: {e}")
            return []

    def get_top_recommendations(self, stocks, top_n=20):
        valid = [s for s in stocks if s and 'final_score' in s]
        valid.sort(key=lambda x: x['final_score'], reverse=True)
        
        buy = [s for s in valid if s.get('ai_action') == 'buy' and s.get('final_score', 0) > 55]
        if len(buy) < top_n:
            remain = [s for s in valid if s not in buy]
            remain.sort(key=lambda x: x['final_score'], reverse=True)
            buy.extend(remain[:top_n - len(buy)])
        
        return buy[:top_n]
