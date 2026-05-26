"""技术指标计算 - 多数据源版"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import logging
import time
import random
import requests

logger = logging.getLogger(__name__)


class TechnicalScreener:

    def __init__(self, max_workers: int = 1):
        self.max_workers = max_workers
        self._err_count = 0
        self._first_fail_logged = False

    def batch_calculate(self, stock_list: List[Dict]) -> List[Dict]:
        if not stock_list:
            return []
        total = len(stock_list)
        logger.info(f"技术指标计算: {total} 只")
        results = []
        self._first_fail_logged = False
        self._err_count = 0
        for i, stock in enumerate(stock_list):
            result = self._calc_one(stock)
            if result:
                results.append(result)
            else:
                self._err_count += 1
            time.sleep(random.uniform(0.5, 1.5))
            if (i + 1) % 20 == 0 or (i + 1) >= total:
                logger.info(f"技术进度: {i+1}/{total} (成功{len(results)} 失败{self._err_count})")
        logger.info(f"技术指标完成: {len(results)}只")
        return results

    def filter_top_stocks(self, stocks: List[Dict], top_n: int = 50, min_score: int = 40) -> List[Dict]:
        valid = [s for s in stocks if s and 'technical_score' in s and s['technical_score'] >= min_score]
        valid.sort(key=lambda x: x['technical_score'], reverse=True)
        top = valid[:top_n]
        if top:
            logger.info(f"技术Top{top_n}: {len(top)}只")
        else:
            logger.warning(f"技术Top: 无")
        return top

    def _calc_one(self, stock: Dict) -> Optional[Dict]:
        code = str(stock.get('code', '')).zfill(6)
        name = stock.get('name', '未知')
        try:
            df = self._fetch_kline(code)
            if df is None or len(df) < 20 or '收盘' not in df.columns:
                return None
            close = df['收盘'].values
            high = df['最高'].values if '最高' in df.columns else close
            low = df['最低'].values if '最低' in df.columns else close
            volume = df['成交量'].values if '成交量' in df.columns else np.zeros(len(close))
            latest_price = float(stock.get('price', close[-1]))
            change_pct = float(stock.get('change_pct', 0))
            turnover = float(stock.get('turnover', 0))
            ind = {'change_pct': change_pct, 'turnover': turnover}
            ind.update(self._calc_ma(close))
            ind.update(self._calc_rsi(close))
            ind.update(self._calc_macd(close))
            ind.update(self._calc_kdj(high, low, close))
            ind.update(self._calc_bollinger(close))
            ind.update(self._calc_volume_metrics(volume))
            ind.update(self._calc_price_position(latest_price, close))
            score = self._calc_technical_score(ind, latest_price)
            bp, sl, t1, t2, t3 = self._calc_trade_prices(ind, latest_price)
            vps = self._get_volume_price_status(ind)
            stock.update({
                'ma5': ind.get('ma5', latest_price), 'ma10': ind.get('ma10', latest_price),
                'ma20': ind.get('ma20', latest_price), 'ma60': ind.get('ma60', latest_price),
                'rsi': ind.get('rsi', 50), 'macd': ind.get('macd', 0),
                'macd_signal': ind.get('macd_signal', 0), 'macd_hist': ind.get('macd_hist', 0),
                'kdj_k': ind.get('kdj_k', 50), 'kdj_d': ind.get('kdj_d', 50), 'kdj_j': ind.get('kdj_j', 50),
                'boll_upper': ind.get('boll_upper', latest_price * 1.1),
                'boll_mid': ind.get('boll_mid', latest_price),
                'boll_lower': ind.get('boll_lower', latest_price * 0.9),
                'volume': float(stock.get('volume', volume[-1])),
                'avg_volume': ind.get('avg_volume', volume[-1]),
                'vol_ratio': ind.get('vol_ratio', 1.0),
                'dist_from_ma5': ind.get('dist_ma5', 0), 'dist_from_ma20': ind.get('dist_ma20', 0),
                'is_bullish': ind.get('is_bullish', False),
                'vol_price_status': vps, 'technical_score': score,
                'ideal_buy_price': bp, 'stop_loss_price': sl,
                'target1': t1, 'target2': t2, 'target3': t3,
            })
            return stock
        except Exception as e:
            if not self._first_fail_logged:
                logger.warning(f"💥 {code} {name}: {e}")
                self._first_fail_logged = True
            return None

    def _fetch_kline(self, code: str) -> Optional[pd.DataFrame]:
        # 只用东方财富接口
        try:
            code = str(code).zfill(6)
            market = '1' if code.startswith(('6', '9')) else '0'
            secid = f"{market}.{code}"
            end = datetime.now().strftime('%Y%m%d')
            start = (datetime.now() - timedelta(days=180)).strftime('%Y%m%d')
            url = (
                f"https://push2his.eastmoney.com/api/qt/stock/kline/get?"
                f"secid={secid}&fields1=f1,f2,f3,f4,f5,f6"
                f"&fields2=f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,f61"
                f"&klt=101&fqt=1&beg={start}&end={end}&lmt=200"
            )
            resp = requests.get(url, timeout=15)
            data = resp.json()
            if data.get('data') and data['data'].get('klines'):
                rows = []
                for line in data['data']['klines']:
                    p = line.split(',')
                    if len(p) >= 7:
                        rows.append({'日期': p[0], '开盘': float(p[1]), '收盘': float(p[2]),
                                     '最高': float(p[3]), '最低': float(p[4]),
                                     '成交量': float(p[5]), '成交额': float(p[6])})
                if rows:
                    return pd.DataFrame(rows)
            return None
        except Exception as e:
            if not self._first_fail_logged:
                logger.warning(f"⚠ {code}: {e}")
                self._first_fail_logged = True
            return None

    def _calc_ma(self, close): 
        r = {}; n = len(close)
        if n>=5: r['ma5']=round(np.mean(close[-5:]),2)
        if n>=10: r['ma10']=round(np.mean(close[-10:]),2)
        if n>=20: r['ma20']=round(np.mean(close[-20:]),2)
        r['ma60']=round(np.mean(close[-60:]),2) if n>=60 else r.get('ma20',close[-1])
        m5=r.get('ma5',close[-1]); m10=r.get('ma10',close[-1]); m20=r.get('ma20',close[-1])
        r['is_bullish']=m5>m10>m20
        return r

    def _calc_rsi(self, close, period=14):
        n=len(close)
        if n<period+1: return {'rsi':50}
        d=np.diff(close[-(period+1):]); g=np.where(d>0,d,0); l=np.where(d<0,-d,0)
        ag=np.mean(g); al=np.mean(l)
        return {'rsi':round(100-(100/(1+ag/al)),1) if al>0 else (100 if ag>0 else 50)}

    def _calc_macd(self, close):
        r={'macd':0,'macd_signal':0,'macd_hist':0}
        if len(close)<35: return r
        e12=self._ema(close,12); e26=self._ema(close,26); d=e12-e26
        de=self._ema(pd.Series(d).values,9); h=2*(d-de)
        r['macd']=round(d[-1],4); r['macd_signal']=round(de[-1],4); r['macd_hist']=round(h[-1],4)
        return r

    def _calc_kdj(self, h, l, c, n=9):
        r={'kdj_k':50,'kdj_d':50,'kdj_j':50}
        if len(c)<n: return r
        rc=c[-n:]; rh=h[-n:]; rl=l[-n:]
        hi=np.max(rh); lo=np.min(rl)
        if hi==lo: return r
        rsv=(rc[-1]-lo)/(hi-lo)*100
        k=2/3*50+1/3*rsv; d=2/3*50+1/3*k; j=3*k-2*d
        r['kdj_k']=round(k,1); r['kdj_d']=round(d,1); r['kdj_j']=round(j,1)
        return r

    def _calc_bollinger(self, close, period=20):
        n=len(close)
        if n<period: return {'boll_mid':close[-1],'boll_upper':close[-1]*1.1,'boll_lower':close[-1]*0.9}
        mid=np.mean(close[-period:]); std=np.std(close[-period:])
        return {'boll_mid':round(mid,2),'boll_upper':round(mid+2*std,2),'boll_lower':round(mid-2*std,2)}

    def _calc_volume_metrics(self, volume):
        n=len(volume); av=np.mean(volume[-6:-1]) if n>=6 else (np.mean(volume[:-1]) if n>=2 else volume[-1])
        return {'avg_volume':round(float(av),0),'vol_ratio':round(volume[-1]/av,2) if av>0 else 1.0}

    def _calc_price_position(self, price, close):
        r={}; n=len(close)
        if n>=5: m5=np.mean(close[-5:]); r['dist_ma5']=round((price-m5)/m5*100,1) if m5>0 else 0
        else: r['dist_ma5']=0
        if n>=20: m20=np.mean(close[-20:]); r['dist_ma20']=round((price-m20)/m20*100,1) if m20>0 else 0
        else: r['dist_ma20']=0
        return r

    def _get_volume_price_status(self, ind):
        c=ind.get('change_pct',0); v=ind.get('vol_ratio',1)
        if c>1 and v>1.5: return "放量上涨"
        elif c>0.5 and v>1.2: return "温和放量上涨"
        elif c>1 and v<0.8: return "缩量上涨"
        elif c<-1 and v>1.5: return "放量下跌"
        elif c<0 and v<0.8: return "缩量下跌"
        elif abs(c)<0.5 and v>2: return "放量滞涨"
        elif abs(c)<0.5 and v<0.5: return "缩量横盘"
        return "量价正常"

    def _calc_technical_score(self, ind, price):
        s=0
        m5=ind.get('ma5',price); m10=ind.get('ma10',price); m20=ind.get('ma20',price); m60=ind.get('ma60',price)
        if m5>m10>m20>m60: s+=25
        elif m5>m10>m20: s+=20
        elif m5>m20 and m10>m20: s+=14
        elif m5<m10<m20: s+=0
        else: s+=8
        rsi=ind.get('rsi',50)
        if 40<=rsi<=60: s+=15
        elif 30<=rsi<=70: s+=12
        elif rsi>70: s+=6
        elif rsi<30: s+=8
        else: s+=8
        macd=ind.get('macd',0); ms=ind.get('macd_signal',0); mh=ind.get('macd_hist',0)
        if macd>0 and macd>ms and mh>0: s+=15
        elif macd>0 and macd>ms: s+=12
        elif macd>ms: s+=10
        elif macd>0: s+=7
        else: s+=3
        cp=ind.get('change_pct',0); vr=ind.get('vol_ratio',1); to=ind.get('turnover',0)
        if cp>1 and vr>1.5: s+=20
        elif cp>0.5 and vr>1.2: s+=16
        elif cp>1 and vr<0.8: s+=6
        elif cp<-1 and vr>1.5: s+=2
        elif cp<0 and vr<0.8: s+=12
        elif abs(cp)<0.5 and vr>2: s+=3
        else: s+=12
        if 1<=to<=8: pass
        elif to>15: s-=5
        elif 0<to<0.5: s-=3
        dm=ind.get('dist_ma20',0)
        if 0<=dm<=3: s+=15
        elif 3<dm<=5: s+=12
        elif -3<=dm<0: s+=10
        elif dm<-5: s+=5
        else: s+=6
        bu=ind.get('boll_upper',price*1.1); up=(bu-price)/price*100
        if up>15: s+=10
        elif up>10: s+=8
        elif up>5: s+=6
        elif up>3: s+=4
        else: s+=2
        return min(100,max(0,s))

    def _calc_trade_prices(self, ind, price):
        m20=ind.get('ma20',price); m60=ind.get('ma60',price)
        bu=ind.get('boll_upper',price*1.1); bl=ind.get('boll_lower',price*0.95)
        bp=round((m20*0.98+bl)/2,2)
        if bp>price: bp=round(price*0.98,2)
        bp=max(bp,round(price*0.95,2))
        s1=round(m60*0.98,2) if m60>0 else round(price*0.93,2)
        s2=round(bl*0.99,2); s3=round(price*0.93,2)
        sl=min(s1,s2,s3); sl=min(sl,round(price*0.95,2))
        t1=round(bu,2)
        if t1<price*1.03: t1=round(price*1.05,2)
        bw=bu-bl if bl>0 else price*0.1
        t2=round(bu+bw*0.3,2)
        if t2<price*1.06: t2=round(price*1.08,2)
        t3=round(price*1.12,2)
        return bp,sl,t1,t2,t3

    @staticmethod
    def _ema(data, period):
        if len(data)<period: return np.full_like(data,np.mean(data))
        r=np.zeros_like(data); r[0]=data[0]; m=2/(period+1)
        for i in range(1,len(data)): r[i]=(data[i]-r[i-1])*m+r[i-1]
        return r
