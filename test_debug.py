"""诊断脚本：测试K线数据获取和技术指标计算"""
import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

print("=" * 60)
print("测试1: 直接调用akshare获取K线")
print("=" * 60)

import akshare as ak

test_cases = [
    ("sh600519", "贵州茅台"),
    ("sz000001", "平安银行"),
    ("sz300234", "开尔新材"),
    ("sh688333", "铂力特"),
    ("bj920000", "安徽凤凰"),
]

for symbol, name in test_cases:
    try:
        df = ak.stock_zh_a_hist(
            symbol=symbol,
            period="daily",
            start_date="20260101",
            end_date="20260526",
            adjust="qfq"
        )
        if df is not None and not df.empty:
            print(f"✅ {symbol} {name}: {len(df)}条数据, 列: {list(df.columns)[:5]}...")
        else:
            print(f"❌ {symbol} {name}: 返回空数据")
    except Exception as e:
        print(f"❌ {symbol} {name}: {type(e).__name__}: {str(e)[:100]}")

print("\n" + "=" * 60)
print("测试2: 通过TechnicalScreener计算技术指标")
print("=" * 60)

from src.market_scanner import MarketScanner
from src.technical_screener import TechnicalScreener

scanner = MarketScanner()
df = scanner.fetch_all_stocks()
df = scanner.quick_filter(df)
stock_list = scanner.get_stock_list()

test_stocks = stock_list[:5]
print(f"\n测试股票 ({len(test_stocks)}只):")
for s in test_stocks:
    print(f"  code={s['code']}, name={s['name']}, price={s['price']}")

screener = TechnicalScreener(max_workers=1)
results = screener.batch_calculate(test_stocks)

print(f"\n成功: {len(results)} 只")
for r in results:
    print(f"  {r['code']} {r['name']}: score={r.get('technical_score')}, ma5={r.get('ma5')}")

print("\n诊断完成")
