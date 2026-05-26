# -*- coding: utf-8 -*-
"""
===================================
A股自选股智能分析系统 - 主调度程序
===================================

职责：
1. 协调各模块完成股票分析流程
2. 实现低并发的线程池调度
3. 全局异常处理，确保单股失败不影响整体
4. 提供命令行入口

使用方式：
    python main.py              # 正常运行
    python main.py --debug      # 调试模式
    python main.py --dry-run    # 仅获取数据不分析

交易理念（已融入分析）：
- 严进策略：不追高，乖离率 > 5% 不买入
- 趋势交易：只做 MA5>MA10>MA20 多头排列
- 效率优先：关注筹码集中度好的股票
- 买点偏好：缩量回踩 MA5/MA10 支撑
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from dotenv import dotenv_values
from src.config import setup_env

_INITIAL_PROCESS_ENV = dict(os.environ)
setup_env()

# 代理配置 - 通过 USE_PROXY 环境变量控制，默认关闭
# GitHub Actions 环境自动跳过代理配置
IS_GITHUB_ACTIONS = os.getenv("GITHUB_ACTIONS", "false").lower() == "true"

if not IS_GITHUB_ACTIONS and os.getenv("USE_PROXY", "false").lower() == "true":
    # 本地开发环境，启用代理（可在 .env 中配置 PROXY_HOST 和 PROXY_PORT）
    proxy_host = os.getenv("PROXY_HOST", "127.0.0.1")
    proxy_port = os.getenv("PROXY_PORT", "10809")
    proxy_url = f"http://{proxy_host}:{proxy_port}"
    os.environ["http_proxy"] = proxy_url
    os.environ["https_proxy"] = proxy_url
    print(f"[INFO] 已启用代理: {proxy_host}:{proxy_port}")
else:
    # GitHub Actions 环境，确保无代理
    os.environ.pop("http_proxy", None)
    os.environ.pop("https_proxy", None)
    if IS_GITHUB_ACTIONS:
        print("[INFO] GitHub Actions 环境，已禁用代理")

import argparse
import logging
import sys
import time
import uuid
from datetime import datetime, timezone, timedelta

from data_provider.base import canonical_stock_code
from src.webui_frontend import prepare_webui_frontend_assets
from src.config import get_config, Config
from src.logging_config import setup_logging


logger = logging.getLogger(__name__)
_RUNTIME_ENV_FILE_KEYS = set()


def _get_active_env_path() -> Path:
    env_file = os.getenv("ENV_FILE")
    if env_file:
        return Path(env_file)
    return Path(__file__).resolve().parent / ".env"


def _read_active_env_values() -> Optional[Dict[str, str]]:
    env_path = _get_active_env_path()
    if not env_path.exists():
        return {}

    try:
        values = dotenv_values(env_path)
    except Exception as exc:  # pragma: no cover - defensive branch
        logger.warning("读取配置文件 %s 失败，继续沿用当前环境变量: %s", env_path, exc)
        return None

    return {
        str(key): "" if value is None else str(value)
        for key, value in values.items()
        if key is not None
    }


_ACTIVE_ENV_FILE_VALUES = _read_active_env_values() or {}
_RUNTIME_ENV_FILE_KEYS = {
    key for key in _ACTIVE_ENV_FILE_VALUES
    if key not in _INITIAL_PROCESS_ENV
}

# setup_env() already ran at import time above.
_env_bootstrapped = True


def _bootstrap_environment() -> None:
    """Load .env and apply optional local proxy settings.

    Guarded to be idempotent so it can safely be called from lazy-import
    paths used by API / bot consumers.
    """
    global _env_bootstrapped
    if _env_bootstrapped:
        return

    from src.config import setup_env

    setup_env()

    if os.getenv("GITHUB_ACTIONS") != "true" and os.getenv("USE_PROXY", "false").lower() == "true":
        proxy_host = os.getenv("PROXY_HOST", "127.0.0.1")
        proxy_port = os.getenv("PROXY_PORT", "10809")
        proxy_url = f"http://{proxy_host}:{proxy_port}"
        os.environ["http_proxy"] = proxy_url
        os.environ["https_proxy"] = proxy_url

    _env_bootstrapped = True


def _setup_bootstrap_logging(debug: bool = False) -> None:
    """Initialize stderr-only logging before config is loaded."""
    level = logging.DEBUG if debug else logging.INFO
    root = logging.getLogger()
    root.setLevel(level)
    if not any(
        isinstance(h, logging.StreamHandler) and getattr(h, "stream", None) is sys.stderr
        for h in root.handlers
    ):
        handler = logging.StreamHandler(sys.stderr)
        handler.setLevel(level)
        handler.setFormatter(
            logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
        )
        root.addHandler(handler)


def _setup_runtime_logging(log_dir: str, debug: bool = False) -> bool:
    """Switch to configured logging, falling back to console on file I/O errors."""
    try:
        setup_logging(log_prefix="stock_analysis", debug=debug, log_dir=log_dir)
        return True
    except OSError as exc:
        logger.warning(
            "文件日志初始化失败，已降级为控制台日志输出；日志目录 %r 当前不可写或不可创建: %s。",
            log_dir, exc,
        )
        return False


def _get_stock_analysis_pipeline():
    """Lazily import StockAnalysisPipeline for external consumers."""
    _bootstrap_environment()
    from src.core.pipeline import StockAnalysisPipeline as _Pipeline
    return _Pipeline


class _LazyPipelineDescriptor:
    """Descriptor that resolves StockAnalysisPipeline on first attribute access."""
    _resolved = None

    def __set_name__(self, owner, name):
        self._name = name

    def __get__(self, obj, objtype=None):
        if self._resolved is None:
            self._resolved = _get_stock_analysis_pipeline()
        return self._resolved


class _ModuleExports:
    StockAnalysisPipeline = _LazyPipelineDescriptor()


_exports = _ModuleExports()


def __getattr__(name: str):
    if name == "StockAnalysisPipeline":
        return _exports.StockAnalysisPipeline
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def _reload_env_file_values_preserving_overrides() -> None:
    """Refresh `.env`-managed env vars without clobbering process env overrides."""
    global _RUNTIME_ENV_FILE_KEYS

    latest_values = _read_active_env_values()
    if latest_values is None:
        return

    managed_keys = {
        key for key in latest_values
        if key not in _INITIAL_PROCESS_ENV
    }

    for key in _RUNTIME_ENV_FILE_KEYS - managed_keys:
        os.environ.pop(key, None)

    for key in managed_keys:
        os.environ[key] = latest_values[key]

    _RUNTIME_ENV_FILE_KEYS = managed_keys


def parse_arguments() -> argparse.Namespace:
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description='A股自选股智能分析系统',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
示例:
  python main.py                    # 正常运行
  python main.py --debug            # 调试模式
  python main.py --dry-run          # 仅获取数据，不进行 AI 分析
  python main.py --stocks 600519,000001  # 指定分析特定股票
  python main.py --no-notify        # 不发送推送通知
  python main.py --check-notify     # 检查通知配置，不发送通知
  python main.py --single-notify    # 启用单股推送模式
  python main.py --schedule         # 启用定时任务模式
  python main.py --market-review    # 仅运行大盘复盘
        '''
    )

    parser.add_argument('--debug', action='store_true', help='启用调试模式')
    parser.add_argument('--dry-run', action='store_true', help='仅获取数据，不进行 AI 分析')
    parser.add_argument('--stocks', type=str, help='指定要分析的股票代码，逗号分隔')
    parser.add_argument('--no-notify', action='store_true', help='不发送推送通知')
    parser.add_argument('--check-notify', action='store_true', help='只读检查通知渠道配置')
    parser.add_argument('--single-notify', action='store_true', help='启用单股推送模式')
    parser.add_argument('--workers', type=int, default=None, help='并发线程数')
    parser.add_argument('--schedule', action='store_true', help='启用定时任务模式')
    parser.add_argument('--no-run-immediately', action='store_true', help='定时任务启动时不立即执行')
    parser.add_argument('--market-review', action='store_true', help='仅运行大盘复盘分析')
    parser.add_argument('--no-market-review', action='store_true', help='跳过大盘复盘分析')
    parser.add_argument('--force-run', action='store_true', help='跳过交易日检查')
    parser.add_argument('--webui', action='store_true', help='启动 Web 管理界面')
    parser.add_argument('--webui-only', action='store_true', help='仅启动 Web 服务')
    parser.add_argument('--serve', action='store_true', help='启动 FastAPI 后端服务')
    parser.add_argument('--serve-only', action='store_true', help='仅启动 FastAPI 后端服务')
    parser.add_argument('--port', type=int, default=8000, help='FastAPI 服务端口')
    parser.add_argument('--host', type=str, default='0.0.0.0', help='FastAPI 服务监听地址')
    parser.add_argument('--no-context-snapshot', action='store_true', help='不保存分析上下文快照')
    parser.add_argument('--backtest', action='store_true', help='运行回测')
    parser.add_argument('--backtest-code', type=str, default=None, help='仅回测指定股票')
    parser.add_argument('--backtest-days', type=int, default=None, help='回测评估窗口')
    parser.add_argument('--backtest-force', action='store_true', help='强制回测')

    return parser.parse_args()


def _compute_trading_day_filter(
    config: Config,
    args: argparse.Namespace,
    stock_codes: List[str],
) -> Tuple[List[str], Optional[str], bool]:
    force_run = getattr(args, 'force_run', False)
    if force_run or not getattr(config, 'trading_day_check_enabled', True):
        return (stock_codes, None, False)

    from src.core.trading_calendar import (
        get_market_for_stock,
        get_open_markets_today,
        compute_effective_region,
    )

    open_markets = get_open_markets_today()
    filtered_codes = []
    for code in stock_codes:
        mkt = get_market_for_stock(code)
        if mkt in open_markets or mkt is None:
            filtered_codes.append(code)

    if config.market_review_enabled and not getattr(args, 'no_market_review', False):
        effective_region = compute_effective_region(
            getattr(config, 'market_review_region', 'cn') or 'cn', open_markets
        )
    else:
        effective_region = None

    should_skip_all = (not filtered_codes) and (effective_region or '') == ''
    return (filtered_codes, effective_region, should_skip_all)


def _run_market_review_with_shared_lock(
    config: Config,
    run_market_review_func: Callable[..., Optional[str]],
    **kwargs: Any,
) -> Optional[str]:
    from src.core.market_review_lock import (
        release_market_review_lock,
        try_acquire_market_review_lock,
    )

    lock_token = try_acquire_market_review_lock(config)
    if lock_token is None:
        logger.warning("大盘复盘正在执行中，跳过本次大盘复盘")
        return None

    try:
        return run_market_review_func(**kwargs)
    finally:
        release_market_review_lock(lock_token)


def run_full_analysis(
    config: Config,
    args: argparse.Namespace,
    stock_codes: Optional[List[str]] = None
):
    from src.core.market_review import run_market_review
    from src.core.pipeline import StockAnalysisPipeline

    try:
        if stock_codes is None:
            config.refresh_stock_list()

        effective_codes = stock_codes if stock_codes is not None else config.stock_list
        filtered_codes, effective_region, should_skip = _compute_trading_day_filter(
            config, args, effective_codes
        )
        if should_skip:
            logger.info("今日所有相关市场均为非交易日，跳过执行。可使用 --force-run 强制执行。")
            return
        if set(filtered_codes) != set(effective_codes):
            skipped = set(effective_codes) - set(filtered_codes)
            logger.info("今日休市股票已跳过: %s", skipped)
        stock_codes = filtered_codes

        if getattr(args, 'single_notify', False):
            config.single_stock_notify = True

        merge_notification = (
            getattr(config, 'merge_email_notification', False)
            and config.market_review_enabled
            and not getattr(args, 'no_market_review', False)
            and not config.single_stock_notify
        )

        save_context_snapshot = None
        if getattr(args, 'no_context_snapshot', False):
            save_context_snapshot = False
        query_id = uuid.uuid4().hex
        pipeline = StockAnalysisPipeline(
            config=config,
            max_workers=args.workers,
            query_id=query_id,
            query_source="cli",
            save_context_snapshot=save_context_snapshot
        )

        results = pipeline.run(
            stock_codes=stock_codes,
            dry_run=args.dry_run,
            send_notification=not args.no_notify,
            merge_notification=merge_notification
        )

        analysis_delay = getattr(config, 'analysis_delay', 0)
        if (
            analysis_delay > 0
            and config.market_review_enabled
            and not args.no_market_review
            and effective_region != ''
        ):
            logger.info(f"等待 {analysis_delay} 秒后执行大盘复盘（避免API限流）...")
            time.sleep(analysis_delay)

        market_report = ""
        if (
            config.market_review_enabled
            and not args.no_market_review
            and effective_region != ''
        ):
            review_result = _run_market_review_with_shared_lock(
                config,
                run_market_review,
                notifier=pipeline.notifier,
                analyzer=pipeline.analyzer,
                search_service=pipeline.search_service,
                send_notification=not args.no_notify,
                merge_notification=merge_notification,
                override_region=effective_region,
            )
            if review_result:
                market_report = review_result

        if merge_notification and (results or market_report) and not args.no_notify:
            parts = []
            if market_report:
                parts.append(f"# 📈 大盘复盘\n\n{market_report}")
            if results:
                dashboard_content = pipeline.notifier.generate_aggregate_report(
                    results,
                    getattr(config, 'report_type', 'simple'),
                )
                parts.append(f"# 🚀 个股决策仪表盘\n\n{dashboard_content}")
            if parts:
                combined_content = "\n\n---\n\n".join(parts)
                if pipeline.notifier.is_available():
                    if pipeline.notifier.send(combined_content, email_send_to_all=True, route_type="report"):
                        logger.info("已合并推送（个股+大盘复盘）")
                    else:
                        logger.warning("合并推送失败")

        if results:
            logger.info("\n===== 分析结果摘要 =====")
            for r in sorted(results, key=lambda x: x.sentiment_score, reverse=True):
                emoji = r.get_emoji()
                logger.info(
                    f"{emoji} {r.name}({r.code}): {r.operation_advice} | "
                    f"评分 {r.sentiment_score} | {r.trend_prediction}"
                )

        logger.info("\n任务执行完成")

        try:
            from src.feishu_doc import FeishuDocManager
            feishu_doc = FeishuDocManager()
            if feishu_doc.is_configured() and (results or market_report):
                logger.info("正在创建飞书云文档...")
                tz_cn = timezone(timedelta(hours=8))
                now = datetime.now(tz_cn)
                doc_title = f"{now.strftime('%Y-%m-%d %H:%M')} 大盘复盘"
                full_content = ""
                if market_report:
                    full_content += f"# 📈 大盘复盘\n\n{market_report}\n\n---\n\n"
                if results:
                    dashboard_content = pipeline.notifier.generate_aggregate_report(
                        results,
                        getattr(config, 'report_type', 'simple'),
                    )
                    full_content += f"# 🚀 个股决策仪表盘\n\n{dashboard_content}"
                doc_url = feishu_doc.create_daily_doc(doc_title, full_content)
                if doc_url:
                    logger.info(f"飞书云文档创建成功: {doc_url}")
                    if not args.no_notify:
                        pipeline.notifier.send(
                            f"[{now.strftime('%Y-%m-%d %H:%M')}] 复盘文档创建成功: {doc_url}",
                            route_type="report",
                        )
        except Exception as e:
            logger.error(f"飞书文档生成失败: {e}")

        try:
            if getattr(config, 'backtest_enabled', False):
                from src.services.backtest_service import BacktestService
                logger.info("开始自动回测...")
                service = BacktestService()
                stats = service.run_backtest(
                    force=False,
                    eval_window_days=getattr(config, 'backtest_eval_window_days', 10),
                    min_age_days=getattr(config, 'backtest_min_age_days', 14),
                    limit=200,
                )
                logger.info(
                    f"自动回测完成: processed={stats.get('processed')} saved={stats.get('saved')} "
                    f"completed={stats.get('completed')} insufficient={stats.get('insufficient')} errors={stats.get('errors')}"
                )
        except Exception as e:
            logger.warning(f"自动回测失败（已忽略）: {e}")

    except Exception as e:
        logger.exception(f"分析流程执行失败: {e}")


def start_api_server(host: str, port: int, config: Config) -> None:
    import threading
    import uvicorn

    def run_server():
        level_name = (config.log_level or "INFO").lower()
        uvicorn.run(
            "api.app:app",
            host=host,
            port=port,
            log_level=level_name,
            log_config=None,
        )

    thread = threading.Thread(target=run_server, daemon=True)
    thread.start()
    logger.info(f"FastAPI 服务已启动: http://{host}:{port}")


def _is_truthy_env(var_name: str, default: str = "true") -> bool:
    value = os.getenv(var_name, default).strip().lower()
    return value not in {"0", "false", "no", "off"}


def start_bot_stream_clients(config: Config) -> None:
    if config.dingtalk_stream_enabled:
        try:
            from bot.platforms import start_dingtalk_stream_background, DINGTALK_STREAM_AVAILABLE
            if DINGTALK_STREAM_AVAILABLE:
                if start_dingtalk_stream_background():
                    logger.info("[Main] Dingtalk Stream client started in background.")
                else:
                    logger.warning("[Main] Dingtalk Stream client failed to start.")
            else:
                logger.warning("[Main] Dingtalk Stream enabled but SDK is missing.")
        except Exception as exc:
            logger.error(f"[Main] Failed to start Dingtalk Stream client: {exc}")

    if getattr(config, 'feishu_stream_enabled', False):
        try:
            from bot.platforms import start_feishu_stream_background, FEISHU_SDK_AVAILABLE
            if FEISHU_SDK_AVAILABLE:
                if start_feishu_stream_background():
                    logger.info("[Main] Feishu Stream client started in background.")
                else:
                    logger.warning("[Main] Feishu Stream client failed to start.")
            else:
                logger.warning("[Main] Feishu Stream enabled but SDK is missing.")
        except Exception as exc:
            logger.error(f"[Main] Failed to start Feishu Stream client: {exc}")


def _resolve_scheduled_stock_codes(stock_codes: Optional[List[str]]) -> Optional[List[str]]:
    if stock_codes is not None:
        logger.warning(
            "定时模式下检测到 --stocks 参数；计划执行将忽略启动时股票快照。"
        )
    return None


def _reload_runtime_config() -> Config:
    _reload_env_file_values_preserving_overrides()
    Config.reset_instance()
    return get_config()


def _build_schedule_time_provider(default_schedule_time: str):
    from src.core.config_manager import ConfigManager
    _SYSTEM_DEFAULT_SCHEDULE_TIME = "18:00"
    manager = ConfigManager()

    def _provider() -> str:
        if "SCHEDULE_TIME" in _INITIAL_PROCESS_ENV:
            return os.getenv("SCHEDULE_TIME", default_schedule_time)
        config_map = manager.read_config_map()
        schedule_time = (config_map.get("SCHEDULE_TIME", "") or "").strip()
        if schedule_time:
            return schedule_time
        return _SYSTEM_DEFAULT_SCHEDULE_TIME

    return _provider

def main() -> int:
    """
    主入口函数

    Returns:
        退出码（0 表示成功）
    """
    # 解析命令行参数
    args = parse_arguments()

    # 在配置加载前先初始化 bootstrap 日志，确保早期失败也能落盘
    try:
        _setup_bootstrap_logging(debug=args.debug)
    except Exception as exc:
        logging.basicConfig(
            level=logging.DEBUG if getattr(args, "debug", False) else logging.INFO,
            format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            stream=sys.stderr,
        )
        logger.warning("Bootstrap 日志初始化失败，已回退到 stderr: %s", exc)

    # 加载配置（在 bootstrap logging 之后执行，确保异常有日志）
    try:
        config = get_config()
    except Exception as exc:
        logger.exception("加载配置失败: %s", exc)
        return 1

    # 配置日志（输出到控制台和文件）
    try:
        _setup_runtime_logging(config.log_dir, debug=args.debug)
    except Exception as exc:
        logger.exception("切换到配置日志目录失败: %s", exc)
        return 1

    logger.info("=" * 60)
    logger.info("A股自选股智能分析系统 启动")
    logger.info(f"运行时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("=" * 60)

    # 验证配置
    warnings = config.validate()
    for warning in warnings:
        logger.warning(warning)

    if getattr(args, "check_notify", False):
        from src.services.notification_diagnostics import (
            format_notification_diagnostics,
            run_notification_diagnostics,
        )

        result = run_notification_diagnostics(config)
        print(format_notification_diagnostics(result))
        return 0 if result.ok else 1

    # 解析股票列表（统一为大写 Issue #355）
    stock_codes = None
    if args.stocks:
        stock_codes = [canonical_stock_code(c) for c in args.stocks.split(',') if (c or "").strip()]
        logger.info(f"使用命令行指定的股票列表: {stock_codes}")

    # === 处理 --webui / --webui-only 参数，映射到 --serve / --serve-only ===
    if args.webui:
        args.serve = True
    if args.webui_only:
        args.serve_only = True

    # 兼容旧版 WEBUI_ENABLED 环境变量
    if config.webui_enabled and not (args.serve or args.serve_only):
        args.serve = True

    # === 启动 Web 服务 (如果启用) ===
    start_serve = (args.serve or args.serve_only) and os.getenv("GITHUB_ACTIONS") != "true"

    # 兼容旧版 WEBUI_HOST/WEBUI_PORT：如果用户未通过 --host/--port 指定，则使用旧变量
    if start_serve:
        if args.host == '0.0.0.0' and os.getenv('WEBUI_HOST'):
            args.host = os.getenv('WEBUI_HOST')
        if args.port == 8000 and os.getenv('WEBUI_PORT'):
            args.port = int(os.getenv('WEBUI_PORT'))

    bot_clients_started = False
    if start_serve:
        if not prepare_webui_frontend_assets():
            logger.warning("前端静态资源未就绪，继续启动 FastAPI 服务（Web 页面可能不可用）")
        try:
            start_api_server(host=args.host, port=args.port, config=config)
            bot_clients_started = True
        except Exception as e:
            logger.error(f"启动 FastAPI 服务失败: {e}")

    if bot_clients_started:
        start_bot_stream_clients(config)

    # === 仅 Web 服务模式：不自动执行分析 ===
    if args.serve_only:
        logger.info("模式: 仅 Web 服务")
        logger.info(f"Web 服务运行中: http://{args.host}:{args.port}")
        logger.info("通过 /api/v1/analysis/analyze 接口触发分析")
        logger.info(f"API 文档: http://{args.host}:{args.port}/docs")
        logger.info("按 Ctrl+C 退出...")
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            logger.info("\n用户中断，程序退出")
        return 0

    try:
        # 模式0: 回测
        if getattr(args, 'backtest', False):
            logger.info("模式: 回测")
            from src.services.backtest_service import BacktestService

            service = BacktestService()
            stats = service.run_backtest(
                code=getattr(args, 'backtest_code', None),
                force=getattr(args, 'backtest_force', False),
                eval_window_days=getattr(args, 'backtest_days', None),
            )
            logger.info(
                f"回测完成: processed={stats.get('processed')} saved={stats.get('saved')} "
                f"completed={stats.get('completed')} insufficient={stats.get('insufficient')} errors={stats.get('errors')}"
            )
            return 0

        # 模式1: 仅大盘复盘
        if args.market_review:
            from src.core.market_review import run_market_review
            from src.core.market_review_runtime import build_market_review_runtime

            # Issue #373: Trading day check for market-review-only mode.
            effective_region = None
            if not getattr(args, 'force_run', False) and getattr(config, 'trading_day_check_enabled', True):
                from src.core.trading_calendar import get_open_markets_today, compute_effective_region as _compute_region
                open_markets = get_open_markets_today()
                effective_region = _compute_region(
                    getattr(config, 'market_review_region', 'cn') or 'cn', open_markets
                )
                if effective_region == '':
                    logger.info("今日大盘复盘相关市场均为非交易日，跳过执行。可使用 --force-run 强制执行。")
                    return 0

            logger.info("模式: 仅大盘复盘")
            notifier, analyzer, search_service = build_market_review_runtime(config)

            _run_market_review_with_shared_lock(
                config,
                run_market_review,
                notifier=notifier,
                analyzer=analyzer,
                search_service=search_service,
                send_notification=not args.no_notify,
                override_region=effective_region,
            )
            return 0

        # 模式2: 定时任务模式
        if args.schedule or config.schedule_enabled:
            logger.info("模式: 定时任务")
            logger.info(f"每日执行时间: {config.schedule_time}")

            should_run_immediately = config.schedule_run_immediately
            if getattr(args, 'no_run_immediately', False):
                should_run_immediately = False

            logger.info(f"启动时立即执行: {should_run_immediately}")

            from src.scheduler import run_with_schedule
            scheduled_stock_codes = _resolve_scheduled_stock_codes(stock_codes)
            schedule_time_provider = _build_schedule_time_provider(config.schedule_time)

            def scheduled_task():
                runtime_config = _reload_runtime_config()
                run_full_analysis(runtime_config, args, scheduled_stock_codes)

            background_tasks = []
            if getattr(config, 'agent_event_monitor_enabled', False):
                from src.agent.events import build_event_monitor_from_config, run_event_monitor_once

                monitor = build_event_monitor_from_config(config)
                if monitor is not None:
                    interval_minutes = max(1, getattr(config, 'agent_event_monitor_interval_minutes', 5))

                    def event_monitor_task():
                        triggered = run_event_monitor_once(monitor)
                        if triggered:
                            logger.info("[EventMonitor] 本轮触发 %d 条提醒", len(triggered))

                    background_tasks.append({
                        "task": event_monitor_task,
                        "interval_seconds": interval_minutes * 60,
                        "run_immediately": True,
                        "name": "agent_event_monitor",
                    })
                else:
                    logger.info("EventMonitor 已启用，但未加载到有效规则，跳过后台提醒任务")

            run_with_schedule(
                task=scheduled_task,
                schedule_time=config.schedule_time,
                run_immediately=should_run_immediately,
                background_tasks=background_tasks,
                schedule_time_provider=schedule_time_provider,
            )
            return 0

        # 模式3: 正常单次运行
        if config.run_immediately:
            run_full_analysis(config, args, stock_codes)
        else:
            logger.info("配置为不立即运行分析 (RUN_IMMEDIATELY=false)")

        logger.info("\n程序执行完成")

        # 如果启用了服务且是非定时任务模式，保持程序运行
        keep_running = start_serve and not (args.schedule or config.schedule_enabled)
        if keep_running:
            logger.info("API 服务运行中 (按 Ctrl+C 退出)...")
            try:
                while True:
                    time.sleep(1)
            except KeyboardInterrupt:
                pass

        return 0

    except KeyboardInterrupt:
        logger.info("\n用户中断，程序退出")
        return 130

    except Exception as e:
        logger.exception(f"程序执行失败: {e}")
        return 1

# ============================================================
# 盘中实时扫描功能
# ============================================================

async def run_intraday_scan():
    """盘中实时扫描模式 - 完整版（含大盘环境评估）"""
    import sys
    import os
    from datetime import datetime
    import logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
    logger = logging.getLogger(__name__)
    
    from src.market_scanner import MarketScanner
    from src.technical_screener import TechnicalScreener
    from src.ai_ranker import AIRanker
    from src.recommendation_store import RecommendationStore
    
    scan_start = datetime.now()
    logger.info("=" * 60)
    logger.info("🚀 盘中实时扫描启动")
    logger.info(f"⏰ {scan_start.strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("=" * 60)
    
    try:
        # [1/5] 获取全市场股票
        logger.info("[1/5] 获取全市场股票...")
        scanner = MarketScanner()
        df = scanner.fetch_all_stocks()
        
        if df is None or df.empty:
            logger.error("❌ 获取全市场股票失败")
            return None
        
        logger.info(f"✅ 获取成功: {len(df)} 只")
        
        # [2/5] 快速筛选
        logger.info("[2/5] 快速筛选...")
        df = scanner.quick_filter(df)
        stock_list = scanner.get_stock_list()
        logger.info(f"✅ 筛选后: {len(stock_list)} 只")
        
        if not stock_list:
            logger.warning("⚠️ 筛选后无股票")
            return None
        
        # [2.5/5] 二次筛选 - 缩小到活跃股票
        logger.info("[2.5/5] 二次筛选 - 仅保留活跃股票...")
        
        active_stocks = [
            s for s in stock_list 
            if abs(s.get('change_pct', 0)) > 1.5
        ]
        logger.info(f"策略1 (涨跌幅>1.5%): {len(active_stocks)} 只")
        
        if len(active_stocks) < 100:
            active_stocks = [
                s for s in stock_list 
                if abs(s.get('change_pct', 0)) > 0.5
            ]
            logger.info(f"策略2 (涨跌幅>0.5%): {len(active_stocks)} 只")
        
        if len(active_stocks) < 50:
            logger.info(f"活跃股票不足50只，切换为成交量Top300")
            active_stocks = sorted(
                stock_list, 
                key=lambda x: x.get('volume', 0), 
                reverse=True
            )[:300]
        
        if len(active_stocks) > 200:
            active_stocks.sort(
                key=lambda x: abs(x.get('change_pct', 0)), 
                reverse=True
            )
            active_stocks = active_stocks[:200]
        
        logger.info(f"✅ 二次筛选后: {len(active_stocks)} 只（进入技术分析）")
        
        # [3/5] 技术指标
        logger.info(f"[3/5] 技术指标计算 ({len(active_stocks)} 只)...")
        screener = TechnicalScreener(max_workers=1)
        stocks = screener.batch_calculate(active_stocks)
        top_stocks = screener.filter_top_stocks(stocks, top_n=50)
        logger.info(f"✅ 技术Top50: {len(top_stocks)} 只")
        
        if not top_stocks:
            logger.warning("⚠️ 技术筛选无结果")
            return None
        
        # [4/5] AI分析（含大盘环境评估）
        logger.info(f"[4/5] AI深度分析 ({len(top_stocks)} 只)...")
        ranker = AIRanker(None, max_workers=5)
        analyzed = ranker.batch_analyze(top_stocks)
        recommendations = ranker.get_top_recommendations(analyzed, top_n=20)
        logger.info(f"✅ 推荐: {len(recommendations)} 只")
        
        # [5/5] 保存
        logger.info("[5/5] 保存结果...")
        store = RecommendationStore()
        scan_time = scan_start.strftime('%Y-%m-%d %H:%M')
        store.save_batch(recommendations, scan_time)
        store.export_csv()
        
        # 生成报告
        report = _gen_report(recommendations, scan_time)
        os.makedirs("reports", exist_ok=True)
        path = f"reports/recommend_{scan_start.strftime('%Y%m%d_%H%M')}.md"
        with open(path, 'w', encoding='utf-8') as f:
            f.write(report)
        
        logger.info(f"📄 报告: {path}")
        
        elapsed = (datetime.now() - scan_start).total_seconds()
        logger.info("=" * 60)
        logger.info(f"✅ 完成! 耗时 {elapsed:.0f}s, 推荐 {len(recommendations)} 只")
        logger.info("=" * 60)
        
        return recommendations
        
    except Exception as e:
        logger.error(f"❌ 扫描异常: {e}")
        import traceback
        traceback.print_exc()
        return None


def _gen_report(recommendations, scan_time):
    """生成推荐报告（含大盘环境和仓位建议）"""
    if not recommendations:
        return "# 今日无推荐"

    # 大盘环境信息
    market_info = ""
    total_position = 0
    if recommendations:
        first = recommendations[0]
        market_adj = first.get('market_adjust', 0)
        total_position = sum(s.get('position_pct', 0) for s in recommendations)
        
        market_level = ""
        if market_adj >= 5:
            market_level = "🟢 强势"
        elif market_adj >= 0:
            market_level = "🟡 中性"
        else:
            market_level = "🔴 弱势"
        
        market_info = (
            f"\n### 📊 大盘环境\n"
            f"- 状态: {market_level} | 评分调整: {market_adj:+.0f}分\n"
            f"- 建议总仓位: **{total_position:.0f}%**\n"
        )

    r = f"""# 🎯 A股买入推荐 Top 20

**📅 {scan_time} | 📊 {len(recommendations)} 只**
{market_info}
| # | 代码 | 名称 | 评分 | 现价 | 买入 | 止损 | 目标1 | 目标2 | 目标3 | 空间 | 仓位 | 理由 |
|---|------|------|------|------|------|------|------|------|------|------|------|------|
"""
    for i, s in enumerate(recommendations):
        name = s.get('name', '')[:4]
        score = s.get('final_score', 0)
        
        # 星级
        if score >= 80:
            star = "⭐⭐⭐"
        elif score >= 65:
            star = "⭐⭐"
        elif score >= 55:
            star = "⭐"
        else:
            star = ""
        
        upside = s.get('upside_pct', 0)
        pos = s.get('position_pct', 0)
        reason = s.get('ai_reason', '')
        if len(reason) > 12:
            reason = reason[:12] + '..'
        
        # 操作建议emoji
        action = s.get('ai_action', 'hold')
        action_emoji = "🟢" if action == 'buy' else ("🟡" if action == 'hold' else "🔴")
        
        r += (
            f"| {i+1} | {s.get('code')} | {name} | {score}{star} | "
            f"{s.get('price')} | {s.get('ideal_buy_price')} | {s.get('stop_loss_price')} | "
            f"{s.get('target1')} | {s.get('target2')} | {s.get('target3')} | "
            f"{upside:.0f}% | {pos:.0f}% | {action_emoji}{reason} |\n"
        )

    # 统计
    buy_count = sum(1 for s in recommendations if s.get('ai_action') == 'buy')
    hold_count = sum(1 for s in recommendations if s.get('ai_action') == 'hold')
    avg_upside = sum(s.get('upside_pct', 0) for s in recommendations) / max(len(recommendations), 1)
    avg_score = sum(s.get('final_score', 0) for s in recommendations) / max(len(recommendations), 1)

    r += f"""

---
### 📈 统计概览
| 指标 | 数值 |
|------|------|
| 推荐总数 | {len(recommendations)} 只 |
| 买入信号 | {buy_count} 只 |
| 持有信号 | {hold_count} 只 |
| 平均评分 | {avg_score:.1f} 分 |
| 平均上涨空间 | {avg_upside:.1f}% |
| 建议总仓位 | {total_position:.0f}% |

### 💰 止盈策略
| 目标 | 涨幅 | 卖出比例 | 说明 |
|------|------|----------|------|
| 目标1（保守） | 3-5% | 30% | 到达第一压力位，锁定基础利润 |
| 目标2（中性） | 5-10% | 40% | 突破后趋势延续，利润大头落袋 |
| 目标3（激进） | 10-15%+ | 30% | 趋势加速，吃完整波行情 |

### 🛡️ 风控规则
- **止损**：跌破止损价立即离场，不犹豫
- **移动止盈**：盈利>5%后，止损上移至成本价
- **时间止损**：持有3天不涨，减半仓
- **大盘联动**：大盘跌>1%，所有仓位减半

---
⚠️ 以上为AI分析参考，不构成投资建议 | 生成: {scan_time}
"""
    return r

async def run_closing_push():
    """收盘飞书推送"""
    from src.recommendation_store import RecommendationStore
    store = RecommendationStore()
    df = store.get_latest(20)
    if df.empty:
        logger.warning("无推荐数据")
        return
    recs = df.to_dict('records')
    scan_time = recs[0].get('scan_time', datetime.now().strftime('%Y-%m-%d %H:%M'))
    report = _gen_report(recs, scan_time)
    try:
        from src.services.notification_service import NotificationService
        notification_manager = NotificationService()
        notification_manager.send_message(report, title="🎯 今日A股买入推荐")
        logger.info("飞书推送成功")
    except Exception as e:
        logger.error(f"飞书推送失败: {e}")


# ============================================================
# 入口点
# ============================================================

if __name__ == "__main__":
    import subprocess, sys
    subprocess.run([sys.executable, "test_debug.py"])
    
    # 检查是否指定了盘中扫描模式
    mode = os.getenv("MODE", "")
    
    if mode == "intraday":
        print(f"🚀 模式: intraday | ⏰ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        asyncio.run(run_intraday_scan())
    elif mode == "closing_push":
        print(f"📤 模式: closing_push | ⏰ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        asyncio.run(run_closing_push())
    else:
        sys.exit(main())
