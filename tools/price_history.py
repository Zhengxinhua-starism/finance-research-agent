"""日 K 行情多源获取。

解决什么问题
    `get_stock_price` 原来只打 AKShare 的东方财富接口（push2his.eastmoney.com）。
    本机开了 HTTP_PROXY 时，这条请求会被送进代理，代理到不了东财，于是整次
    行情工具以 ProxyError 失败。投资建议、区间涨跌这类问题会因此拒答。

核心设计决策
    1. **腾讯 → 新浪 → 东财**，而不是继续把东财当唯一源。
       腾讯/新浪是单票直连、负载小、国内可直达；东财字段最全（自带涨跌幅）
       但全市场接口更容易被墙/被代理劫持，所以放最后。
    2. 三路都用 httpx 且 `trust_env=False`。根因不是东财接口本身挂了，
       而是 requests/akshare 会读环境变量里的代理。国内行情站走代理必挂，
       LLM（DeepSeek）才需要代理，两套流量不能共用同一份 trust_env。
    3. 不引入 efinance / DataFetcherManager。那套是交易日推送系统的全市场
       缓存与熔断，体量远超研报 Agent 对「最近 N 根日 K」的需求。
    4. 腾讯、东财取前复权；新浪轻量 K 线接口是不复权。窗口内若遇除权，
       新浪路径会在 notes 里标明，避免把除权缺口当成暴跌。

为什么不用其他方案
    - 不继续调 `ak.stock_zh_a_hist`：它内部用 requests，会重新踩代理。
    - 不接 Tushare：要 token，复现门槛高。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Any, Callable

import httpx

from config import get_config
from data.schemas import PriceBar
from tools.data_client import (
    DataSourceError,
    exchange_prefix,
    normalize_ticker,
    parse_amount,
    parse_ratio,
    parse_report_date,
)

logger = logging.getLogger(__name__)

PROVIDER_TENCENT = "tencent"
PROVIDER_SINA = "sina"
PROVIDER_EASTMONEY = "eastmoney"

DEFAULT_SOURCE_ORDER = (PROVIDER_TENCENT, PROVIDER_SINA, PROVIDER_EASTMONEY)

SOURCE_LABELS = {
    PROVIDER_TENCENT: "腾讯财经日 K · 前复权（直连）",
    PROVIDER_SINA: "新浪财经日 K · 不复权（直连）",
    PROVIDER_EASTMONEY: "东方财富日 K · 前复权（直连）",
}

_TENCENT_KLINE = "https://web.ifzq.gtimg.cn/appstock/app/fqkline/get"
_SINA_KLINE = "https://money.finance.sina.com.cn/quotes_service/api/json_v2.php/CN_MarketData.getKLineData"
_EASTMONEY_KLINE = "https://push2his.eastmoney.com/api/qt/stock/kline/get"

_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
)


@dataclass
class PriceFetchResult:
    bars: list[PriceBar] = field(default_factory=list)
    provider: str = ""
    notes: list[str] = field(default_factory=list)
    company_name: str = ""

    @property
    def source_label(self) -> str:
        return SOURCE_LABELS.get(self.provider, self.provider or "未知行情源")

    @property
    def is_qfq(self) -> bool:
        return self.provider != PROVIDER_SINA


def fetch_company_name(ticker: str) -> str | None:
    """用腾讯实时报价拿简称。东财 `stock_individual_info_em` 走代理会挂。"""
    symbol = exchange_prefix(ticker)
    urls = (f"https://qt.gtimg.cn/q={symbol}", f"http://qt.gtimg.cn/q={symbol}")
    with _direct_client(timeout=5.0) as client:
        for url in urls:
            try:
                response = client.get(url, headers={"Referer": "https://finance.qq.com/"})
                response.raise_for_status()
                text = response.content.decode("gbk", errors="replace")
            except Exception as exc:  # noqa: BLE001
                logger.debug("腾讯简称接口失败 %s: %s", url, exc)
                continue
            name = _parse_tencent_quote_name(text, symbol)
            if name:
                return name
    return None


def _parse_tencent_quote_name(text: str, symbol: str) -> str:
    start = text.find('"')
    end = text.rfind('"')
    if start == -1 or end <= start:
        return ""
    payload = text[start + 1 : end]
    parts = payload.split("~")
    if len(parts) > 1 and parts[1].strip() and parts[1].strip() != symbol:
        return parts[1].strip()
    return ""


def fetch_price_history(ticker: str, days: int = 30) -> list[PriceBar]:
    """兼容 FinancialDataClient.get_price_history 的简单签名。"""
    return fetch_price_history_with_meta(ticker, days=days).bars


def fetch_price_history_with_meta(ticker: str, days: int = 30) -> PriceFetchResult:
    """按配置顺序逐源尝试，第一路成功即返回。"""
    days = max(1, min(int(days), 250))
    code = normalize_ticker(ticker)
    symbol = exchange_prefix(code)
    end = date.today()
    start = end - timedelta(days=max(days * 2, days + 15))

    result = PriceFetchResult()
    fetchers: dict[str, Callable[[], PriceFetchResult]] = {
        PROVIDER_TENCENT: lambda: _fetch_tencent(symbol, start, end, days),
        PROVIDER_SINA: lambda: _fetch_sina(symbol, days),
        PROVIDER_EASTMONEY: lambda: _fetch_eastmoney(code, start, end, days),
    }

    for provider in _source_order():
        fetcher = fetchers.get(provider)
        if fetcher is None:
            result.notes.append(f"未知行情源 {provider}，已跳过")
            continue
        try:
            fetched = fetcher()
        except Exception as exc:  # noqa: BLE001 — 各源异常类型不可枚举，失败就换下一路
            logger.warning("行情源 %s 失败 (%s): %s", provider, code, exc)
            result.notes.append(f"{SOURCE_LABELS.get(provider, provider)} 失败：{type(exc).__name__}: {exc}")
            continue
        if not fetched.bars:
            result.notes.append(f"{SOURCE_LABELS.get(provider, provider)} 返回空数据")
            continue
        fetched.notes = result.notes + fetched.notes
        if provider == PROVIDER_SINA:
            fetched.notes.append("新浪日 K 为不复权；窗口内若有除权，区间涨跌可能含缺口")
        logger.info("行情源命中 %s：%s 共 %s 根", provider, code, len(fetched.bars))
        return fetched

    if not result.notes:
        result.notes.append("未配置任何可用行情源")
    raise DataSourceError(f"{code} 行情数据源全部不可用：" + "；".join(result.notes))


def _source_order() -> tuple[str, ...]:
    raw = getattr(get_config(), "price_source_priority", "") or ",".join(DEFAULT_SOURCE_ORDER)
    names = [item.strip().lower() for item in str(raw).split(",") if item.strip()]
    valid = tuple(name for name in names if name in SOURCE_LABELS)
    return valid or DEFAULT_SOURCE_ORDER


def _direct_client(timeout: float = 8.0) -> httpx.Client:
    """国内行情站必须绕过系统代理，否则会重现东财 ProxyError。"""
    return httpx.Client(
        timeout=httpx.Timeout(timeout, connect=5.0),
        trust_env=False,
        follow_redirects=True,
        headers={"User-Agent": _USER_AGENT, "Accept": "application/json,text/plain,*/*"},
    )


def _fetch_tencent(symbol: str, start: date, end: date, days: int) -> PriceFetchResult:
    lookback = max(30, min(800, int(days * 1.8) + 20))
    param = f"{symbol},day,{start.isoformat()},{end.isoformat()},{lookback},qfq"
    with _direct_client() as client:
        response = client.get(
            _TENCENT_KLINE,
            params={"param": param},
            headers={"Referer": "https://finance.qq.com/"},
        )
        response.raise_for_status()
        payload = response.json()
    rows = _parse_tencent_payload(payload, symbol)
    return PriceFetchResult(
        bars=_finalize_bars(rows, days, volume_is_lots=True),
        provider=PROVIDER_TENCENT,
        company_name=_tencent_company_name(payload, symbol),
    )


def _parse_tencent_payload(payload: dict[str, Any], symbol: str) -> list[dict[str, Any]]:
    data = payload.get("data") if isinstance(payload, dict) else None
    item = data.get(symbol) if isinstance(data, dict) else None
    if not isinstance(item, dict):
        return []
    raw_rows = item.get("qfqday") or item.get("day") or []
    parsed: list[dict[str, Any]] = []
    for row in raw_rows:
        if not isinstance(row, list) or len(row) < 6:
            continue
        parsed.append(
            {
                "date": row[0],
                "open": row[1],
                "close": row[2],
                "high": row[3],
                "low": row[4],
                "volume": row[5],
                "amount": row[6] if len(row) > 6 else None,
            }
        )
    return parsed


def _tencent_company_name(payload: dict[str, Any], symbol: str) -> str:
    data = payload.get("data") if isinstance(payload, dict) else None
    item = data.get(symbol) if isinstance(data, dict) else None
    qt = item.get("qt") if isinstance(item, dict) else None
    raw = qt.get(symbol) if isinstance(qt, dict) else None
    if isinstance(raw, list) and len(raw) > 1:
        name = str(raw[1]).strip()
        return name if name and name != symbol else ""
    if isinstance(raw, str) and "~" in raw:
        parts = raw.split("~")
        if len(parts) > 1:
            return parts[1].strip()
    return ""


def _fetch_sina(symbol: str, days: int) -> PriceFetchResult:
    datalen = max(days + 10, min(days * 2, 1023))
    params = {"symbol": symbol, "scale": 240, "ma": "no", "datalen": datalen}
    headers = {"Referer": "https://finance.sina.com.cn/"}
    payload: Any = None
    last_error: Exception | None = None
    with _direct_client() as client:
        for url in (_SINA_KLINE, _SINA_KLINE.replace("https://", "http://")):
            try:
                response = client.get(url, params=params, headers=headers)
                response.raise_for_status()
                payload = response.json()
                break
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                payload = None
    if payload is None and last_error is not None:
        raise last_error
    if not isinstance(payload, list):
        return PriceFetchResult(provider=PROVIDER_SINA)
    rows: list[dict[str, Any]] = []
    for item in payload:
        if not isinstance(item, dict):
            continue
        rows.append(
            {
                "date": item.get("day") or item.get("date"),
                "open": item.get("open"),
                "high": item.get("high"),
                "low": item.get("low"),
                "close": item.get("close"),
                "volume": item.get("volume"),
                "amount": item.get("amount"),
            }
        )
    return PriceFetchResult(
        bars=_finalize_bars(rows, days, volume_is_lots=False),
        provider=PROVIDER_SINA,
    )


def _fetch_eastmoney(code: str, start: date, end: date, days: int) -> PriceFetchResult:
    prefix = exchange_prefix(code)[:2]
    market = {"sh": "1", "sz": "0", "bj": "0"}.get(prefix, "0")
    params = {
        "fields1": "f1,f2,f3,f4,f5,f6",
        "fields2": "f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,f61",
        "ut": "7eea3edcaed734bea9cbfc24409ed989",
        "klt": "101",
        "fqt": "1",  # 前复权
        "secid": f"{market}.{code}",
        "beg": start.strftime("%Y%m%d"),
        "end": end.strftime("%Y%m%d"),
    }
    with _direct_client() as client:
        response = client.get(
            _EASTMONEY_KLINE,
            params=params,
            headers={"Referer": "https://quote.eastmoney.com/"},
        )
        response.raise_for_status()
        payload = response.json()
    klines = ((payload.get("data") or {}) if isinstance(payload, dict) else {}).get("klines") or []
    rows: list[dict[str, Any]] = []
    for item in klines:
        parts = str(item).split(",")
        if len(parts) < 7:
            continue
        rows.append(
            {
                "date": parts[0],
                "open": parts[1],
                "close": parts[2],
                "high": parts[3],
                "low": parts[4],
                "volume": parts[5],
                "amount": parts[6],
                "change_pct": parts[8] if len(parts) > 8 else None,
            }
        )
    return PriceFetchResult(
        bars=_finalize_bars(rows, days, volume_is_lots=False),
        provider=PROVIDER_EASTMONEY,
    )


def _finalize_bars(
    rows: list[dict[str, Any]],
    days: int,
    *,
    volume_is_lots: bool,
) -> list[PriceBar]:
    bars: list[PriceBar] = []
    for row in rows:
        trade_date = parse_report_date(row.get("date"))
        close = parse_amount(row.get("close"))
        if trade_date is None or close is None:
            continue
        volume = parse_amount(row.get("volume"))
        if volume is not None and volume_is_lots:
            volume *= 100
        bars.append(
            PriceBar(
                trade_date=trade_date,
                open=parse_amount(row.get("open")),
                high=parse_amount(row.get("high")),
                low=parse_amount(row.get("low")),
                close=close,
                volume=volume,
                turnover=parse_amount(row.get("amount")),
                change_pct=parse_ratio(row.get("change_pct")) if row.get("change_pct") is not None else None,
            )
        )
    bars.sort(key=lambda bar: bar.trade_date)
    # 去重：多源接口偶发重复交易日
    unique: list[PriceBar] = []
    seen: set[date] = set()
    for bar in bars:
        if bar.trade_date in seen:
            continue
        seen.add(bar.trade_date)
        unique.append(bar)
    unique = unique[-days:]
    _fill_change_pct(unique)
    return unique


def _fill_change_pct(bars: list[PriceBar]) -> None:
    for index, bar in enumerate(bars):
        if bar.change_pct is not None:
            continue
        if index == 0 or not bars[index - 1].close or not bar.close:
            continue
        prev = bars[index - 1].close
        if prev:
            bar.change_pct = (bar.close - prev) / prev
