import json
import os
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import yfinance as yf

# ========= CONFIG =========
SYMBOLS = {
    # =========================
    # 🇺🇸 NEW YORK (NYSE)
    # =========================
    "^DJI": "Dow Jones Industrial Average (DJIA)",
    "^NYA": "NYSE Composite Index",

    # =========================
    # 🇺🇸 CHICAGO / NASDAQ
    # =========================
    "^IXIC": "Nasdaq Composite Index",
    "^NDX": "Nasdaq 100 (NDX)",

    # =========================
    # 🇨🇦 TORONTO
    # =========================
    "^GSPTSE": "S&P/TSX Composite Index",
    "TX60.TS": "S&P/TSX 60 Index",

    # =========================
    # 🇬🇧 LONDON
    # =========================
    "^FTSE": "FTSE 100",
    "^FTMC": "FTSE 250",

    # =========================
    # 🇪🇺 EURONEXT
    # =========================
    "^FCHI": "CAC 40 (France)",
    "^AEX": "AEX (Netherlands)",
    "^BFX": "BEL 20 (Belgium)",

    # =========================
    # 🇩🇪 FRANKFURT
    # =========================
    "^GDAXI": "DAX 40 (Germany)",
    "^MDAXI": "MDAX (Germany Mid Caps)",

    # =========================
    # 🇨🇭 ZURICH
    # =========================
    "^SSMI": "SMI - Swiss Market Index",
    "^SSHI": "SPI - Swiss Performance Index",

    # =========================
    # 🇮🇳 INDIA
    # =========================
    "^BSESN": "SENSEX (India)",
    "^NSEI": "NIFTY 50 (India)",

    # =========================
    # 🇧🇷 BRAZIL - B3
    # =========================
    "^BVSP": "Ibovespa (IBOV)",
    "^IBX50": "IBrX 50",
    "BRAX11.SA": "iShares IBrX-Índice Brasil (IBrX-100) ETF (proxy do IBrX 100)",

    # =========================
    # 🇯🇵 JAPAN - TOKYO
    # =========================
    "^N225": "Nikkei 225",
    "1306.T": "NEXT FUNDS TOPIX ETF (proxy do TOPIX)",

    # =========================
    # 🇰🇷 SOUTH KOREA - SEOUL
    # =========================
    "^KS11": "KOSPI (South Korea)",
    "^KQ11": "KOSDAQ (South Korea)",

    # =========================
    # 🇨🇳 CHINA - SHANGHAI / SHENZHEN
    # =========================
    "000001.SS": "SSE Composite Index (Shanghai)",
    "000300.SS": "CSI 300 (Shanghai + Shenzhen)",
    "399001.SZ": "SZSE Component Index (Shenzhen)",
    "399006.SZ": "ChiNext Index (Shenzhen)",

    # =========================
    # 🇭🇰 HONG KONG
    # =========================
    "^HSI": "Hang Seng Index (HK50)",

    # =========================
    # 🇦🇺 AUSTRALIA - SYDNEY
    # =========================
    "^AXJO": "S&P/ASX 200",

    # =========================
    # 🇸🇬 SINGAPORE
    # =========================
    "^STI": "Straits Times Index (Singapore)",
}

LOOKBACK = "400d"
INTERVAL = "1d"
TRADING_DAYS = 252

# janelas (em pregões) para vol anualizada por janela
WINDOWS = {
    "weekly": 5,
    "monthly": 21,
    "quarterly": 63,
    "semiannual": 126,
}

OUT_JSON = "data/marketdata.json"
OUT_CSV = None  # ex.: "data/marketdata.csv"


def chunked(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i: i + n]


def _safe_float(x, ndigits=None):
    """Converte para float (com round) e troca NaN por None."""
    if x is None:
        return None
    try:
        if pd.isna(x):
            return None
        fx = float(x)
        if ndigits is not None:
            fx = float(round(fx, ndigits))
        return fx
    except Exception:
        return None


def _append_nulls(results, batch):
    """Se um batch falhar, adiciona linhas com null para não quebrar o pipeline."""
    for sym in batch:
        company_name = SYMBOLS.get(sym, sym)
        results.append(
            {
                "symbol": sym,
                "name": company_name,
                "price": None,
                "close_d1": None,
                "change_pts": None,
                "change_pct": None,
                "daily_7d": [],
                "vol_annual": None,
                "vol_semiannual": None,
                "vol_quarterly": None,
                "vol_monthly": None,
                "vol_weekly": None,
            }
        )


def _extract_close_df(df: pd.DataFrame) -> pd.DataFrame:
    """Extrai a matriz de fechamentos (Close) para 1 ou vários tickers."""
    if df is None or df.empty:
        return pd.DataFrame()

    if "Close" in df.columns:
        close = df["Close"]
        if isinstance(close, pd.Series):
            close = close.to_frame()
        return close

    if isinstance(df.columns, pd.MultiIndex):
        if "Close" in df.columns.get_level_values(0):
            close = df.xs("Close", axis=1, level=0, drop_level=True)
            if isinstance(close, pd.Series):
                close = close.to_frame()
            return close

    return pd.DataFrame()


def _ann_vol_from_logret_window(logret: pd.DataFrame, window: int) -> pd.Series:
    """
    Vol anualizada estimada usando apenas os últimos 'window' retornos diários.
    Retorna Series indexada pelos tickers.
    """
    if logret is None or logret.empty:
        return pd.Series(dtype="float64")

    tail = logret.tail(window)

    # Precisa de pelo menos 2 observações para std com ddof=1
    if len(tail.index) < 2:
        return pd.Series(index=logret.columns, dtype="float64")

    return tail.std(axis=0, ddof=1) * np.sqrt(TRADING_DAYS)


def _try_get_live_price(sym: str):
    """
    Tenta puxar o preço 'agora' via fast_info/info.
    Se falhar, retorna None.
    """
    try:
        t = yf.Ticker(sym)
        fi = getattr(t, "fast_info", None)
        if fi and "lastPrice" in fi and fi["lastPrice"] is not None:
            return fi["lastPrice"]
    except Exception:
        pass

    try:
        t = yf.Ticker(sym)
        info = getattr(t, "info", None) or {}
        p = info.get("regularMarketPrice", None)
        if p is not None:
            return p
    except Exception:
        pass

    return None


def _to_utc_day_index(idx) -> pd.DatetimeIndex:
    """
    Converte índice para UTC e normaliza para dia (00:00 UTC).
    - Se vier tz-naive, assume UTC (comum no yfinance).
    """
    idx = pd.to_datetime(idx)
    if getattr(idx, "tz", None) is None:
        idx = idx.tz_localize("UTC")
    else:
        idx = idx.tz_convert("UTC")
    return idx.normalize()


def _clean_close_index(close: pd.DataFrame) -> pd.DataFrame:
    """
    Corrige o BUG do 'dia que não existe' / 'dia adiantado':
    - padroniza índice em UTC-day
    - remove datas duplicadas (keep=last)
    - remove qualquer linha no futuro (em relação ao hoje UTC)
    """
    if close is None or close.empty:
        return pd.DataFrame()

    close2 = close.copy()
    close2.index = _to_utc_day_index(close2.index)
    close2 = close2[~close2.index.duplicated(keep="last")]

    today_utc = pd.Timestamp.utcnow().normalize()
    close2 = close2.loc[close2.index <= today_utc]
    return close2


def main():
    os.makedirs("data", exist_ok=True)
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    results = []

    for batch in chunked(list(SYMBOLS.keys()), 100):
        try:
            df = yf.download(
                tickers=batch,
                period=LOOKBACK,
                interval=INTERVAL,
                auto_adjust=False,
                progress=False,
                threads=True,
                group_by="column",
            )
        except Exception as e:
            print(f"[WARN] Batch falhou (exception): {e}")
            _append_nulls(results, batch)
            continue

        close = _extract_close_df(df)
        close = _clean_close_index(close)

        if close.empty or len(close.index) == 0:
            print("[WARN] Batch retornou vazio/sem Close após limpeza. Registrando nulls.")
            _append_nulls(results, batch)
            continue

        # Garante colunas para todos os símbolos do batch
        for sym in batch:
            if sym not in close.columns:
                close[sym] = np.nan
        close = close[batch]

        # Log-retornos diários (vols com base em close real; sem ffill para não 'inventar' pregão)
        logret = np.log(close / close.shift(1))

        # Vol anualizada usando todo o período
        vol_annual = logret.std(axis=0, ddof=1) * np.sqrt(TRADING_DAYS)

        # Vol anualizada por janelas
        vol_weekly = _ann_vol_from_logret_window(logret, WINDOWS["weekly"])
        vol_monthly = _ann_vol_from_logret_window(logret, WINDOWS["monthly"])
        vol_quarterly = _ann_vol_from_logret_window(logret, WINDOWS["quarterly"])
        vol_semiannual = _ann_vol_from_logret_window(logret, WINDOWS["semiannual"])

        for sym in batch:
            company_name = SYMBOLS.get(sym, sym)

            s = close[sym].dropna()

            if s.empty:
                results.append(
                    {
                        "symbol": sym,
                        "name": company_name,
                        "price": None,
                        "close_d1": None,
                        "change_pts": None,
                        "change_pct": None,
                        "daily_7d": [],
                        "vol_annual": None,
                        "vol_semiannual": None,
                        "vol_quarterly": None,
                        "vol_monthly": None,
                        "vol_weekly": None,
                    }
                )
                continue

            # last_close_daily = último fechamento válido (último pregão real)
            last_close_daily = s.iloc[-1] if len(s) >= 1 else np.nan

            # close_d1 = fechamento do pregão anterior (penúltimo close)
            close_d1 = s.iloc[-2] if len(s) >= 2 else np.nan

            # daily_7d = últimos 7 fechamentos (UTC day)
            tail7 = s.tail(7)
            daily_7d = [
                {"date": idx.strftime("%Y-%m-%d"), "close": _safe_float(val, 6)}
                for idx, val in tail7.items()
            ]

            # preço "agora" (intraday) se possível; senão, usa o último close diário
            live_price = _try_get_live_price(sym)
            price = live_price if live_price is not None else last_close_daily

            # >>> REGRA QUE VOCÊ PEDIU:
            # Se NÃO tiver live_price, mantém a diferença D1-D2 (último pregão vs pregão anterior).
            ref_price_for_change = live_price if live_price is not None else last_close_daily

            if (not pd.isna(ref_price_for_change)) and (not pd.isna(close_d1)) and float(close_d1) != 0.0:
                change_pts = float(ref_price_for_change) - float(close_d1)
                change_pct = change_pts / float(close_d1)
            else:
                change_pts = np.nan
                change_pct = np.nan

            vA = vol_annual.get(sym, np.nan)
            vW = vol_weekly.get(sym, np.nan)
            vM = vol_monthly.get(sym, np.nan)
            vQ = vol_quarterly.get(sym, np.nan)
            vS = vol_semiannual.get(sym, np.nan)

            results.append(
                {
                    "symbol": sym,
                    "name": company_name,

                    # preço atual (ou fallback: último close diário)
                    "price": _safe_float(price, 6),

                    # fechamento do pregão anterior e variação
                    "close_d1": _safe_float(close_d1, 6),
                    "change_pts": _safe_float(change_pts, 6),
                    "change_pct": _safe_float(change_pct, 8),  # decimal (ex: 0.0042 = +0.42%)

                    # fechamentos dos últimos 7 pregões (datas em UTC)
                    "daily_7d": daily_7d,

                    # vols
                    "vol_annual": _safe_float(vA, 8),
                    "vol_semiannual": _safe_float(vS, 8),
                    "vol_quarterly": _safe_float(vQ, 8),
                    "vol_monthly": _safe_float(vM, 8),
                    "vol_weekly": _safe_float(vW, 8),
                }
            )

    payload = {
        "generated_at_utc": now,
        "source": "yfinance",
        "interval": INTERVAL,
        "lookback": LOOKBACK,
        "trading_days": TRADING_DAYS,
        "count": len(results),
        "data": results,
    }

    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    if OUT_CSV:
        pd.DataFrame(results).to_csv(OUT_CSV, index=False, encoding="utf-8")

    ok_prices = sum(1 for r in results if r["price"] is not None)
    ok_close_d1 = sum(1 for r in results if r["close_d1"] is not None)
    ok_change = sum(1 for r in results if r["change_pct"] is not None)
    ok_volA = sum(1 for r in results if r["vol_annual"] is not None)

    print(f"OK: atualizado {OUT_JSON} com {len(results)} tickers.")
    print(f"   Preços OK: {ok_prices}")
    print(f"   Close D-1 OK: {ok_close_d1}")
    print(f"   Variação OK: {ok_change}")
    print(f"   Vol anual OK: {ok_volA}")


if __name__ == "__main__":
    main()
