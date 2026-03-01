"""
Coinbase Advanced Trade API Client
Handles authentication, market data fetching, and order execution.
"""

import time
import hmac
import hashlib
import json
import logging
import requests
import pandas as pd
from datetime import datetime, timezone
from typing import Optional

import config

logger = logging.getLogger("TradeBot.Client")

BASE_URL = "https://api.coinbase.com"


class CoinbaseClient:
    def __init__(self):
        self.api_key    = config.API_KEY
        self.api_secret = config.API_SECRET
        self.session    = requests.Session()
        logger.info("CoinbaseClient initialised.")

    # ── Authentication ─────────────────────────────────────────────────────────
    def _headers(self, method: str, path: str, body: str = "") -> dict:
        timestamp = str(int(time.time()))
        message   = timestamp + method.upper() + path + body
        signature = hmac.new(
            self.api_secret.encode("utf-8"),
            message.encode("utf-8"),
            digestmod=hashlib.sha256
        ).hexdigest()
        return {
            "CB-ACCESS-KEY":       self.api_key,
            "CB-ACCESS-SIGN":      signature,
            "CB-ACCESS-TIMESTAMP": timestamp,
            "Content-Type":        "application/json",
        }

    def _get(self, path: str, params: dict = None) -> dict:
        url     = BASE_URL + path
        headers = self._headers("GET", path)
        r = self.session.get(url, headers=headers, params=params, timeout=15)
        r.raise_for_status()
        return r.json()

    def _post(self, path: str, payload: dict) -> dict:
        body    = json.dumps(payload)
        url     = BASE_URL + path
        headers = self._headers("POST", path, body)
        r = self.session.post(url, headers=headers, data=body, timeout=15)
        r.raise_for_status()
        return r.json()

    # ── Account Info ───────────────────────────────────────────────────────────
    def get_accounts(self) -> list:
        data = self._get("/api/v3/brokerage/accounts")
        return data.get("accounts", [])

    def get_portfolio_value(self) -> float:
        """Returns total portfolio USD value."""
        accounts = self.get_accounts()
        total = 0.0
        for acct in accounts:
            if acct.get("currency") == "USD":
                total += float(acct.get("available_balance", {}).get("value", 0))
        return total

    def get_usd_balance(self) -> float:
        for acct in self.get_accounts():
            if acct.get("currency") == "USD":
                return float(acct["available_balance"]["value"])
        return 0.0

    # ── Market Data ────────────────────────────────────────────────────────────
    def get_candles(self, product_id: str, granularity: str, limit: int = 300) -> pd.DataFrame:
        """
        Fetch OHLCV candles and return as a DataFrame sorted oldest→newest.
        granularity: ONE_MINUTE | FIVE_MINUTE | FIFTEEN_MINUTE | THIRTY_MINUTE
                     ONE_HOUR | TWO_HOUR | SIX_HOUR | ONE_DAY
        """
        path   = f"/api/v3/brokerage/products/{product_id}/candles"
        end    = int(time.time())
        gran_seconds = {
            "ONE_MINUTE": 60, "FIVE_MINUTE": 300, "FIFTEEN_MINUTE": 900,
            "THIRTY_MINUTE": 1800, "ONE_HOUR": 3600, "TWO_HOUR": 7200,
            "SIX_HOUR": 21600, "ONE_DAY": 86400,
        }
        start = end - gran_seconds[granularity] * limit
        params = {"start": start, "end": end, "granularity": granularity}
        data   = self._get(path, params=params)
        candles = data.get("candles", [])
        if not candles:
            return pd.DataFrame()
        df = pd.DataFrame(candles, columns=["start","low","high","open","close","volume"])
        df["start"]  = pd.to_numeric(df["start"])
        df["open"]   = pd.to_numeric(df["open"])
        df["high"]   = pd.to_numeric(df["high"])
        df["low"]    = pd.to_numeric(df["low"])
        df["close"]  = pd.to_numeric(df["close"])
        df["volume"] = pd.to_numeric(df["volume"])
        df = df.sort_values("start").reset_index(drop=True)
        df["datetime"] = pd.to_datetime(df["start"], unit="s", utc=True)
        return df

    def get_ticker(self, product_id: str) -> float:
        path = f"/api/v3/brokerage/products/{product_id}"
        data = self._get(path)
        return float(data.get("price", 0))

    def get_open_orders(self, product_id: Optional[str] = None) -> list:
        path   = "/api/v3/brokerage/orders/historical/batch"
        params = {"order_status": "OPEN"}
        if product_id:
            params["product_id"] = product_id
        return self._get(path, params).get("orders", [])

    # ── Order Execution ────────────────────────────────────────────────────────
    def place_market_buy(self, product_id: str, usd_amount: float) -> dict:
        if config.DRY_RUN:
            logger.info(f"[DRY RUN] BUY {product_id} ${usd_amount:.2f}")
            return {"dry_run": True, "side": "BUY", "product_id": product_id, "amount": usd_amount}
        payload = {
            "client_order_id": f"bot_{int(time.time())}",
            "product_id": product_id,
            "side": "BUY",
            "order_configuration": {
                "market_market_ioc": {"quote_size": f"{usd_amount:.2f}"}
            }
        }
        result = self._post("/api/v3/brokerage/orders", payload)
        logger.info(f"BUY order placed: {result}")
        return result

    def place_market_sell(self, product_id: str, base_amount: float) -> dict:
        if config.DRY_RUN:
            logger.info(f"[DRY RUN] SELL {product_id} {base_amount:.8f}")
            return {"dry_run": True, "side": "SELL", "product_id": product_id, "amount": base_amount}
        payload = {
            "client_order_id": f"bot_{int(time.time())}",
            "product_id": product_id,
            "side": "SELL",
            "order_configuration": {
                "market_market_ioc": {"base_size": f"{base_amount:.8f}"}
            }
        }
        result = self._post("/api/v3/brokerage/orders", payload)
        logger.info(f"SELL order placed: {result}")
        return result

    def place_limit_buy(self, product_id: str, price: float, usd_amount: float) -> dict:
        base_size = usd_amount / price
        if config.DRY_RUN:
            logger.info(f"[DRY RUN] LIMIT BUY {product_id} @ ${price:.2f} qty={base_size:.6f}")
            return {"dry_run": True}
        payload = {
            "client_order_id": f"bot_{int(time.time())}",
            "product_id": product_id,
            "side": "BUY",
            "order_configuration": {
                "limit_limit_gtc": {
                    "base_size":   f"{base_size:.6f}",
                    "limit_price": f"{price:.2f}",
                    "post_only":   False,
                }
            }
        }
        return self._post("/api/v3/brokerage/orders", payload)

    def cancel_order(self, order_id: str) -> dict:
        if config.DRY_RUN:
            logger.info(f"[DRY RUN] Cancel order {order_id}")
            return {}
        return self._post("/api/v3/brokerage/orders/batch_cancel", {"order_ids": [order_id]})
