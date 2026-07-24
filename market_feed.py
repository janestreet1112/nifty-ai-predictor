from SmartApi import SmartConnect
import pyotp
import pandas as pd
from datetime import datetime, timedelta
import yfinance as yf

def get_live_nifty_candles(api_key=None, client_id=None, pin=None, totp_secret=None):
    """Fetches real intraday 15-min candles for NIFTY 50 via SmartAPI with yfinance fallback."""
    
    # --- Option 1: Try Angel One SmartAPI ---
    if api_key and client_id and pin and totp_secret:
        try:
            api_key = str(api_key).strip()
            client_id = str(client_id).strip()
            pin = str(pin).strip()
            totp_secret = str(totp_secret).strip().replace(" ", "")

            smart_api = SmartConnect(api_key=api_key)
            totp = pyotp.TOTP(totp_secret).now()
            session = smart_api.generateSession(client_id, pin, totp)

            if session.get('status'):
                # Request 5 days of data up to current timestamp
                from_date = (datetime.now() - timedelta(days=5)).strftime("%Y-%m-%d 09:15")
                to_date = datetime.now().strftime("%Y-%m-%d %H:%M")

                historic_param = {
                    "exchange": "NSE",
                    "symboltoken": "99926000", # NIFTY 50 Index Token
                    "interval": "FIFTEEN_MINUTE",
                    "fromdate": from_date,
                    "todate": to_date
                }

                response = smart_api.getCandleData(historic_param)
                if response.get('status') and response.get('data'):
                    data_list = response['data']
                    if data_list:
                        df = pd.DataFrame(data_list, columns=['time', 'open', 'high', 'low', 'close', 'volume'])
                        df['time'] = pd.to_datetime(df['time'])
                        for col in ['open', 'high', 'low', 'close', 'volume']:
                            df[col] = df[col].astype(float)
                        df = df[(df['time'].dt.hour >= 9) & (df['time'].dt.hour <= 15)]
                        return df.tail(60)
        except Exception as e:
            print("❌ SmartAPI Fetch Exception:", e)

    # --- Option 2: Real-time Live Fallback via Yahoo Finance (^NSEI) ---
    try:
        print("⚡ Fetching live NIFTY 50 data from yfinance...")
        ticker = yf.Ticker("^NSEI")
        df_yf = ticker.history(period="5d", interval="15m")
        if not df_yf.empty:
            df_yf = df_yf.reset_index()
            df_yf.rename(columns={
                'Datetime': 'time',
                'Open': 'open',
                'High': 'high',
                'Low': 'low',
                'Close': 'close',
                'Volume': 'volume'
            }, inplace=True)
            
            # Ensure proper timezone handling & formatting
            df_yf['time'] = pd.to_datetime(df_yf['time']).dt.tz_localize(None)
            df = df_yf[['time', 'open', 'high', 'low', 'close', 'volume']].copy()
            df = df[(df['time'].dt.hour >= 9) & (df['time'].dt.hour <= 15)]
            return df.tail(60)
    except Exception as e:
        print("❌ yfinance Fetch Exception:", e)

    return None
