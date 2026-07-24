from SmartApi import SmartConnect
import pyotp
import pandas as pd
from datetime import datetime, timedelta

def get_live_nifty_candles(api_key, client_id, pin, totp_secret):
    """
    Establishes an Angel One SmartAPI session using TOTP authentication
    and fetches real intraday OHLC candle data for the NIFTY 50 Index.
    """
    try:
        # Sanitize parameters (strip whitespaces/newlines)
        api_key = str(api_key).strip()
        client_id = str(client_id).strip()
        pin = str(pin).strip()
        totp_secret = str(totp_secret).strip().replace(" ", "")

        # 1. Initialize SmartConnect Instance
        smart_api = SmartConnect(api_key=api_key)
        
        # 2. Generate 2FA TOTP Code
        totp = pyotp.TOTP(totp_secret).now()

        # 3. Authenticate Session
        session = smart_api.generateSession(client_id, pin, totp)
        
        if not session.get('status'):
            print("❌ SmartAPI Session Generation Failed:", session.get('message'))
            return None

        # 4. Set Parameters for NIFTY 50 Spot Index (Token: 99926000)
        from_date = (datetime.now() - timedelta(days=5)).strftime("%Y-%m-%d 09:15")
        to_date = datetime.now().strftime("%Y-%m-%d %H:%M")

        historic_param = {
            "exchange": "NSE",
            "symboltoken": "99926000",
            "interval": "FIFTEEN_MINUTE",
            "fromdate": from_date,
            "todate": to_date
        }

        # 5. Fetch Candlestick Data
        response = smart_api.getCandleData(historic_param)
        
        if response.get('status') and response.get('data'):
            data_list = response['data']
            if not data_list:
                print("⚠️ SmartAPI returned empty candle list.")
                return None

            df = pd.DataFrame(
                data_list, 
                columns=['time', 'open', 'high', 'low', 'close', 'volume']
            )
            
            # Format types
            df['time'] = pd.to_datetime(df['time'])
            df['open'] = df['open'].astype(float)
            df['high'] = df['high'].astype(float)
            df['low'] = df['low'].astype(float)
            df['close'] = df['close'].astype(float)
            df['volume'] = df['volume'].astype(int)
            
            return df
        else:
            print("⚠️ SmartAPI Candle Fetch Error:", response.get('message'))
            return None

    except Exception as e:
        print("❌ Exception in market_feed.py:", e)
        return None
