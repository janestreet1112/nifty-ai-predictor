from SmartApi import SmartConnect
import pyotp
import pandas as pd
from datetime import datetime, timedelta

def get_live_nifty_candles(api_key, client_id, pin, totp_secret):
    """
    Establishes SmartAPI session and fetches 15-minute historical/live candles for NIFTY 50.
    """
    try:
        # Initialize SmartAPI
        smart_api = SmartConnect(api_key=api_key)
        totp = pyotp.TOTP(totp_secret).now()
        data = smart_api.generateSession(client_id, pin, totp)
        
        if not data.get('status'):
            print("Auth Error:", data.get('message'))
            return None

        # NIFTY 50 Index Token = "99926000", Exchange = "NSE"
        from_date = (datetime.now() - timedelta(days=5)).strftime("%Y-%m-%d 09:15")
        to_date = datetime.now().strftime("%Y-%m-%d %H:%M")

        historic_param = {
            "exchange": "NSE",
            "symboltoken": "99926000",
            "interval": "FIFTEEN_MINUTE",
            "fromdate": from_date,
            "todate": to_date
        }

        candle_data = smart_api.getCandleData(historic_param)
        
        if candle_data.get('status') and candle_data.get('data'):
            df = pd.DataFrame(candle_data['data'], columns=['time', 'open', 'high', 'low', 'close', 'volume'])
            df['time'] = pd.to_datetime(df['time'])
            return df
        return None

    except Exception as e:
        print(f"SmartAPI Feed Error: {e}")
        return None
