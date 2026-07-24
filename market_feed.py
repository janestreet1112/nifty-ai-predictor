from SmartApi import SmartConnect
import pyotp
import pandas as pd
from datetime import datetime, timedelta

def get_live_nifty_candles(api_key, client_id, pin, totp_secret):
    """Fetches real 15-minute intraday candles from Angel One and filters session gaps."""
    try:
        api_key = str(api_key).strip()
        client_id = str(client_id).strip()
        pin = str(pin).strip()
        totp_secret = str(totp_secret).strip().replace(" ", "")

        smart_api = SmartConnect(api_key=api_key)
        totp = pyotp.TOTP(totp_secret).now()
        session = smart_api.generateSession(client_id, pin, totp)
        
        if not session.get('status'):
            print("❌ SmartAPI Auth Failed:", session.get('message'))
            return None

        # Fetch candles for last 3 trading days
        from_date = (datetime.now() - timedelta(days=4)).strftime("%Y-%m-%d 09:15")
        to_date = datetime.now().strftime("%Y-%m-%d %H:%M")

        historic_param = {
            "exchange": "NSE",
            "symboltoken": "99926000", # NIFTY 50 Spot Token
            "interval": "FIFTEEN_MINUTE",
            "fromdate": from_date,
            "todate": to_date
        }

        response = smart_api.getCandleData(historic_param)
        
        if response.get('status') and response.get('data'):
            data_list = response['data']
            if not data_list:
                return None

            df = pd.DataFrame(data_list, columns=['time', 'open', 'high', 'low', 'close', 'volume'])
            df['time'] = pd.to_datetime(df['time'])
            
            # Numeric conversion
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df[col] = df[col].astype(float)
            
            # Filter trading hours only (09:15 to 15:30) to prevent gap distortions
            df = df[(df['time'].dt.hour >= 9) & (df['time'].dt.hour <= 15)]
            df = df.tail(60) # Keep latest 60 active intraday candles
            
            return df
        return None

    except Exception as e:
        print("❌ Error fetching live ticks:", e)
        return None
