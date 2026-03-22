import pandas as pd
import numpy as np
import requests
import matplotlib.pyplot as plt
from datetime import datetime
import os
import io
import sys
import warnings
import matplotlib
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.image import MIMEImage

# 环境配置
matplotlib.use('Agg')
warnings.filterwarnings("ignore")

CONFIG = {
    'initial_capital': 10000,
    'buy_threshold': -0.03,      # 乖离率低于 -3% 买入
    'sell_threshold': 0.07,      # 乖离率高于 7% 卖出
    'switch_threshold': 0.015,   # 换仓缓冲：新目标比旧目标低 1.5% 才换
    'stop_loss_threshold': -0.1,  
    'cooldown_days': 10,         
    'ma_period': 120,
    'start_date': '2023-01-01',
    'data_dir': './fund_data',
    'commission_rate': 0.0001,   # 实战费率：万一
    'money_fund_yield': 0.02,    # 货基年化收益模拟
}

FUND_POOL = {
    "515080": "CS_Dividend", "510880": "SSE_Dividend", "515180": "E_Fund_Div",
    "513530": "HK_Dividend", "563020": "Low_Vol_Div", "510720": "SOE_Dividend",
    "159209": "Quality_Div"
}
BENCHMARK = {"code": "510300", "name": "HS300"}

os.makedirs(CONFIG['data_dir'], exist_ok=True)

def get_fund_data(code):
    url = f"https://fundmobapi.eastmoney.com/FundMNewApi/FundMNHisNetList?FCODE={code}&pageSize=4000&deviceid=1"
    try:
        r = requests.get(url, timeout=10).json()
        df = pd.DataFrame([{'date': i['FSRQ'], 'close': float(i['DWJZ'])} for i in r['Datas']])
        df['date'] = pd.to_datetime(df['date'])
        return df.sort_values('date')
    except: return pd.DataFrame()

def prepare_data():
    all_dfs = []
    for code in FUND_POOL.keys():
        df = get_fund_data(code).rename(columns={'close': f'close_{code}'})
        all_dfs.append(df.set_index('date'))
    bench = get_fund_data(BENCHMARK['code']).rename(columns={'close': 'close_bench'}).set_index('date')
    all_dfs.append(bench)
    merged = pd.concat(all_dfs, axis=1).ffill().dropna()
    for code in FUND_POOL.keys():
        ma = merged[f'close_{code}'].rolling(CONFIG['ma_period']).mean()
        merged[f'dev_{code}'] = merged[f'close_{code}'] / ma - 1
    return merged[merged.index >= CONFIG['start_date']].reset_index()

def run_strategy(df):
    cap = CONFIG['initial_capital']
    hold_code, hold_price, hold_qty = None, 0, 0
    money_val = 0
    trades, history = [], []
    daily_m_rate = (1 + CONFIG['money_fund_yield'])**(1/252)

    for i, row in df.iterrows():
        date = row['date']
        money_val *= daily_m_rate
        devs = {c: row[f'dev_{c}'] for c in FUND_POOL.keys()}
        
        # 1. 卖出/止损逻辑
        if hold_code:
            cur_p = row[f'close_{hold_code}']
            if devs[hold_code] > CONFIG['sell_threshold'] or (cur_p/hold_price-1) < CONFIG['stop_loss_threshold']:
                money_val = hold_qty * cur_p * (1 - CONFIG['commission_rate'])
                trades.append({'date': date, 'action': 'SELL', 'code': hold_code, 'price': cur_p})
                hold_code, hold_qty = None, 0

        # 2. 买入/换仓逻辑
        candidates = {c: d for c, d in devs.items() if d < CONFIG['buy_threshold']}
        if candidates:
            best = min(candidates, key=candidates.get)
            if not hold_code or devs[best] < (devs[hold_code] - CONFIG['switch_threshold']):
                if hold_code:
                    money_val = hold_qty * row[f'close_{hold_code}'] * (1 - CONFIG['commission_rate'])
                buy_cap = (money_val + cap) * (1 - CONFIG['commission_rate'])
                hold_code, hold_price = best, row[f'close_{best}']
                hold_qty = buy_cap / hold_price
                money_val, cap = 0, 0
                trades.append({'date': date, 'action': 'BUY/SWITCH', 'code': best, 'price': hold_price})

        total = cap + money_val + (hold_qty * row[f'close_{hold_code}'] if hold_code else 0)
        history.append({'date': date, 'value': total, 'holding': hold_code or 'CASH', 'bench_close': row['close_bench']})
    
    return pd.DataFrame(history), trades

def plot_report(hist):
    plt.figure(figsize=(10, 6))
    strategy_ret = (hist['value'] / hist['value'].iloc[0] - 1) * 100
    bench_ret = (hist['bench_close'] / hist['bench_close'].iloc[0] - 1) * 100
    plt.plot(hist['date'], strategy_ret, label='Dividend Strategy', color='red')
    plt.plot(hist['date'], bench_ret, label='Benchmark (HS300)', color='blue', alpha=0.5)
    plt.title('Strategy Cumulative Return (%)')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    img_path = os.path.join(CONFIG['data_dir'], 'report.png')
    plt.savefig(img_path)
    plt.close()
    return img_path

def send_email(subject, content, img_path=None):
    sender = os.environ.get('EMAIL_SENDER')
    pwd = os.environ.get('EMAIL_PASSWORD')
    if not sender or not pwd:
        print("❌ Error: Missing Email Credentials.")
        return

    msg = MIMEMultipart()
    msg['Subject'] = subject
    msg['From'] = sender
    msg['To'] = sender
    msg.attach(MIMEText(content, 'plain', 'utf-8'))

    if img_path and os.path.exists(img_path):
        with open(img_path, 'rb') as f:
            img = MIMEImage(f.read())
            img.add_header('Content-Disposition', 'attachment', filename="report.png")
            msg.attach(img)

    try:
        smtp_server = "smtp.qq.com" if "qq.com" in sender else "smtp.163.com" if "163.com" in sender else "smtp.gmail.com"
        with smtplib.SMTP_SSL(smtp_server, 465) as server:
            server.login(sender, pwd)
            server.sendmail(sender, [sender], msg.as_string())
        print("✅ Email Sent Successfully!")
    except Exception as e:
        print(f"❌ Failed to send email: {e}")

def main():
    log_capture = io.StringIO()
    sys.stdout = log_capture
    try:
        data = prepare_data()
        hist, trades = run_strategy(data)
        img_path = plot_report(hist)
        
        last_day = hist.iloc[-1]
        print(f"--- Strategy Summary ---")
        print(f"Update Date: {last_day['date'].strftime('%Y-%m-%d')}")
        print(f"Current Holding: {last_day['holding']}")
        print(f"Total Return: {((last_day['value']/CONFIG['initial_capital'])-1)*100:.2f}%")
        
        today_trade = [t for t in trades if t['date'].date() == datetime.now().date()]
        subject = f"Dividend Report - {datetime.now().strftime('%Y-%m-%d')}"
        trade_msg = ""
        if today_trade:
            subject = "⚠️【交易提醒】红利策略有变动"
            trade_msg = "🚨 Action Required:\n" + "\n".join([f"{t['action']} {t['code']} @ {t['price']}" for t in today_trade])
        
        report_text = trade_msg + "\n\nLog Details:\n" + log_capture.getvalue()
        sys.stdout = sys.__stdout__
        print(report_text)
        
        send_email(subject, report_text, img_path)
        with open("log.txt", "w", encoding='utf-8') as f: f.write(report_text)
            
    except Exception as e:
        sys.stdout = sys.__stdout__
        print(f"Main Error: {e}")

if __name__ == "__main__":
    main()
