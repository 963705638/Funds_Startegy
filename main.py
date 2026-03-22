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
    # 模拟真实手机浏览器的请求头
    headers = {
        'User-Agent': 'Mozilla/5.0 (iPhone; CPU iPhone OS 13_2_3 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/13.0.3 Mobile/15E148 Safari/604.1',
        'Referer': 'https://j5.dfcfw.com/',
        'Host': 'fundmobapi.eastmoney.com'
    }
    url = f"https://fundmobapi.eastmoney.com/FundMNewApi/FundMNHisNetList"
    params = {
        'FCODE': code,
        'pageSize': '4000',
        'deviceid': '1',
        'plat': 'Iphone',
        'product': 'EFund',
        'version': '6.2.9'
    }

    # 循环重试 3 次
    for attempt in range(3):
        try:
            r = requests.get(url, params=params, headers=headers, timeout=15)
            if r.status_code == 200:
                data = r.json()
                if data and data.get('Datas'):
                    df = pd.DataFrame([{'date': i['FSRQ'], 'close': float(i['DWJZ'])} for i in data['Datas']])
                    df['date'] = pd.to_datetime(df['date'])
                    print(f"✅ 成功获取数据: {code} ({len(df)} 条记录)")
                    return df.sort_values('date')
            print(f"⚠️ 第 {attempt+1} 次尝试获取 {code} 失败，状态码: {r.status_code}")
        except Exception as e:
            print(f"⚠️ 第 {attempt+1} 次尝试获取 {code} 出错: {e}")
    
    return pd.DataFrame()
    
def prepare_data():
    all_dfs = []
    # 1. 获取所有基金数据
    for code in FUND_POOL.keys():
        df = get_fund_data(code)
        if not df.empty:
            df = df.rename(columns={'close': f'close_{code}'})
            all_dfs.append(df.set_index('date'))
        else:
            print(f"⚠️ 警告: 无法获取基金 {code} 的数据，跳过该基金。")
    
    # 2. 获取基准数据
    bench = get_fund_data(BENCHMARK['code'])
    if not bench.empty:
        bench = bench.rename(columns={'close': 'close_bench'}).set_index('date')
        all_dfs.append(bench)
    else:
        raise Exception("❌ 严重错误: 无法获取基准(HS300)数据，回测无法继续。")
    
    # 3. 合并数据
    # 使用 inner join 确保日期对齐，或者 outer join 后填充
    merged = pd.concat(all_dfs, axis=1).ffill().dropna()
    
    # 4. 把日期从 Index 变回正常的列
    merged = merged.reset_index()
    
    # 5. 计算乖离率
    for code in FUND_POOL.keys():
        col_name = f'close_{code}'
        if col_name in merged.columns:
            ma = merged[col_name].rolling(CONFIG['ma_period']).mean()
            merged[f'dev_{code}'] = merged[col_name] / ma - 1
            
    # 确保最后返回的 df 包含 date 列
    if 'date' not in merged.columns:
        # 如果还是没有，尝试强制转换
        merged.index.name = 'date'
        merged = merged.reset_index()
        
    return merged[merged['date'] >= CONFIG['start_date']]

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
