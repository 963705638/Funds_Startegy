import pandas as pd
import numpy as np
import requests
import matplotlib.pyplot as plt
from datetime import datetime
import os
import json
import warnings
import matplotlib
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.image import MIMEImage
import io
import sys

# 设置为无界面模式，适配 GitHub Actions
matplotlib.use('Agg')
warnings.filterwarnings("ignore")

# ========== 配置参数 ==========
CONFIG = {
    'initial_capital': 10000,
    'buy_threshold': -0.03,
    'sell_threshold': 0.07,
    'switch_threshold': 0.02,
    'stop_loss_threshold': -0.1,  
    'cooldown_days': 15,          
    'ma_period': 120,
    'start_date': '2023-01-01',  # 已修改为过去日期以便回测
    'data_dir': './fund_data',
    'use_money_fund': True,
    'money_fund_yield_annual': 0.022,
}

FUND_POOL = {
    "515080": "CS_Dividend_ETF",
    "510880": "SSE_Dividend_ETF",
    "515180": "E_Fund_Dividend_ETF",
    "513530": "HK_Dividend_ETF",
    "563020": "Low_Vol_Dividend_ETF",
    "510720": "SOE_Dividend_ETF",
    "159209": "Quality_Dividend_ETF",
}

MONEY_FUND = {"code": "511880", "name": "Money_Market_Fund"}
BENCHMARK = {"code": "510300", "name": "HS300_ETF"}

os.makedirs(CONFIG['data_dir'], exist_ok=True)

# ========== 1. 数据获取 ==========
def get_fund_k_history(fund_code: str, pz: int = 4000) -> pd.DataFrame:
    headers = {
        'User-Agent': 'Mozilla/5.0 (iPhone; CPU iPhone OS 13_2_3 like Mac OS X)',
        'Content-Type': 'application/x-www-form-urlencoded',
        'Host': 'fundmobapi.eastmoney.com',
    }
    data = {
        'FCODE': fund_code,
        'pageSize': str(pz),
        'deviceid': '1',
    }
    url = 'https://fundmobapi.eastmoney.com/FundMNewApi/FundMNHisNetList'
    try:
        resp = requests.get(url, headers=headers, params=data, timeout=10).json()
        if not resp or not resp.get('Datas'):
            return pd.DataFrame()
        rows = []
        for item in resp['Datas']:
            rows.append({
                'date': item['FSRQ'],
                'close': item['DWJZ'],
            })
        df = pd.DataFrame(rows)
        df['close'] = pd.to_numeric(df['close'], errors='coerce')
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        df = df.sort_values('date').reset_index(drop=True)
        return df
    except Exception as e:
        print(f"Error fetching {fund_code}: {e}")
        return pd.DataFrame()

def fetch_all_fund_data():
    print("=" * 60)
    print(f"Fetching Data... {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    fund_data = {}
    for code, name in FUND_POOL.items():
        df = get_fund_k_history(code)
        if len(df) > 0:
            df_clean = df[['date', 'close']].dropna()
            fund_data[code] = df_clean
            print(f"✓ {code} ({name}): {len(df_clean)} records")
    
    benchmark_df = get_fund_k_history(BENCHMARK['code'])
    return fund_data, benchmark_df

def prepare_merged_data(fund_data, benchmark_df):
    merged_df = None
    for code, df in fund_data.items():
        df_copy = df.rename(columns={'close': f'close_{code}'})
        merged_df = df_copy if merged_df is None else pd.merge(merged_df, df_copy, on='date', how='outer')

    bench_copy = benchmark_df.rename(columns={'close': 'close_benchmark'})
    merged_df = pd.merge(merged_df, bench_copy, on='date', how='outer')
    merged_df = merged_df.sort_values('date').ffill().bfill()

    for code in fund_data.keys():
        col = f'close_{code}'
        ma_col = f'MA{CONFIG["ma_period"]}_{code}'
        merged_df[ma_col] = merged_df[col].rolling(window=CONFIG['ma_period']).mean()
        merged_df[f'deviation_{code}'] = merged_df[col] / merged_df[ma_col] - 1

    merged_df = merged_df[merged_df['date'] >= CONFIG['start_date']].dropna().reset_index(drop=True)
    return merged_df

# ========== 2. 回测逻辑 ==========
def run_backtest(merged_df, fund_data):
    capital = CONFIG['initial_capital']
    position = {code: 0.0 for code in fund_data.keys()}
    money_position = 0.0
    holding_code, holding_cost = None, None
    cooldown_counter = 0

    portfolio_values, trade_log, daily_status = [], [], []

    # 获取货基数据
    raw_money = get_fund_k_history(MONEY_FUND['code'])
    mdf = raw_money.rename(columns={'close': 'close_money', 'date': 'date'})[['date', 'close_money']]
    merged_df = pd.merge(merged_df, mdf, on='date', how='left').ffill().bfill()

    for idx, row in merged_df.iterrows():
        date = row['date']
        deviations = {code: row[f'deviation_{code}'] for code in fund_data.keys()}
        if cooldown_counter > 0: cooldown_counter -= 1

        # 卖出/止损判断
        if holding_code:
            cur_dev = deviations[holding_code]
            cur_price = row[f'close_{holding_code}']
            cur_ret = (cur_price / holding_cost) - 1
            
            is_tp = cur_dev > CONFIG['sell_threshold']
            is_sl = cur_ret <= CONFIG['stop_loss_threshold']

            if is_tp or is_sl:
                capital = position[holding_code] * cur_price
                trade_log.append({'date': date, 'action': 'SELL(SL)' if is_sl else 'SELL(TP)', 'code': holding_code, 'price': cur_price, 'value': capital})
                if is_sl: cooldown_counter = CONFIG['cooldown_days']
                position[holding_code] = 0
                holding_code = None
                # 转入货基
                money_position = capital / row['close_money']
                capital = 0

        # 买入/换仓判断
        if cooldown_counter == 0:
            candidates = {c: d for c, d in deviations.items() if d < CONFIG['buy_threshold']}
            if candidates:
                best_code = min(candidates, key=candidates.get)
                if holding_code is None:
                    if money_position > 0: capital = money_position * row['close_money']
                    buy_price = row[f'close_{best_code}']
                    position[best_code] = capital / buy_price
                    holding_code, holding_cost = best_code, buy_price
                    trade_log.append({'date': date, 'action': 'BUY', 'code': best_code, 'price': buy_price, 'value': capital})
                    money_position, capital = 0, 0

        # 记录每日状态
        pv = capital + (position[holding_code] * row[f'close_{holding_code}'] if holding_code else 0) + (money_position * row['close_money'])
        portfolio_values.append({'date': date, 'portfolio_value': pv, 'holding_name': FUND_POOL.get(holding_code, 'CASH/MONEY')})
        daily_status.append({'date': date, 'portfolio_value': pv, 'benchmark_close': row['close_benchmark']})

    return portfolio_values, trade_log, daily_status, merged_df

# ========== 3. 绘图 (全英文) ==========
def plot_charts_eng(portfolio_df, daily_df, fund_data):
    plt.style.use('ggplot')
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

    # Strategy Return
    p_ret = (portfolio_df['portfolio_value'] / CONFIG['initial_capital'] - 1) * 100
    b_ret = (daily_df['benchmark_close'] / daily_df['benchmark_close'].iloc[0] - 1) * 100
    
    ax1.plot(portfolio_df['date'], p_ret, label='Strategy', color='red', linewidth=2)
    ax1.plot(daily_df['date'], b_ret, label='Benchmark (HS300)', color='blue', alpha=0.7)
    ax1.set_title('Strategy vs Benchmark Cumulative Return (%)')
    ax1.legend()
    ax1.grid(True)

    # Net Value
    ax2.plot(portfolio_df['date'], portfolio_df['portfolio_value'], color='green')
    ax2.set_title('Portfolio Net Value (Capital)')
    ax2.grid(True)

    plt.tight_layout()
    chart_path = f"{CONFIG['data_dir']}/report_chart.png"
    plt.savefig(chart_path)
    return chart_path

# ========== 4. 邮件发送 ==========
def send_email(content, chart_path):
    sender = os.environ.get('EMAIL_SENDER')
    pwd = os.environ.get('EMAIL_PASSWORD')
    if not sender or not pwd:
        print("Missing Email Credentials. Skip Sending.")
        return

    msg = MIMEMultipart()
    msg['Subject'] = f"Strategy Report - {datetime.now().strftime('%Y-%m-%d')}"
    msg['From'] = sender
    msg['To'] = sender

    msg.attach(MIMEText(content, 'plain', 'utf-8'))
    if os.path.exists(chart_path):
        with open(chart_path, 'rb') as f:
            img = MIMEImage(f.read())
            img.add_header('Content-Disposition', 'attachment', filename="chart.png")
            msg.attach(img)

    try:
        # 默认使用 Gmail。如果是 QQ 邮箱改用 smtp.qq.com
        smtp_server = "smtp.gmail.com" if "gmail" in sender else "smtp.qq.com"
        with smtplib.SMTP_SSL(smtp_server, 465) as server:
            server.login(sender, pwd)
            server.sendmail(sender, [sender], msg.as_string())
        print("Email Sent Successfully!")
    except Exception as e:
        print(f"Failed to send email: {e}")

# ========== 5. 主函数 ==========
def main():
    # 捕获打印内容
    output_capture = io.StringIO()
    sys.stdout = output_capture

    try:
        fund_data, benchmark_df = fetch_all_fund_data()
        merged_df = prepare_merged_data(fund_data, benchmark_df)
        pv_list, trades, daily_list, final_merged = run_backtest(merged_df, fund_data)
        
        pdf = pd.DataFrame(pv_list)
        ddf = pd.DataFrame(daily_list)
        
        # 打印简单报告
        print("\n--- PERFORMANCE SUMMARY ---")
        final_val = pdf['portfolio_value'].iloc[-1]
        total_ret = (final_val / CONFIG['initial_capital'] - 1) * 100
        print(f"Total Return: {total_ret:.2f}%")
        print(f"Current Holding: {pdf['holding_name'].iloc[-1]}")
        print(f"Last Update: {pdf['date'].iloc[-1].strftime('%Y-%m-%d')}")
        
        chart_file = plot_charts_eng(pdf, ddf, fund_data)
        
    finally:
        report_text = output_capture.getvalue()
        sys.stdout = sys.__stdout__
        print(report_text)

    send_email(report_text, chart_file)

if __name__ == "__main__":
    main()
