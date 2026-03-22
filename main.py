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

matplotlib.use('Agg')
warnings.filterwarnings("ignore")

# ========== 配置参数 ==========
CONFIG = {
    'initial_capital': 10000,
    'buy_threshold': -0.03,
    'sell_threshold': 0.07,
    'switch_threshold': 0.015,   # 换仓缓冲：新目标比旧目标低 1.5% 才换
    'stop_loss_threshold': -0.1,  
    'cooldown_days': 15,          
    'ma_period': 120,
    'start_date': '2023-01-01',
    'data_dir': './fund_data',
    'commission_rate': 0.0001,   # 费率：万一 (0.01%)
    'money_fund_yield_annual': 0.02, 
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

# (get_fund_k_history, fetch_all_fund_data, prepare_merged_data 函数保持不变，此处略)
# [请保留你原代码中的这三个函数，仅在 prepare_merged_data 后开始修改]

def run_backtest(merged_df, fund_data):
    capital = CONFIG['initial_capital']
    position = {code: 0.0 for code in fund_data.keys()}
    money_position = 0.0 
    holding_code, holding_cost = None, None
    cooldown_counter = 0
    
    portfolio_values, trade_log, daily_status = [], [], []

    # 简单模拟货基价格（恒定增长）
    daily_money_rate = (1 + CONFIG['money_fund_yield_annual']) ** (1/252)
    current_money_price = 1.0

    for idx, row in merged_df.iterrows():
        date = row['date']
        current_money_price *= daily_money_rate
        deviations = {code: row[f'deviation_{code}'] for code in fund_data.keys()}
        if cooldown_counter > 0: cooldown_counter -= 1

        # 1. 卖出逻辑 (止盈/止损)
        if holding_code:
            cur_dev = deviations[holding_code]
            cur_price = row[f'close_{holding_code}']
            cur_ret = (cur_price / holding_cost) - 1
            
            is_tp = cur_dev > CONFIG['sell_threshold']
            is_sl = cur_ret <= CONFIG['stop_loss_threshold']

            if is_tp or is_sl:
                # 卖出金额，扣除费率
                sell_val = position[holding_code] * cur_price * (1 - CONFIG['commission_rate'])
                trade_log.append({'date': date, 'action': 'SELL(SL)' if is_sl else 'SELL(TP)', 'code': holding_code, 'price': cur_price, 'value': sell_val})
                
                if is_sl: cooldown_counter = CONFIG['cooldown_days']
                position[holding_code] = 0
                holding_code = None
                # 立即买入货基
                money_position = sell_val / current_money_price
                capital = 0

        # 2. 买入/换仓逻辑
        if cooldown_counter == 0:
            candidates = {c: d for c, d in deviations.items() if d < CONFIG['buy_threshold']}
            if candidates:
                best_code = min(candidates, key=candidates.get)
                
                # 确定是否需要买入或切仓
                need_action = False
                if holding_code is None:
                    need_action = True
                elif deviations[best_code] < (deviations[holding_code] - CONFIG['switch_threshold']):
                    need_action = True
                
                if need_action:
                    # 如果有旧持仓，先卖出
                    if holding_code:
                        sell_val = position[holding_code] * row[f'close_{holding_code}'] * (1 - CONFIG['commission_rate'])
                        position[holding_code] = 0
                        money_position = sell_val / current_money_price
                    
                    # 从货基/现金池买入新目标
                    buy_capital = (money_position * current_money_price + capital) * (1 - CONFIG['commission_rate'])
                    buy_price = row[f'close_{best_code}']
                    
                    position[best_code] = buy_capital / buy_price
                    holding_code, holding_cost = best_code, buy_price
                    money_position, capital = 0, 0
                    trade_log.append({'date': date, 'action': 'BUY/SWITCH', 'code': best_code, 'price': buy_price, 'value': buy_capital})

        # 3. 记录
        pv = capital + (position[holding_code] * row[f'close_{holding_code}'] if holding_code else 0) + (money_position * current_money_price)
        portfolio_values.append({'date': date, 'portfolio_value': pv, 'holding_name': FUND_POOL.get(holding_code, 'MONEY_FUND')})
        daily_status.append({'date': date, 'portfolio_value': pv, 'benchmark_close': row['close_benchmark']})

    return portfolio_values, trade_log, daily_status

# (plot_charts_eng 保持不变)

def send_email(report_text, chart_path, trades):
    sender = os.environ.get('EMAIL_SENDER')
    pwd = os.environ.get('EMAIL_PASSWORD')
    if not sender or not pwd: return

    # 检查今天是否有操作
    today_str = datetime.now().strftime('%Y-%m-%d')
    today_trades = [t for t in trades if t['date'].strftime('%Y-%m-%d') == today_str]
    
    subject = f"Strategy Report - {today_str}"
    if today_trades:
        subject = f"⚠️【交易建议】 红利策略 - {today_str}"
        action_msg = "\n今日操作建议：\n" + "\n".join([f"{t['action']}: {t['code']} (Value: {t['value']:.2f})" for t in today_trades])
        report_text = action_msg + "\n" + "="*30 + "\n" + report_text

    msg = MIMEMultipart()
    msg['Subject'] = subject
    msg['From'] = sender
    msg['To'] = sender
    msg.attach(MIMEText(report_text, 'plain', 'utf-8'))
    
    if os.path.exists(chart_path):
        with open(chart_path, 'rb') as f:
            img = MIMEImage(f.read())
            img.add_header('Content-Disposition', 'attachment', filename="chart.png")
            msg.attach(img)

    try:
        smtp_server = "smtp.gmail.com" if "gmail" in sender else "smtp.qq.com"
        with smtplib.SMTP_SSL(smtp_server, 465) as server:
            server.login(sender, pwd)
            server.sendmail(sender, [sender], msg.as_string())
        print("Email Sent Successfully!")
    except Exception as e:
        print(f"Failed: {e}")

def main():
    output_capture = io.StringIO()
    sys.stdout = output_capture
    try:
        fund_data, benchmark_df = fetch_all_fund_data()
        merged_df = prepare_merged_data(fund_data, benchmark_df)
        pv_list, trades, daily_list = run_backtest(merged_df, fund_data)
        
        pdf = pd.DataFrame(pv_list)
        ddf = pd.DataFrame(daily_list)
        
        print(f"Update: {pdf['date'].iloc[-1].strftime('%Y-%m-%d')}")
        print(f"Status: {pdf['holding_name'].iloc[-1]}")
        print(f"Total Return: {((pdf['portfolio_value'].iloc[-1]/CONFIG['initial_capital'])-1)*100:.2f}%")
        
        chart_file = plot_charts_eng(pdf, ddf, fund_data)
        report_text = output_capture.getvalue()
        sys.stdout = sys.__stdout__
        print(report_text)
        send_email(report_text, chart_file, trades)
    except Exception as e:
        sys.stdout = sys.__stdout__
        print(f"Main Error: {e}")

if __name__ == "__main__":
    main()
