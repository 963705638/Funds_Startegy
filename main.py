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
    'cooldown_days': 10,         # 止损后的冷静期
    'ma_period': 120,
    'start_date': '2023-01-01',
    'data_dir': './fund_data',
    'commission_rate': 0.0001,   # ⭐实战费率：万一
    'money_fund_yield': 0.02,    # 货基年化收益模拟
}

FUND_POOL = {
    "515080": "CS_Dividend", "510880": "SSE_Dividend", "515180": "E_Fund_Div",
    "513530": "HK_Dividend", "563020": "Low_Vol_Div", "510720": "SOE_Dividend",
    "159209": "Quality_Div"
}
BENCHMARK = {"code": "510300", "name": "HS300"}

os.makedirs(CONFIG['data_dir'], exist_ok=True)

# --- 数据获取 (保持你原有的逻辑) ---
def get_fund_data(code):
    url = f"https://fundmobapi.eastmoney.com/FundMNewApi/FundMNHisNetList?FCODE={code}&pageSize=4000&deviceid=1"
    try:
        r = requests.get(url, timeout=10).json()
        df = pd.DataFrame([{'date': i['FSRQ'], 'close': float(i['DWJZ'])} for i in r['Datas']])
        df['date'] = pd.to_datetime(df['date'])
        return df.sort_values('date')
    except: return pd.DataFrame()

# --- 整合数据与计算指标 ---
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

# --- 回测与交易建议引擎 ---
def run_strategy(df):
    cap = CONFIG['initial_capital']
    hold_code, hold_price, hold_qty = None, 0, 0
    money_val = 0
    trades, history = [], []
    
    # 模拟货基每日收益
    daily_m_rate = (1 + CONFIG['money_fund_yield'])**(1/252)

    for i, row in df.iterrows():
        date = row['date']
        money_val *= daily_m_rate
        devs = {c: row[f'dev_{c}'] for c in FUND_POOL.keys()}
        
        # 1. 卖出/止损
        if hold_code:
            cur_p = row[f'close_{hold_code}']
            if devs[hold_code] > CONFIG['sell_threshold'] or (cur_p/hold_price-1) < CONFIG['stop_loss_threshold']:
                money_val = hold_qty * cur_p * (1 - CONFIG['commission_rate'])
                trades.append({'date': date, 'action': 'SELL', 'code': hold_code, 'price': cur_p})
                hold_code, hold_qty = None, 0

        # 2. 买入/换仓
        candidates = {c: d for c, d in devs.items() if d < CONFIG['buy_threshold']}
        if candidates:
            best = min(candidates, key=candidates.get)
            if not hold_code or devs[best] < (devs[hold_code] - CONFIG['switch_threshold']):
                if hold_code: # 先卖
                    money_val = hold_qty * row[f'close_{hold_code}'] * (1 - CONFIG['commission_rate'])
                
                # 买入
                buy_cap = (money_val + cap) * (1 - CONFIG['commission_rate'])
                hold_code, hold_price = best, row[f'close_{best}']
                hold_qty = buy_cap / hold_price
                money_val, cap = 0, 0
                trades.append({'date': date, 'action': 'BUY/SWITCH', 'code': best, 'price': hold_price})

        # 记录净值
        total = cap + money_val + (hold_qty * row[f'close_{hold_code}'] if hold_code else 0)
        history.append({'date': date, 'value': total, 'holding': hold_code or 'CASH'})
    
    return pd.DataFrame(history), trades

# --- 邮件与主逻辑 ---
def main():
    log_capture = io.StringIO()
    sys.stdout = log_capture
    
    try:
        data = prepare_data()
        hist, trades = run_strategy(data)
        
        # 打印简报
        last_day = hist.iloc[-1]
        print(f"Update: {last_day['date'].strftime('%Y-%m-%d')}")
        print(f"Current Holding: {last_day['holding']}")
        print(f"Total Return: {((last_day['value']/CONFIG['initial_capital'])-1)*100:.2f}%")
        
        # 检查今日指令
        today_trade = [t for t in trades if t['date'].date() == datetime.now().date()]
        subject = "Dividend Strategy Report"
        trade_msg = ""
        if today_trade:
            subject = "⚠️【交易提醒】红利策略指令"
            trade_msg = "🚨 今日操作建议：\n" + "\n".join([f"{t['action']} {t['code']} @ {t['price']}" for t in today_trade])
        
        # 发送邮件 (这里省略绘图代码，结构参考上一轮回复)
        report = trade_msg + "\n\n详细日志：\n" + log_capture.getvalue()
        print(report) # 控制台也输出一份
        
        # 保存日志到本地以便 GitHub Action 提交
        with open("log.txt", "w") as f: f.write(report)
            
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
