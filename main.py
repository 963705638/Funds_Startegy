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

# 1. 增强版数据获取 (防屏蔽)
def get_fund_data(code):
    headers = {
        'User-Agent': 'Mozilla/5.0 (iPhone; CPU iPhone OS 13_2_3 like Mac OS X) AppleWebKit/605.1.15',
        'Referer': 'https://j5.dfcfw.com/',
        'Host': 'fundmobapi.eastmoney.com'
    }
    url = f"https://fundmobapi.eastmoney.com/FundMNewApi/FundMNHisNetList"
    params = {'FCODE': code, 'pageSize': '4000', 'deviceid': '1', 'plat': 'Iphone', 'product': 'EFund', 'version': '6.2.9'}

    for attempt in range(3):
        try:
            r = requests.get(url, params=params, headers=headers, timeout=15)
            if r.status_code == 200:
                data = r.json()
                if data and data.get('Datas'):
                    df = pd.DataFrame([{'date': i['FSRQ'], 'close': float(i['DWJZ'])} for i in data['Datas']])
                    df['date'] = pd.to_datetime(df['date'])
                    return df.sort_values('date')
        except Exception as e:
            pass
    return pd.DataFrame()

# 2. 修复版数据合并 (防止 Date 列丢失)
def prepare_data():
    all_dfs = []
    for code in FUND_POOL.keys():
        df = get_fund_data(code)
        if not df.empty:
            df = df.rename(columns={'close': f'close_{code}'})
            all_dfs.append(df.set_index('date'))
            
    bench = get_fund_data(BENCHMARK['code'])
    if not bench.empty:
        bench = bench.rename(columns={'close': 'close_bench'}).set_index('date')
        all_dfs.append(bench)
    else:
        raise Exception("无法获取基准数据(HS300)")

    merged = pd.concat(all_dfs, axis=1).ffill().dropna().reset_index()
    if 'date' not in merged.columns and 'index' in merged.columns:
        merged = merged.rename(columns={'index': 'date'})
        
    for code in FUND_POOL.keys():
        col_name = f'close_{code}'
        if col_name in merged.columns:
            ma = merged[col_name].rolling(CONFIG['ma_period']).mean()
            merged[f'dev_{code}'] = merged[col_name] / ma - 1
            
    return merged[merged['date'] >= CONFIG['start_date']].reset_index(drop=True)

# 3. 回测与信号生成引擎
def run_strategy(df):
    cap = CONFIG['initial_capital']
    hold_code, hold_price, hold_qty = None, 0, 0
    money_val = 0
    trades, history = [], []
    daily_m_rate = (1 + CONFIG['money_fund_yield'])**(1/252)

    for i, row in df.iterrows():
        date = row['date']
        money_val *= daily_m_rate
        devs = {c: row[f'dev_{c}'] for c in FUND_POOL.keys() if f'dev_{c}' in row}
        
        # 卖出/止损
        if hold_code:
            cur_p = row[f'close_{hold_code}']
            if devs.get(hold_code, 0) > CONFIG['sell_threshold'] or (cur_p/hold_price-1) < CONFIG['stop_loss_threshold']:
                money_val = hold_qty * cur_p * (1 - CONFIG['commission_rate'])
                trades.append({'date': date, 'action': 'SELL', 'code': hold_code, 'price': cur_p})
                hold_code, hold_qty = None, 0

        # 买入/换仓
        candidates = {c: d for c, d in devs.items() if d < CONFIG['buy_threshold']}
        if candidates:
            best = min(candidates, key=candidates.get)
            if not hold_code or devs[best] < (devs.get(hold_code, 0) - CONFIG['switch_threshold']):
                if hold_code:
                    money_val = hold_qty * row[f'close_{hold_code}'] * (1 - CONFIG['commission_rate'])
                buy_cap = (money_val + cap) * (1 - CONFIG['commission_rate'])
                hold_code, hold_price = best, row[f'close_{best}']
                hold_qty = buy_cap / hold_price
                money_val, cap = 0, 0
                trades.append({'date': date, 'action': 'BUY/SWITCH', 'code': best, 'price': hold_price})

        total = cap + money_val + (hold_qty * row.get(f'close_{hold_code}', 0) if hold_code else 0)
        history.append({'date': date, 'value': total, 'holding': hold_code or 'CASH', 'bench_close': row['close_bench']})
    
    return pd.DataFrame(history), trades, df

# 4. 绘图
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

# 5. 邮件发送
def send_email(subject, content, img_path=None):
    sender = os.environ.get('EMAIL_SENDER')
    pwd = os.environ.get('EMAIL_PASSWORD')
    if not sender or not pwd: return

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
    except Exception as e:
        print(f"邮件发送失败: {e}")

# 6. 主程序 (专注于实盘指令生成)
def main():
    log_capture = io.StringIO()
    sys.stdout = log_capture
    
    try:
        data = prepare_data()
        hist, trades, raw_df = run_strategy(data)
        img_path = plot_report(hist)
        
        # --- 获取最新一日的数据状态 ---
        last_date = data['date'].iloc[-1]
        last_date_str = last_date.strftime('%Y-%m-%d')
        last_hist = hist.iloc[-1]
        last_raw = raw_df.iloc[-1]
        
        # --- 提取今日交易指令 ---
        today_trades = [t for t in trades if t['date'].date() == last_date.date()]
        
        # 构建指令头部
        if today_trades:
            subject = f"⚠️【交易指令】红利策略调仓 - {last_date_str}"
            instruction_text = "🚨 【今日需执行以下操作】\n"
            for t in today_trades:
                action_cn = "买入" if "BUY" in t['action'] else "卖出"
                fund_name = FUND_POOL.get(t['code'], t['code'])
                instruction_text += f"   ➡️ {action_cn}: {fund_name} ({t['code']}) | 触发价: {t['price']:.4f}\n"
        else:
            subject = f"📊【持仓观望】红利策略日报 - {last_date_str}"
            instruction_text = "⏸️ 【今日无操作信号，继续持有当前仓位】\n"

        # 构建数据看板
        dashboard = []
        dashboard.append("=" * 40)
        dashboard.append(f"📅 数据更新至: {last_date_str}")
        dashboard.append(f"🏦 当前持有标的: {FUND_POOL.get(last_hist['holding'], last_hist['holding'])}")
        dashboard.append(f"💰 模拟账户净值: {last_hist['value']:.2f} 元 (初始 {CONFIG['initial_capital']})")
        total_ret = ((last_hist['value'] / CONFIG['initial_capital']) - 1) * 100
        dashboard.append(f"📈 策略累计收益: {total_ret:.2f}%")
        dashboard.append("=" * 40)
        
        # 构建各基金的乖离率雷达 (让你知道离买卖点还有多远)
        dashboard.append("\n📡 【成分基乖离率扫描】 (低于 -3% 触发买入，高于 7% 触发卖出)")
        for code, name in FUND_POOL.items():
            col = f'dev_{code}'
            if col in last_raw:
                dev_val = last_raw[col]
                # 加点视觉提示
                if dev_val < CONFIG['buy_threshold']: icon = "🟢 [超跌]"
                elif dev_val > CONFIG['sell_threshold']: icon = "🔴 [超涨]"
                else: icon = "⚪ [正常]"
                dashboard.append(f"   {icon} {name} ({code}): {dev_val*100:>5.2f}%")
        
        # 拼接最终邮件内容
        final_report = f"{instruction_text}\n" + "\n".join(dashboard)
        
        # 恢复标准输出打印
        sys.stdout = sys.__stdout__
        print(final_report)
        print("\n(正在发送邮件并保存日志...)")
        
        send_email(subject, final_report, img_path)
        
        with open("log.txt", "w", encoding='utf-8') as f: 
            f.write(final_report + "\n\n--- 历史详细日志 ---\n" + log_capture.getvalue())
            
    except Exception as e:
        sys.stdout = sys.__stdout__
        print(f"❌ 运行报错: {e}")

if __name__ == "__main__":
    main()
