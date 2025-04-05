import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from datetime import datetime, timedelta
import scipy.stats as stats
import math

# 페이지 설정
st.set_page_config(
    page_title="Option Calculator",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# 스타일 추가 (스피너 버튼 제거 CSS 포함)
st.markdown(
    """
<style>
    .stMarkdown {
        padding: 10px 0px;
    }
    div.block-container {
        padding-top: 2rem;
    }
    .highlight {
        background-color: #f0f2f6;
        padding: 10px;
        border-radius: 4px;
    }
    .profit {
        color: #1cd4c8;
    }
    .loss {
        color: #d41c78;
    }
    .main-text {
        font-size: 18px;
    }
    .header-text {
        font-size: 24px;
        font-weight: bold;
    }
    .center-text {
        text-align: center;
    }
    .sidebar-header {
        font-size: 20px;
        font-weight: bold;
    }
    .small-font {
        font-size: 14px;
    }
</style>
""",
    unsafe_allow_html=True,
)


# 유틸리티 함수
def fetch_data(ticker):
    """주식 정보 및 옵션 체인 가져오기"""
    try:
        stock = yf.Ticker(ticker)
        stock_info = stock.info

        underlying_price = stock_info.get("currentPrice", None)
        if underlying_price is None:
            underlying_price = stock_info.get("regularMarketPrice", None)

        try:
            rf = round(yf.Ticker("^TNX").history(period="1d").iloc[0, 3], 2)
        except:
            rf = 3.95

        dividend_yield = stock_info.get("dividendYield", 0.0)

        open_price = stock_info.get("open", None)
        high = stock_info.get("dayHigh", None)
        low = stock_info.get("dayLow", None)

        previous_close = stock_info.get("previousClose", None)
        if previous_close and underlying_price:
            chg = round(((underlying_price / previous_close) - 1) * 100, 3)
        else:
            chg = 0.0

        name = stock_info.get("shortName", "N/A")
        sector = stock_info.get("sector", "N/A")

        expiry_dates = stock.options

        try:
            stock_data = yf.download(ticker, period="1y", progress=False)
            close_col = "Close"
            stock_data["Returns"] = np.log(
                stock_data[close_col] / stock_data[close_col].shift(1)
            )
            stock_data.dropna(inplace=True)
            daily_volatility = stock_data["Returns"].std()
            annual_volatility = daily_volatility * np.sqrt(252)
            vol = round(annual_volatility * 100, 2)
        except Exception as e:
            st.warning(f"Volatility calculation warning: {e}")
            vol = 30.0

        return {
            "underlying_price": underlying_price,
            "rf": rf,
            "dividend_yield": dividend_yield,
            "expiry_dates": expiry_dates,
            "open_price": open_price,
            "high": high,
            "low": low,
            "chg": chg,
            "name": name,
            "sector": sector,
            "vol": vol,
        }
    except Exception as e:
        st.error(f"Error fetching data: {e}")
        return None


def get_option_chain(ticker, expiry_date):
    """특정 만기일에 대한 옵션 체인 가져오기"""
    try:
        stock = yf.Ticker(ticker)
        calls = stock.option_chain(expiry_date).calls
        puts = stock.option_chain(expiry_date).puts

        calls["type"] = "CALL"
        puts["type"] = "PUT"

        return calls, puts
    except Exception as e:
        st.error(f"Error fetching option chain: {e}")
        return None, None


def calculate_days_to_expiry(expiry_date):
    """만기일까지 남은 일수 계산 (만기일 포함)"""
    if not expiry_date:
        return None
    expiry = datetime.strptime(expiry_date, "%Y-%m-%d")
    # 만기일에 하루를 더함
    expiry = expiry + timedelta(days=1)
    today = datetime.now()
    return (expiry - today).days


def bs_model(s, k, rf, tau, sigma, y, option_type="c"):
    """블랙-숄즈 모델로 옵션 가격 계산"""
    T = 252
    pct = 100
    tau = tau / T
    sigma = sigma / pct
    rf = rf / pct
    y = y / pct

    d1 = (np.log(s / k) + (rf - y + 0.5 * sigma**2) * tau) / (sigma * np.sqrt(tau))
    d2 = d1 - sigma * np.sqrt(tau)

    call_price = s * np.exp(-y * tau) * stats.norm.cdf(d1) - k * np.exp(
        -rf * tau
    ) * stats.norm.cdf(d2)
    put_price = k * np.exp(-rf * tau) * stats.norm.cdf(-d2) - s * np.exp(
        -y * tau
    ) * stats.norm.cdf(-d1)

    if option_type == "c":
        return call_price
    elif option_type == "p":
        return put_price
    else:
        return {"call_price": call_price, "put_price": put_price}


def option_greeks(s, k, rf, sigma, tau, y):
    """옵션 그릭스 계산"""
    T = 252
    pct = 100

    tau = tau / T
    sigma = sigma / pct
    rf = rf / pct
    y = y / pct

    d1 = (np.log(s / k) + (rf - y + sigma**2 / 2) * tau) / (sigma * np.sqrt(tau))
    d2 = d1 - (sigma * np.sqrt(tau))

    nd1 = (1 / (np.sqrt(2 * np.pi))) * np.exp(-(d1**2 / 2))

    call_delta = stats.norm.cdf(d1)
    put_delta = stats.norm.cdf(d1) - 1

    gamma = nd1 / (s * sigma * np.sqrt(tau))

    call_theta = (
        -((s * sigma) / (2 * np.sqrt(tau))) * nd1
        - k * np.exp(-rf * tau) * rf * stats.norm.cdf(d2)
    ) / T
    put_theta = (
        -((s * sigma) / (2 * np.sqrt(tau))) * nd1
        - k * np.exp(-rf * tau) * rf * (stats.norm.cdf(d2) - 1)
    ) / T

    call_rho = (k * tau * np.exp(-rf * tau) * stats.norm.cdf(d2)) / pct
    put_rho = (k * tau * np.exp(-rf * tau) * (stats.norm.cdf(d2) - 1)) / pct

    vega = (s * np.sqrt(tau) * nd1) / pct

    return {
        "call_delta": call_delta,
        "put_delta": put_delta,
        "gamma": gamma,
        "vega": vega,
        "call_theta": call_theta,
        "put_theta": put_theta,
        "call_rho": call_rho,
        "put_rho": put_rho,
    }


def calculate_payoff_df(
    strategy,
    side,
    option_type,
    s,
    k1,
    k2=None,
    k3=None,
    k4=None,
    price1=None,
    price2=None,
    price3=None,
    price4=None,
    size=1,
    tau=30,
    rf=3.95,
    sigma=30,
    y=0,
    option_chain=None,
):
    """옵션 전략에 따른 손익 및 그릭스 계산"""
    k_values = [k for k in [k1, k2, k3, k4] if k is not None]
    if not k_values:
        return None

    k_min, k_max = min(k_values), max(k_values)
    price_range_min = max(0.1, s * 0.7, k_min * 0.7)
    price_range_max = max(s * 1.3, k_max * 1.3)

    x = np.arange(price_range_min, price_range_max, 0.1)
    df = pd.DataFrame({"x": x})

    def calculate_call_payoff(spot_price, strike, premium):
        return np.where(spot_price <= strike, -premium, (spot_price - strike) - premium)

    def calculate_put_payoff(spot_price, strike, premium):
        return np.where(spot_price <= strike, (strike - spot_price) - premium, -premium)

    call_delta1 = np.zeros(len(x))
    call_delta2 = np.zeros(len(x))
    call_delta3 = np.zeros(len(x))
    call_delta4 = np.zeros(len(x))

    put_delta1 = np.zeros(len(x))
    put_delta2 = np.zeros(len(x))
    put_delta3 = np.zeros(len(x))
    put_delta4 = np.zeros(len(x))

    gamma1 = np.zeros(len(x))
    gamma2 = np.zeros(len(x))
    gamma3 = np.zeros(len(x))
    gamma4 = np.zeros(len(x))

    vega1 = np.zeros(len(x))
    vega2 = np.zeros(len(x))
    vega3 = np.zeros(len(x))
    vega4 = np.zeros(len(x))

    call_theta1 = np.zeros(len(x))
    call_theta2 = np.zeros(len(x))
    call_theta3 = np.zeros(len(x))
    call_theta4 = np.zeros(len(x))

    put_theta1 = np.zeros(len(x))
    put_theta2 = np.zeros(len(x))
    put_theta3 = np.zeros(len(x))
    put_theta4 = np.zeros(len(x))

    call_rho1 = np.zeros(len(x))
    call_rho2 = np.zeros(len(x))
    call_rho3 = np.zeros(len(x))
    call_rho4 = np.zeros(len(x))

    put_rho1 = np.zeros(len(x))
    put_rho2 = np.zeros(len(x))
    put_rho3 = np.zeros(len(x))
    put_rho4 = np.zeros(len(x))

    if k1 is not None:
        greeks1 = [option_greeks(price, k1, rf, sigma, tau, y) for price in x]
        call_delta1 = np.array([g["call_delta"] for g in greeks1])
        put_delta1 = np.array([g["put_delta"] for g in greeks1])
        gamma1 = np.array([g["gamma"] for g in greeks1])
        vega1 = np.array([g["vega"] for g in greeks1])
        call_theta1 = np.array([g["call_theta"] for g in greeks1])
        put_theta1 = np.array([g["put_theta"] for g in greeks1])
        call_rho1 = np.array([g["call_rho"] for g in greeks1])
        put_rho1 = np.array([g["put_rho"] for g in greeks1])

    if k2 is not None:
        greeks2 = [option_greeks(price, k2, rf, sigma, tau, y) for price in x]
        call_delta2 = np.array([g["call_delta"] for g in greeks2])
        put_delta2 = np.array([g["put_delta"] for g in greeks2])
        gamma2 = np.array([g["gamma"] for g in greeks2])
        vega2 = np.array([g["vega"] for g in greeks2])
        call_theta2 = np.array([g["call_theta"] for g in greeks2])
        put_theta2 = np.array([g["put_theta"] for g in greeks2])
        call_rho2 = np.array([g["call_rho"] for g in greeks2])
        put_rho2 = np.array([g["put_rho"] for g in greeks2])

    if k3 is not None:
        greeks3 = [option_greeks(price, k3, rf, sigma, tau, y) for price in x]
        call_delta3 = np.array([g["call_delta"] for g in greeks3])
        put_delta3 = np.array([g["put_delta"] for g in greeks3])
        gamma3 = np.array([g["gamma"] for g in greeks3])
        vega3 = np.array([g["vega"] for g in greeks3])
        call_theta3 = np.array([g["call_theta"] for g in greeks3])
        put_theta3 = np.array([g["put_theta"] for g in greeks3])
        call_rho3 = np.array([g["call_rho"] for g in greeks3])
        put_rho3 = np.array([g["put_rho"] for g in greeks3])

    if k4 is not None:
        greeks4 = [option_greeks(price, k4, rf, sigma, tau, y) for price in x]
        call_delta4 = np.array([g["call_delta"] for g in greeks4])
        put_delta4 = np.array([g["put_delta"] for g in greeks4])
        gamma4 = np.array([g["gamma"] for g in greeks4])
        vega4 = np.array([g["vega"] for g in greeks4])
        call_theta4 = np.array([g["call_theta"] for g in greeks4])
        put_theta4 = np.array([g["put_theta"] for g in greeks4])
        call_rho4 = np.array([g["call_rho"] for g in greeks4])
        put_rho4 = np.array([g["put_rho"] for g in greeks4])

    c1 = (
        calculate_call_payoff(x, k1, price1)
        if k1 is not None and price1 is not None
        else 0
    )
    p1 = (
        calculate_put_payoff(x, k1, price1)
        if k1 is not None and price1 is not None
        else 0
    )

    c2 = (
        calculate_call_payoff(x, k2, price2)
        if k2 is not None and price2 is not None
        else 0
    )
    p2 = (
        calculate_put_payoff(x, k2, price2)
        if k2 is not None and price2 is not None
        else 0
    )

    c3 = (
        calculate_call_payoff(x, k3, price3)
        if k3 is not None and price3 is not None
        else 0
    )
    p3 = (
        calculate_put_payoff(x, k3, price3)
        if k3 is not None and price3 is not None
        else 0
    )

    c4 = (
        calculate_call_payoff(x, k4, price4)
        if k4 is not None and price4 is not None
        else 0
    )
    p4 = (
        calculate_put_payoff(x, k4, price4)
        if k4 is not None and price4 is not None
        else 0
    )

    if strategy == "Single":
        if side == "LONG" and option_type == "CALL":
            y_values = c1
            delta = call_delta1
            gamma = gamma1
            vega = vega1
            theta = call_theta1
            rho = call_rho1
            strategy_type = "Bullish"
            risk_level = "Moderate Risk"
        elif side == "SHORT" and option_type == "CALL":
            y_values = -c1
            delta = -call_delta1
            gamma = -gamma1
            vega = -vega1
            theta = -call_theta1
            rho = -call_rho1
            strategy_type = "Bearish"
            risk_level = "High Risk"
        elif side == "LONG" and option_type == "PUT":
            y_values = p1
            delta = put_delta1
            gamma = gamma1
            vega = vega1
            theta = put_theta1
            rho = put_rho1
            strategy_type = "Bearish"
            risk_level = "Moderate Risk"
        elif side == "SHORT" and option_type == "PUT":
            y_values = -p1
            delta = -put_delta1
            gamma = -gamma1
            vega = -vega1
            theta = -put_theta1
            rho = -put_rho1
            strategy_type = "Bullish"
            risk_level = "High Risk"

    elif strategy == "Straddle":
        if side == "LONG":
            y_values = p1 + c1
            delta = put_delta1 + call_delta1
            gamma = gamma1 + gamma1
            vega = vega1 + vega1
            theta = put_theta1 + call_theta1
            rho = put_rho1 + call_rho1
            strategy_type = "Neutral"
            risk_level = "High Risk"
        elif side == "SHORT":
            y_values = -(p1 + c1)
            delta = -(put_delta1 + call_delta1)
            gamma = -(gamma1 + gamma1)
            vega = -(vega1 + vega1)
            theta = -(put_theta1 + call_theta1)
            rho = -(put_rho1 + call_rho1)
            strategy_type = "Neutral"
            risk_level = "High Risk"

    elif strategy == "Strangle":
        if side == "LONG":
            y_values = p1 + c2
            delta = put_delta1 + call_delta2
            gamma = gamma1 + gamma2
            vega = vega1 + vega2
            theta = put_theta1 + call_theta2
            rho = put_rho1 + call_rho2
            strategy_type = "Neutral"
            risk_level = "High Risk"
        elif side == "SHORT":
            y_values = -(p1 + c2)
            delta = -(put_delta1 + call_delta2)
            gamma = -(gamma1 + gamma2)
            vega = -(vega1 + vega2)
            theta = -(put_theta1 + call_theta2)
            rho = -(put_rho1 + call_rho2)
            strategy_type = "Neutral"
            risk_level = "High Risk"

    elif strategy == "Spread":
        if side == "LONG" and option_type == "CALL":
            y_values = c1 - c2
            delta = call_delta1 - call_delta2
            gamma = gamma1 - gamma2
            vega = vega1 - vega2
            theta = call_theta1 - call_theta2
            rho = call_rho1 - call_rho2
            strategy_type = "Bullish"
            risk_level = "Moderate Risk"
        elif side == "SHORT" and option_type == "CALL":
            y_values = -(c1 - c2)
            delta = -(call_delta1 - call_delta2)
            gamma = -(gamma1 - gamma2)
            vega = -(vega1 - vega2)
            theta = -(call_theta1 - call_theta2)
            rho = -(call_rho1 - call_rho2)
            strategy_type = "Bearish"
            risk_level = "Moderate Risk"
        elif side == "SHORT" and option_type == "PUT":
            y_values = p1 - p2
            delta = put_delta1 - put_delta2
            gamma = gamma1 - gamma2
            vega = vega1 - vega2
            theta = put_theta1 - put_theta2
            rho = put_rho1 - put_rho2
            strategy_type = "Bullish"
            risk_level = "Moderate Risk"
        elif side == "LONG" and option_type == "PUT":
            y_values = -(p1 - p2)
            delta = -(put_delta1 - put_delta2)
            gamma = -(gamma1 - gamma2)
            vega = -(vega1 - vega2)
            theta = -(put_theta1 - put_theta2)
            rho = -(put_rho1 - put_rho2)
            strategy_type = "Bearish"
            risk_level = "Moderate Risk"

    elif strategy == "Covered" and option_type == "PUT":
        y_values = (s - x) - p1
        delta = -1 + put_delta1
        gamma = gamma1
        vega = vega1
        theta = -put_theta1
        rho = -put_rho1
        strategy_type = "Bearish"
        risk_level = "Moderate Risk"

    elif strategy == "Covered" and option_type == "CALL":
        y_values = (x - s) - c1
        delta = 1 - call_delta1
        gamma = -gamma1
        vega = -vega1
        theta = -call_theta1
        rho = -call_rho1
        strategy_type = "Bullish"
        risk_level = "Low Risk"

    elif strategy == "Protective" and option_type == "PUT":
        y_values = (x - s) + p1
        delta = 1 + put_delta1
        gamma = gamma1
        vega = vega1
        theta = put_theta1
        rho = put_rho1
        strategy_type = "Bullish"
        risk_level = "Moderate Risk"

    elif strategy == "Protective" and option_type == "CALL":
        y_values = (s - x) + c1
        delta = -1 + call_delta1
        gamma = gamma1
        vega = vega1
        theta = call_theta1
        rho = call_rho1
        strategy_type = "Bearish"
        risk_level = "Low Risk"

    elif strategy == "Strip":
        y_values = c1 + 2 * p1
        delta = call_delta1 + 2 * put_delta1
        gamma = 3 * gamma1
        vega = 3 * vega1
        theta = call_theta1 + 2 * put_theta1
        rho = call_rho1 + 2 * put_rho1
        strategy_type = "Bearish"
        risk_level = "High Risk"

    elif strategy == "Strap":
        y_values = 2 * c1 + p1
        delta = 2 * call_delta1 + put_delta1
        gamma = 3 * gamma1
        vega = 3 * vega1
        theta = 2 * call_theta1 + put_theta1
        rho = 2 * call_rho1 + put_rho1
        strategy_type = "Bullish"
        risk_level = "High Risk"

    elif strategy == "Butterfly":
        if side == "LONG" and option_type == "CALL":
            y_values = c1 - 2 * c2 + c3
            delta = call_delta1 - 2 * call_delta2 + call_delta3
            gamma = gamma1 - 2 * gamma2 + gamma3
            vega = vega1 - 2 * vega2 + vega3
            theta = call_theta1 - 2 * call_theta2 + call_theta3
            rho = call_rho1 - 2 * call_rho2 + call_rho3
            strategy_type = "Neutral"
            risk_level = "Low Risk"
        elif side == "SHORT" and option_type == "CALL":
            y_values = -(c1 - 2 * c2 + c3)
            delta = -(call_delta1 - 2 * call_delta2 + call_delta3)
            gamma = -(gamma1 - 2 * gamma2 + gamma3)
            vega = -(vega1 - 2 * vega2 + vega3)
            theta = -(call_theta1 - 2 * call_theta2 + call_theta3)
            rho = -(call_rho1 - 2 * call_rho2 + call_rho3)
            strategy_type = "Neutral"
            risk_level = "Low Risk"
        elif side == "LONG" and option_type == "PUT":
            y_values = p1 - 2 * p2 + p3
            delta = put_delta1 - 2 * put_delta2 + put_delta3
            gamma = gamma1 - 2 * gamma2 + gamma3
            vega = vega1 - 2 * vega2 + vega3
            theta = put_theta1 - 2 * put_theta2 + put_theta3
            rho = put_rho1 - 2 * put_rho2 + put_rho3
            strategy_type = "Neutral"
            risk_level = "Low Risk"
        elif side == "SHORT" and option_type == "PUT":
            y_values = -(p1 - 2 * p2 + p3)
            delta = -(put_delta1 - 2 * put_delta2 + put_delta3)
            gamma = -(gamma1 - 2 * gamma2 + gamma3)
            vega = -(vega1 - 2 * vega2 + vega3)
            theta = -(put_theta1 - 2 * put_theta2 + put_theta3)
            rho = -(put_rho1 - 2 * put_rho2 + put_rho3)
            strategy_type = "Neutral"
            risk_level = "Low Risk"

    elif strategy == "Ladder":
        if side == "LONG" and option_type == "CALL":
            y_values = c1 - c2 - c3
            delta = call_delta1 - call_delta2 - call_delta3
            gamma = gamma1 - gamma2 - gamma3
            vega = vega1 - vega2 - vega3
            theta = call_theta1 - call_theta2 - call_theta3
            rho = call_rho1 - call_rho2 - call_rho3
            strategy_type = "Bullish"
            risk_level = "High Risk"
        elif side == "SHORT" and option_type == "CALL":
            y_values = -(c1 - c2 - c3)
            delta = -(call_delta1 - call_delta2 - call_delta3)
            gamma = -(gamma1 - gamma2 - gamma3)
            vega = -(vega1 - vega2 - vega3)
            theta = -(call_theta1 - call_theta2 - call_theta3)
            rho = -(call_rho1 - call_rho2 - call_rho3)
            strategy_type = "Bearish"
            risk_level = "High Risk"
        elif side == "SHORT" and option_type == "PUT":
            y_values = p1 + p2 - p3
            delta = put_delta1 + put_delta2 - put_delta3
            gamma = gamma1 + gamma2 - gamma3
            vega = vega1 + vega2 - vega3
            theta = put_theta1 + put_theta2 - put_theta3
            rho = put_rho1 + put_rho2 - put_rho3
            strategy_type = "Bullish"
            risk_level = "High Risk"
        elif side == "LONG" and option_type == "PUT":
            y_values = -(p1 + p2 - p3)
            delta = -(put_delta1 + put_delta2 - put_delta3)
            gamma = -(gamma1 + gamma2 - gamma3)
            vega = -(vega1 + vega2 - vega3)
            theta = -(put_theta1 + put_theta2 - put_theta3)
            rho = -(put_rho1 + put_rho2 - put_rho3)
            strategy_type = "Bearish"
            risk_level = "High Risk"

    elif strategy == "Jade Lizard":
        y_values = -p1 - c2 + c3
        delta = -put_delta1 - call_delta2 + call_delta3
        gamma = -gamma1 - gamma2 + gamma3
        vega = -vega1 - vega2 + vega3
        theta = -put_theta1 - call_theta2 + call_theta3
        rho = -put_rho1 - call_rho2 + call_rho3
        strategy_type = "Bullish"
        risk_level = "Moderate Risk"

    elif strategy == "Reverse Jade Lizard":
        y_values = p1 - p2 - c3
        delta = put_delta1 - put_delta2 - call_delta3
        gamma = gamma1 - gamma2 - gamma3
        vega = vega1 - vega2 - vega3
        theta = put_theta1 - put_theta2 - call_theta3
        rho = put_rho1 - put_rho2 - call_rho3
        strategy_type = "Bearish"
        risk_level = "Moderate Risk"

    elif strategy == "Condor":
        if side == "LONG" and option_type == "CALL":
            y_values = c1 - c2 - c3 + c4
            delta = call_delta1 - call_delta2 - call_delta3 + call_delta4
            gamma = gamma1 - gamma2 - gamma3 + gamma4
            vega = vega1 - vega2 - vega3 + vega4
            theta = call_theta1 - call_theta2 - call_theta3 + call_theta4
            rho = call_rho1 - call_rho2 - call_rho3 + call_rho4
            strategy_type = "Neutral"
            risk_level = "Low Risk"
        elif side == "SHORT" and option_type == "CALL":
            y_values = -(c1 - c2 - c3 + c4)
            delta = -(call_delta1 - call_delta2 - call_delta3 + call_delta4)
            gamma = -(gamma1 - gamma2 - gamma3 + gamma4)
            vega = -(vega1 - vega2 - vega3 + vega4)
            theta = -(call_theta1 - call_theta2 - call_theta3 + call_theta4)
            rho = -(call_rho1 - call_rho2 - call_rho3 + call_rho4)
            strategy_type = "Neutral"
            risk_level = "Low Risk"
        elif side == "LONG" and option_type == "PUT":
            y_values = p1 - p2 - p3 + p4
            delta = put_delta1 - put_delta2 - put_delta3 + put_delta4
            gamma = gamma1 - gamma2 - gamma3 + gamma4
            vega = vega1 - vega2 - vega3 + vega4
            theta = put_theta1 - put_theta2 - put_theta3 + put_theta4
            rho = put_rho1 - put_rho2 - put_rho3 + put_rho4
            strategy_type = "Neutral"
            risk_level = "Low Risk"
        elif side == "SHORT" and option_type == "PUT":
            y_values = -(p1 - p2 - p3 + p4)
            delta = -(put_delta1 - put_delta2 - put_delta3 + put_delta4)
            gamma = -(gamma1 - gamma2 - gamma3 + gamma4)
            vega = -(vega1 - vega2 - vega3 + vega4)
            theta = -(put_theta1 - put_theta2 - put_theta3 + put_theta4)
            rho = -(put_rho1 - put_rho2 - put_rho3 + put_rho4)
            strategy_type = "Neutral"
            risk_level = "Low Risk"

    else:
        y_values = np.zeros(len(x))
        delta = np.zeros(len(x))
        gamma = np.zeros(len(x))
        vega = np.zeros(len(x))
        theta = np.zeros(len(x))
        rho = np.zeros(len(x))
        strategy_type = "Unknown"
        risk_level = "Unknown"

    y_values = y_values * size
    delta = delta * size
    gamma = gamma * size
    vega = vega * size
    theta = theta * size
    rho = rho * size

    df["y"] = y_values
    df["profit"] = np.where(y_values >= 0, y_values, np.nan)
    df["loss"] = np.where(y_values < 0, y_values, np.nan)
    df["Delta"] = delta
    df["Gamma"] = gamma
    df["Vega"] = vega
    df["Theta"] = theta
    df["Rho"] = rho

    calculated_max_profit = np.max(y_values)
    calculated_min_profit = np.min(y_values)

    sign_changes = np.where(np.diff(np.signbit(y_values)))[0]
    bep1 = x[sign_changes[0]] if len(sign_changes) > 0 else None
    bep2 = x[sign_changes[1]] if len(sign_changes) > 1 else None

    # 이론적으로 무한대인 경우 처리
    # 전략과 포지션에 따른 최대/최소 이익 계산
    if strategy == "Single":
        if side == "LONG":
            max_profit = float("inf")
            min_profit = calculated_min_profit
        elif side == "SHORT":
            max_profit = calculated_max_profit
            min_profit = float("-inf")

    elif strategy == "Straddle" or strategy == "Strangle":
        if side == "LONG":
            max_profit = float("inf")
            min_profit = calculated_min_profit
        elif side == "SHORT":
            max_profit = calculated_max_profit
            min_profit = float("-inf")

    elif strategy == "Spread":
        max_profit = calculated_max_profit
        min_profit = calculated_min_profit

    elif strategy == "Covered":
        if option_type == "PUT":
            max_profit = calculated_max_profit
            min_profit = float("-inf")
        else:
            max_profit = calculated_max_profit
            min_profit = float("-inf")

    elif strategy == "Protective":
        max_profit = float("inf")
        min_profit = calculated_min_profit

    elif strategy in ["Strip", "Strap"]:
        max_profit = float("inf")
        min_profit = calculated_min_profit

    elif strategy == "Butterfly" or strategy == "Condor":
        max_profit = calculated_max_profit
        min_profit = calculated_min_profit

    elif strategy == "Ladder":
        if side == "SHORT":
            max_profit = float("inf")
            min_profit = calculated_min_profit
        elif side == "LONG":
            max_profit = calculated_max_profit
            min_profit = float("-inf")

    elif strategy in ["Jade Lizard", "Reverse Jade Lizard"]:
        if strategy == "Jade Lizard":
            max_profit = calculated_max_profit
            min_profit = float("-inf")
        else:
            max_profit = calculated_max_profit
            min_profit = float("-inf")

    else:
        max_profit = calculated_max_profit
        min_profit = calculated_min_profit

    strategy_info = {
        "strategy_type": strategy_type,
        "risk_level": risk_level,
        "bep1": bep1,
        "bep2": bep2,
        "max_profit": max_profit,
        "min_profit": min_profit,
    }

    return df, strategy_info


def plot_option_strategy(df, s, greeks, strategy_info):
    """옵션 전략의 손익 및 그릭스 그래프 생성"""
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=df["x"],
            y=df["profit"],
            fill="tozeroy",
            name="Profit",
            line=dict(color="skyblue"),
            fillcolor="rgba(135, 206, 235, 0.25)",
            hoverinfo="x+y",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=df["x"],
            y=df["loss"],
            fill="tozeroy",
            name="Loss",
            line=dict(color="red"),
            fillcolor="rgba(255, 0, 0, 0.25)",
            hoverinfo="x+y",
        )
    )

    if greeks == "Delta":
        fig.add_trace(
            go.Scatter(
                x=df["x"],
                y=df["Delta"],
                mode="lines",
                name="Delta",
                line=dict(dash="dot", color="orange", width=1.5),
                yaxis="y2",
                hoverinfo="x+y",
            )
        )
    elif greeks == "Gamma":
        fig.add_trace(
            go.Scatter(
                x=df["x"],
                y=df["Gamma"],
                mode="lines",
                name="Gamma",
                line=dict(dash="dot", color="purple", width=1.5),
                yaxis="y2",
                hoverinfo="x+y",
            )
        )
    elif greeks == "Vega":
        fig.add_trace(
            go.Scatter(
                x=df["x"],
                y=df["Vega"],
                mode="lines",
                name="Vega",
                line=dict(dash="dot", color="green", width=1.5),
                yaxis="y2",
                hoverinfo="x+y",
            )
        )
    elif greeks == "Theta":
        fig.add_trace(
            go.Scatter(
                x=df["x"],
                y=df["Theta"],
                mode="lines",
                name="Theta",
                line=dict(dash="dot", color="brown", width=1.5),
                yaxis="y2",
                hoverinfo="x+y",
            )
        )
    elif greeks == "Rho":
        fig.add_trace(
            go.Scatter(
                x=df["x"],
                y=df["Rho"],
                mode="lines",
                name="Rho",
                line=dict(dash="dot", color="blue", width=1.5),
                yaxis="y2",
                hoverinfo="x+y",
            )
        )

    y_min = min(df["y"]) * 1.1 if min(df["y"]) < 0 else -1
    y_max = max(df["y"]) * 1.1 if max(df["y"]) > 0 else 1

    if np.isinf(strategy_info["max_profit"]):
        y_max = max(df["y"]) * 1.5
    if np.isinf(strategy_info["min_profit"]):
        y_min = min(df["y"]) * 1.5

    # 현재 주가 수직선 추가
    fig.add_shape(
        type="line",
        x0=s,
        x1=s,
        y0=y_min,
        y1=y_max,
        line=dict(color="green", width=2, dash="dash"),
    )

    fig.add_annotation(
        x=s,
        y=y_max,
        text=f"Current Price: {s:.2f}",
        showarrow=True,
        arrowhead=2,
        arrowsize=1,
        arrowwidth=1,
        arrowcolor="green",
        ax=0,
        ay=-40,
    )

    if strategy_info["bep1"] is not None:
        fig.add_shape(
            type="line",
            x0=strategy_info["bep1"],
            x1=strategy_info["bep1"],
            y0=y_min,
            y1=y_max,
            line=dict(dash="dot", color="black", width=1),
        )

        fig.add_annotation(
            x=strategy_info["bep1"],
            y=0,
            text=f"BEP: {strategy_info['bep1']:.2f}",
            showarrow=True,
            arrowhead=2,
            arrowsize=1,
            arrowwidth=1,
            arrowcolor="black",
            ax=0,
            ay=-40,
        )

    if strategy_info["bep2"] is not None:
        fig.add_shape(
            type="line",
            x0=strategy_info["bep2"],
            x1=strategy_info["bep2"],
            y0=y_min,
            y1=y_max,
            line=dict(dash="dot", color="black", width=1),
        )

        fig.add_annotation(
            x=strategy_info["bep2"],
            y=0,
            text=f"BEP: {strategy_info['bep2']:.2f}",
            showarrow=True,
            arrowhead=2,
            arrowsize=1,
            arrowwidth=1,
            arrowcolor="black",
            ax=0,
            ay=-40,
        )

    fig.update_layout(
        title="Option Strategy P/L",
        xaxis=dict(
            title="Underlying Price",
            showgrid=True,
            gridcolor="rgba(200, 200, 200, 0.3)",
            showspikes=True,
            spikethickness=0.6,
            spikecolor="rgba(120, 120, 120, 0.7)",
            spikedash="solid",
        ),
        yaxis=dict(
            title="P/L",
            range=[y_min, y_max],
            zeroline=True,
            zerolinecolor="black",
            zerolinewidth=1,
            showgrid=True,
            gridcolor="rgba(200, 200, 200, 0.3)",
        ),
        yaxis2=dict(
            title="Greeks", overlaying="y", side="right", zeroline=False, showgrid=False
        ),
        hovermode="x unified",
        hoverlabel=dict(bgcolor="white", font_size=12, font_family="Arial"),
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=60, r=60, t=50, b=60),
        paper_bgcolor="white",
        plot_bgcolor="white",
        height=500,
    )

    return fig


def main():
    st.sidebar.title("Option Calculator")

    ticker = st.sidebar.text_input("Enter Ticker Symbol:", "AAPL").upper()

    fetch_button = st.sidebar.button("Fetch Data")

    if fetch_button:
        with st.sidebar:
            with st.spinner("Fetching data..."):
                data = fetch_data(ticker)

                if data:
                    st.session_state.data = data
                    st.success(f"Successfully fetched data for {ticker}")
                else:
                    st.error("Failed to fetch data. Please check the ticker symbol.")

    if "data" in st.session_state:
        data = st.session_state.data

        with st.sidebar:
            st.markdown(f"### {data['name']} ({ticker})")
            st.markdown(f"**Sector:** {data['sector']}")

            st.markdown("#### Stock Information")
            col1, col2 = st.columns(2)

            with col1:
                st.metric(
                    "Price", f"${data['underlying_price']:.2f}", f"{data['chg']:.2f}%"
                )
                st.metric("High", f"${data['high']:.2f}")

            with col2:
                st.metric("Open", f"${data['open_price']:.2f}")
                st.metric("Low", f"${data['low']:.2f}")

            st.metric("Volatility (52w)", f"{data['vol']:.2f}%")

    tab1, tab3, tab2 = st.tabs(
        ["Option Calculator & Strategy", "Option Chain", "About"]
    )
    with tab1:
        if "data" in st.session_state:
            data = st.session_state.data

            st.subheader("Option Parameters")

            # CSS 스타일을 먼저 적용 (컬럼 외부에 적용)
            st.markdown(
                """
            <style>
            /* 전체 폰트 굵게 설정 */
            .streamlit-container {
                font-weight: bold;
            }
            .positive { color: #1cd4c8; font-weight: bold; }
            .negative { color: #d41c78; font-weight: bold; }

            /* 파라미터 레이블 스타일 */
            .param-label {
                font-size: 16px;
                font-weight: bold;
                margin-bottom: 10px;
            }

            /* 옵션 레이블 스타일 */
            .option-label {
                font-weight: bold;
                margin-bottom: 0px !important;
                padding-bottom: 0px !important;
            }

            
            </style>
            """,
                unsafe_allow_html=True,
            )

            # 이제 5개 컬럼 유지 - 하지만 5번째 컬럼은 빈 컬럼으로 남겨둠
            param_cols = st.columns(5)

            with param_cols[0]:
                s = st.number_input(
                    "Underlying Price (S)",
                    value=float(data["underlying_price"]),
                    step=0.01,
                    format="%.2f",
                )

                expiry = st.selectbox(
                    "Expiry Date",
                    options=data["expiry_dates"] if "expiry_dates" in data else [],
                )

                if expiry:
                    tau = calculate_days_to_expiry(expiry)
                    st.number_input(
                        "DTE (days)",
                        value=tau,
                        disabled=True,
                        label_visibility="visible",
                    )
                else:
                    tau = st.number_input(
                        "DTE (days)", value=30, min_value=1, label_visibility="visible"
                    )

                rf = st.number_input(
                    "Risk-free Rate (%)",
                    value=float(data["rf"]),
                    step=0.01,
                    format="%.2f",
                )

                y = st.number_input(
                    "Dividend Yield (%)",
                    value=float(data["dividend_yield"]),
                    step=0.01,
                    format="%.2f",
                )

            with param_cols[1]:
                side = st.selectbox("Side", options=["LONG", "SHORT"])
                option_type = st.selectbox("Option Type", options=["CALL", "PUT"])

                strategy = st.selectbox(
                    "Strategy",
                    options=[
                        "Single",
                        "Covered",
                        "Protective",
                        "Spread",
                        "Straddle",
                        "Strangle",
                        "Strip",
                        "Strap",
                        "Butterfly",
                        "Ladder",
                        "Jade Lizard",
                        "Reverse Jade Lizard",
                        "Condor",
                    ],
                )

                greeks = st.selectbox(
                    "Greeks", options=["Delta", "Gamma", "Vega", "Theta", "Rho"]
                )
                sigma = st.number_input(
                    "Volatility (%)",
                    value=float(data["vol"]),
                    step=0.1,
                    format="%.1f",
                    label_visibility="visible",
                )

            with param_cols[2]:
                if strategy in ["Strangle", "Spread"]:
                    if expiry:
                        calls, puts = get_option_chain(ticker, expiry)
                        if calls is not None and puts is not None:
                            all_strikes = sorted(
                                set(calls["strike"].tolist() + puts["strike"].tolist())
                            )

                            k1 = st.selectbox(
                                "Strike Price (k1; Lower)",
                                options=all_strikes,
                                index=min(len(all_strikes) // 2 - 1, 0),
                            )
                            k2 = st.selectbox(
                                "Strike Price (k2; Higher)",
                                options=all_strikes,
                                index=min(
                                    len(all_strikes) // 2 + 1, len(all_strikes) - 1
                                ),
                            )
                            k3, k4 = None, None
                        else:
                            k1 = st.number_input(
                                "Strike Price (k1; Lower)",
                                value=s * 0.95,
                                step=0.5,
                                format="%.1f",
                                label_visibility="visible",
                            )
                            k2 = st.number_input(
                                "Strike Price (k2; Higher)",
                                value=s * 1.05,
                                step=0.5,
                                format="%.1f",
                                label_visibility="visible",
                            )
                            k3, k4 = None, None
                    else:
                        k1 = st.number_input(
                            "Strike Price (k1; Lower)",
                            value=s * 0.95,
                            step=0.5,
                            format="%.1f",
                            label_visibility="visible",
                        )
                        k2 = st.number_input(
                            "Strike Price (k2; Higher)",
                            value=s * 1.05,
                            step=0.5,
                            format="%.1f",
                            label_visibility="visible",
                        )
                        k3, k4 = None, None

                elif strategy in [
                    "Butterfly",
                    "Ladder",
                    "Jade Lizard",
                    "Reverse Jade Lizard",
                ]:
                    if expiry:
                        calls, puts = get_option_chain(ticker, expiry)
                        if calls is not None and puts is not None:
                            all_strikes = sorted(
                                set(calls["strike"].tolist() + puts["strike"].tolist())
                            )

                            k1 = st.selectbox(
                                "Strike (k1; Lower)",
                                options=all_strikes,
                                index=min(len(all_strikes) // 2 - 2, 0),
                            )
                            k2 = st.selectbox(
                                "Strike (k2; Middle)",
                                options=all_strikes,
                                index=min(len(all_strikes) // 2, len(all_strikes) - 1),
                            )
                            k3 = st.selectbox(
                                "Strike (k3; Higher)",
                                options=all_strikes,
                                index=min(
                                    len(all_strikes) // 2 + 2, len(all_strikes) - 1
                                ),
                            )
                            k4 = None
                        else:
                            k1 = st.number_input(
                                "Strike (k1; Lower)",
                                value=s * 0.9,
                                step=0.5,
                                format="%.1f",
                                label_visibility="visible",
                            )
                            k2 = st.number_input(
                                "Strike (k2; Middle)",
                                value=s,
                                step=0.5,
                                format="%.1f",
                                label_visibility="visible",
                            )
                            k3 = st.number_input(
                                "Strike (k3; Higher)",
                                value=s * 1.1,
                                step=0.5,
                                format="%.1f",
                                label_visibility="visible",
                            )
                            k4 = None
                    else:
                        k1 = st.number_input(
                            "Strike (k1; Lower)",
                            value=s * 0.9,
                            step=0.5,
                            format="%.1f",
                            label_visibility="visible",
                        )
                        k2 = st.number_input(
                            "Strike (k2; Middle)",
                            value=s,
                            step=0.5,
                            format="%.1f",
                            label_visibility="visible",
                        )
                        k3 = st.number_input(
                            "Strike (k3; Higher)",
                            value=s * 1.1,
                            step=0.5,
                            format="%.1f",
                            label_visibility="visible",
                        )
                        k4 = None

                elif strategy == "Condor":
                    if expiry:
                        calls, puts = get_option_chain(ticker, expiry)
                        if calls is not None and puts is not None:
                            all_strikes = sorted(
                                set(calls["strike"].tolist() + puts["strike"].tolist())
                            )

                            k1 = st.selectbox(
                                "Strike (k1; Lowest)",
                                options=all_strikes,
                                index=min(len(all_strikes) // 2 - 3, 0),
                            )
                            k2 = st.selectbox(
                                "Strike (k2; Lower-Mid)",
                                options=all_strikes,
                                index=min(
                                    len(all_strikes) // 2 - 1, len(all_strikes) - 3
                                ),
                            )
                            k3 = st.selectbox(
                                "Strike (k3; Upper-Mid)",
                                options=all_strikes,
                                index=min(
                                    len(all_strikes) // 2 + 1, len(all_strikes) - 2
                                ),
                            )
                            k4 = st.selectbox(
                                "Strike (k4; Highest)",
                                options=all_strikes,
                                index=min(
                                    len(all_strikes) // 2 + 3, len(all_strikes) - 1
                                ),
                            )
                        else:
                            k1 = st.number_input(
                                "Strike (k1; Lowest)",
                                value=s * 0.85,
                                step=0.5,
                                format="%.1f",
                                label_visibility="visible",
                            )
                            k2 = st.number_input(
                                "Strike (k2; Lower-Mid)",
                                value=s * 0.95,
                                step=0.5,
                                format="%.1f",
                                label_visibility="visible",
                            )
                            k3 = st.number_input(
                                "Strike (k3; Upper-Mid)",
                                value=s * 1.05,
                                step=0.5,
                                format="%.1f",
                                label_visibility="visible",
                            )
                            k4 = st.number_input(
                                "Strike (k4; Highest)",
                                value=s * 1.15,
                                step=0.5,
                                format="%.1f",
                                label_visibility="visible",
                            )
                    else:
                        k1 = st.number_input(
                            "Strike (k1; Lowest)",
                            value=s * 0.85,
                            step=0.5,
                            format="%.1f",
                            label_visibility="visible",
                        )
                        k2 = st.number_input(
                            "Strike (k2; Lower-Mid)",
                            value=s * 0.95,
                            step=0.5,
                            format="%.1f",
                            label_visibility="visible",
                        )
                        k3 = st.number_input(
                            "Strike (k3; Upper-Mid)",
                            value=s * 1.05,
                            step=0.5,
                            format="%.1f",
                            label_visibility="visible",
                        )
                        k4 = st.number_input(
                            "Strike (k4; Highest)",
                            value=s * 1.15,
                            step=0.5,
                            format="%.1f",
                            label_visibility="visible",
                        )
                else:
                    if expiry:
                        calls, puts = get_option_chain(ticker, expiry)
                        if calls is not None and puts is not None:
                            all_strikes = sorted(
                                set(calls["strike"].tolist() + puts["strike"].tolist())
                            )
                            k1 = st.selectbox(
                                "Strike Price (k)",
                                options=all_strikes,
                                index=len(all_strikes) // 2,
                            )
                            k2, k3, k4 = k1, None, None
                        else:
                            k1 = st.number_input(
                                "Strike Price (k)",
                                value=s,
                                step=0.5,
                                format="%.1f",
                                label_visibility="visible",
                            )
                            k2, k3, k4 = k1, None, None
                    else:
                        k1 = st.number_input(
                            "Strike Price (k)",
                            value=s,
                            step=0.5,
                            format="%.1f",
                            label_visibility="visible",
                        )
                        k2, k3, k4 = k1, None, None

                size = st.number_input(
                    "Size (@100)", value=1, min_value=1, label_visibility="visible"
                )

            # Option Prices 섹션을 4번째 컬럼으로 이동
            with param_cols[3]:
                st.markdown(
                    "<div class='param-label'>Option Prices</div>",
                    unsafe_allow_html=True,
                )

                if strategy == "Single":
                    if option_type == "CALL":
                        sign = "+" if side == "LONG" else "-"
                        css_class = "positive" if side == "LONG" else "negative"
                        st.markdown(
                            f"<div class='option-label'><span class='{css_class}'>{sign}</span> Call (k)</div>",
                            unsafe_allow_html=True,
                        )
                        price1 = st.number_input(
                            "",
                            value=bs_model(s, k1, rf, tau, sigma, y, "c"),
                            step=0.01,
                            format="%.2f",
                            key="single_call_price",
                            label_visibility="collapsed",
                        )
                        price2, price3, price4 = None, None, None
                    else:
                        sign = "+" if side == "LONG" else "-"
                        css_class = "positive" if side == "LONG" else "negative"
                        st.markdown(
                            f"<div class='option-label'><span class='{css_class}'>{sign}</span> Put (k)</div>",
                            unsafe_allow_html=True,
                        )
                        price1 = st.number_input(
                            "",
                            value=bs_model(s, k1, rf, tau, sigma, y, "p"),
                            step=0.01,
                            format="%.2f",
                            key="single_put_price",
                            label_visibility="collapsed",
                        )
                        price2, price3, price4 = None, None, None

                elif strategy == "Spread":
                    if option_type == "CALL":
                        sign1 = "+" if side == "LONG" else "-"
                        css_class1 = "positive" if side == "LONG" else "negative"
                        st.markdown(
                            f"<div class='option-label'><span class='{css_class1}'>{sign1}</span> Call (k1)</div>",
                            unsafe_allow_html=True,
                        )
                        price1 = st.number_input(
                            "",
                            value=bs_model(s, k1, rf, tau, sigma, y, "c"),
                            step=0.01,
                            format="%.2f",
                            key="spread_call_price1",
                            label_visibility="collapsed",
                        )

                        sign2 = "+" if side == "SHORT" else "-"
                        css_class2 = "positive" if side == "SHORT" else "negative"
                        st.markdown(
                            f"<div class='option-label'><span class='{css_class2}'>{sign2}</span> Call (k2)</div>",
                            unsafe_allow_html=True,
                        )
                        price2 = st.number_input(
                            "",
                            value=bs_model(s, k2, rf, tau, sigma, y, "c"),
                            step=0.01,
                            format="%.2f",
                            key="spread_call_price2",
                            label_visibility="collapsed",
                        )
                        price3, price4 = None, None
                    else:
                        sign1 = "+" if side == "SHORT" else "-"
                        css_class1 = "positive" if side == "SHORT" else "negative"
                        st.markdown(
                            f"<div class='option-label'><span class='{css_class1}'>{sign1}</span> Put (k1)</div>",
                            unsafe_allow_html=True,
                        )
                        price1 = st.number_input(
                            "",
                            value=bs_model(s, k1, rf, tau, sigma, y, "p"),
                            step=0.01,
                            format="%.2f",
                            key="spread_put_price1",
                            label_visibility="collapsed",
                        )

                        sign2 = "+" if side == "LONG" else "-"
                        css_class2 = "positive" if side == "LONG" else "negative"
                        st.markdown(
                            f"<div class='option-label'><span class='{css_class2}'>{sign2}</span> Put (k2)</div>",
                            unsafe_allow_html=True,
                        )
                        price2 = st.number_input(
                            "",
                            value=bs_model(s, k2, rf, tau, sigma, y, "p"),
                            step=0.01,
                            format="%.2f",
                            key="spread_put_price2",
                            label_visibility="collapsed",
                        )
                        price3, price4 = None, None

                elif strategy == "Straddle":
                    sign = "+" if side == "LONG" else "-"
                    css_class = "positive" if side == "LONG" else "negative"

                    st.markdown(
                        f"<div class='option-label'><span class='{css_class}'>{sign}</span> Put (k)</div>",
                        unsafe_allow_html=True,
                    )
                    price1 = st.number_input(
                        "",
                        value=bs_model(s, k1, rf, tau, sigma, y, "p"),
                        step=0.01,
                        format="%.2f",
                        key="straddle_put_price",
                        label_visibility="collapsed",
                    )

                    st.markdown(
                        f"<div class='option-label'><span class='{css_class}'>{sign}</span> Call (k)</div>",
                        unsafe_allow_html=True,
                    )
                    price2 = st.number_input(
                        "",
                        value=bs_model(s, k1, rf, tau, sigma, y, "c"),
                        step=0.01,
                        format="%.2f",
                        key="straddle_call_price",
                        label_visibility="collapsed",
                    )
                    price3, price4 = None, None

                elif strategy == "Strangle":
                    sign = "+" if side == "LONG" else "-"
                    css_class = "positive" if side == "LONG" else "negative"

                    st.markdown(
                        f"<div class='option-label'><span class='{css_class}'>{sign}</span> Put (k1)</div>",
                        unsafe_allow_html=True,
                    )
                    price1 = st.number_input(
                        "",
                        value=bs_model(s, k1, rf, tau, sigma, y, "p"),
                        step=0.01,
                        format="%.2f",
                        key="strangle_put_price",
                        label_visibility="collapsed",
                    )

                    st.markdown(
                        f"<div class='option-label'><span class='{css_class}'>{sign}</span> Call (k2)</div>",
                        unsafe_allow_html=True,
                    )
                    price2 = st.number_input(
                        "",
                        value=bs_model(s, k2, rf, tau, sigma, y, "c"),
                        step=0.01,
                        format="%.2f",
                        key="strangle_call_price",
                        label_visibility="collapsed",
                    )
                    price3, price4 = None, None

                elif strategy == "Strip":
                    st.markdown(
                        "<div class='option-label'><span class='positive'>+</span> 2x Put (k)</div>",
                        unsafe_allow_html=True,
                    )
                    price1 = st.number_input(
                        "",
                        value=bs_model(s, k1, rf, tau, sigma, y, "p"),
                        step=0.01,
                        format="%.2f",
                        key="strip_put_price",
                        label_visibility="collapsed",
                    )

                    st.markdown(
                        "<div class='option-label'><span class='positive'>+</span> Call (k)</div>",
                        unsafe_allow_html=True,
                    )
                    price2 = st.number_input(
                        "",
                        value=bs_model(s, k1, rf, tau, sigma, y, "c"),
                        step=0.01,
                        format="%.2f",
                        key="strip_call_price",
                        label_visibility="collapsed",
                    )
                    price3, price4 = None, None

                elif strategy == "Strap":
                    st.markdown(
                        "<div class='option-label'><span class='positive'>+</span> Put (k)</div>",
                        unsafe_allow_html=True,
                    )
                    price1 = st.number_input(
                        "",
                        value=bs_model(s, k1, rf, tau, sigma, y, "p"),
                        step=0.01,
                        format="%.2f",
                        key="strap_put_price",
                        label_visibility="collapsed",
                    )

                    st.markdown(
                        "<div class='option-label'><span class='positive'>+</span> 2x Call (k)</div>",
                        unsafe_allow_html=True,
                    )
                    price2 = st.number_input(
                        "",
                        value=bs_model(s, k1, rf, tau, sigma, y, "c"),
                        step=0.01,
                        format="%.2f",
                        key="strap_call_price",
                        label_visibility="collapsed",
                    )
                    price3, price4 = None, None

                elif strategy == "Butterfly":
                    if option_type == "CALL":
                        sign1 = "+" if side == "LONG" else "-"
                        css_class1 = "positive" if side == "LONG" else "negative"
                        st.markdown(
                            f"<div class='option-label'><span class='{css_class1}'>{sign1}</span> Call (k1)</div>",
                            unsafe_allow_html=True,
                        )
                        price1 = st.number_input(
                            "",
                            value=bs_model(s, k1, rf, tau, sigma, y, "c"),
                            step=0.01,
                            format="%.2f",
                            key="butterfly_call_price1",
                            label_visibility="collapsed",
                        )

                        sign2 = "+" if side == "SHORT" else "-"
                        css_class2 = "positive" if side == "SHORT" else "negative"
                        st.markdown(
                            f"<div class='option-label'><span class='{css_class2}'>{sign2}</span> 2x Call (k2)</div>",
                            unsafe_allow_html=True,
                        )
                        price2 = st.number_input(
                            "",
                            value=bs_model(s, k2, rf, tau, sigma, y, "c"),
                            step=0.01,
                            format="%.2f",
                            key="butterfly_call_price2",
                            label_visibility="collapsed",
                        )

                        sign3 = "+" if side == "LONG" else "-"
                        css_class3 = "positive" if side == "LONG" else "negative"
                        st.markdown(
                            f"<div class='option-label'><span class='{css_class3}'>{sign3}</span> Call (k3)</div>",
                            unsafe_allow_html=True,
                        )
                        price3 = st.number_input(
                            "",
                            value=bs_model(s, k3, rf, tau, sigma, y, "c"),
                            step=0.01,
                            format="%.2f",
                            key="butterfly_call_price3",
                            label_visibility="collapsed",
                        )
                        price4 = None
                    else:
                        sign1 = "+" if side == "LONG" else "-"
                        css_class1 = "positive" if side == "LONG" else "negative"
                        st.markdown(
                            f"<div class='option-label'><span class='{css_class1}'>{sign1}</span> Put (k1)</div>",
                            unsafe_allow_html=True,
                        )
                        price1 = st.number_input(
                            "",
                            value=bs_model(s, k1, rf, tau, sigma, y, "p"),
                            step=0.01,
                            format="%.2f",
                            key="butterfly_put_price1",
                            label_visibility="collapsed",
                        )

                        sign2 = "+" if side == "SHORT" else "-"
                        css_class2 = "positive" if side == "SHORT" else "negative"
                        st.markdown(
                            f"<div class='option-label'><span class='{css_class2}'>{sign2}</span> 2x Put (k2)</div>",
                            unsafe_allow_html=True,
                        )
                        price2 = st.number_input(
                            "",
                            value=bs_model(s, k2, rf, tau, sigma, y, "p"),
                            step=0.01,
                            format="%.2f",
                            key="butterfly_put_price2",
                            label_visibility="collapsed",
                        )

                        sign3 = "+" if side == "LONG" else "-"
                        css_class3 = "positive" if side == "LONG" else "negative"
                        st.markdown(
                            f"<div class='option-label'><span class='{css_class3}'>{sign3}</span> Put (k3)</div>",
                            unsafe_allow_html=True,
                        )
                        price3 = st.number_input(
                            "",
                            value=bs_model(s, k3, rf, tau, sigma, y, "p"),
                            step=0.01,
                            format="%.2f",
                            key="butterfly_put_price3",
                            label_visibility="collapsed",
                        )
                        price4 = None

                elif strategy == "Ladder":
                    if option_type == "CALL":
                        sign1 = "+" if side == "LONG" else "-"
                        css_class1 = "positive" if side == "LONG" else "negative"
                        st.markdown(
                            f"<div class='option-label'><span class='{css_class1}'>{sign1}</span> Call (k1)</div>",
                            unsafe_allow_html=True,
                        )
                        price1 = st.number_input(
                            "",
                            value=bs_model(s, k1, rf, tau, sigma, y, "c"),
                            step=0.01,
                            format="%.2f",
                            key="ladder_call_price1",
                            label_visibility="collapsed",
                        )

                        sign2 = "+" if side == "SHORT" else "-"
                        css_class2 = "positive" if side == "SHORT" else "negative"
                        st.markdown(
                            f"<div class='option-label'><span class='{css_class2}'>{sign2}</span> Call (k2)</div>",
                            unsafe_allow_html=True,
                        )
                        price2 = st.number_input(
                            "",
                            value=bs_model(s, k2, rf, tau, sigma, y, "c"),
                            step=0.01,
                            format="%.2f",
                            key="ladder_call_price2",
                            label_visibility="collapsed",
                        )

                        sign3 = "+" if side == "SHORT" else "-"
                        css_class3 = "positive" if side == "SHORT" else "negative"
                        st.markdown(
                            f"<div class='option-label'><span class='{css_class3}'>{sign3}</span> Call (k3)</div>",
                            unsafe_allow_html=True,
                        )
                        price3 = st.number_input(
                            "",
                            value=bs_model(s, k3, rf, tau, sigma, y, "c"),
                            step=0.01,
                            format="%.2f",
                            key="ladder_call_price3",
                            label_visibility="collapsed",
                        )
                        price4 = None
                    else:
                        sign1 = "+" if side == "SHORT" else "-"
                        css_class1 = "positive" if side == "SHORT" else "negative"
                        st.markdown(
                            f"<div class='option-label'><span class='{css_class1}'>{sign1}</span> Put (k1)</div>",
                            unsafe_allow_html=True,
                        )
                        price1 = st.number_input(
                            "",
                            value=bs_model(s, k1, rf, tau, sigma, y, "p"),
                            step=0.01,
                            format="%.2f",
                            key="ladder_put_price1",
                            label_visibility="collapsed",
                        )

                        sign2 = "+" if side == "SHORT" else "-"
                        css_class2 = "positive" if side == "SHORT" else "negative"
                        st.markdown(
                            f"<div class='option-label'><span class='{css_class2}'>{sign2}</span> Put (k2)</div>",
                            unsafe_allow_html=True,
                        )
                        price2 = st.number_input(
                            "",
                            value=bs_model(s, k2, rf, tau, sigma, y, "p"),
                            step=0.01,
                            format="%.2f",
                            key="ladder_put_price2",
                            label_visibility="collapsed",
                        )

                        sign3 = "+" if side == "LONG" else "-"
                        css_class3 = "positive" if side == "LONG" else "negative"
                        st.markdown(
                            f"<div class='option-label'><span class='{css_class3}'>{sign3}</span> Put (k3)</div>",
                            unsafe_allow_html=True,
                        )
                        price3 = st.number_input(
                            "",
                            value=bs_model(s, k3, rf, tau, sigma, y, "p"),
                            step=0.01,
                            format="%.2f",
                            key="ladder_put_price3",
                            label_visibility="collapsed",
                        )
                        price4 = None

                elif strategy == "Jade Lizard":
                    st.markdown(
                        "<div class='option-label'><span class='negative'>-</span> Put (k1)</div>",
                        unsafe_allow_html=True,
                    )
                    price1 = st.number_input(
                        "",
                        value=bs_model(s, k1, rf, tau, sigma, y, "p"),
                        step=0.01,
                        format="%.2f",
                        key="jade_lizard_put_price",
                        label_visibility="collapsed",
                    )

                    st.markdown(
                        "<div class='option-label'><span class='negative'>-</span> Call (k2)</div>",
                        unsafe_allow_html=True,
                    )
                    price2 = st.number_input(
                        "",
                        value=bs_model(s, k2, rf, tau, sigma, y, "c"),
                        step=0.01,
                        format="%.2f",
                        key="jade_lizard_call_price1",
                        label_visibility="collapsed",
                    )

                    st.markdown(
                        "<div class='option-label'><span class='positive'>+</span> Call (k3)</div>",
                        unsafe_allow_html=True,
                    )
                    price3 = st.number_input(
                        "",
                        value=bs_model(s, k3, rf, tau, sigma, y, "c"),
                        step=0.01,
                        format="%.2f",
                        key="jade_lizard_call_price2",
                        label_visibility="collapsed",
                    )
                    price4 = None

                elif strategy == "Reverse Jade Lizard":
                    st.markdown(
                        "<div class='option-label'><span class='positive'>+</span> Put (k1)</div>",
                        unsafe_allow_html=True,
                    )
                    price1 = st.number_input(
                        "",
                        value=bs_model(s, k1, rf, tau, sigma, y, "p"),
                        step=0.01,
                        format="%.2f",
                        key="rev_jade_lizard_put_price1",
                        label_visibility="collapsed",
                    )

                    st.markdown(
                        "<div class='option-label'><span class='negative'>-</span> Put (k2)</div>",
                        unsafe_allow_html=True,
                    )
                    price2 = st.number_input(
                        "",
                        value=bs_model(s, k2, rf, tau, sigma, y, "p"),
                        step=0.01,
                        format="%.2f",
                        key="rev_jade_lizard_put_price2",
                        label_visibility="collapsed",
                    )

                    st.markdown(
                        "<div class='option-label'><span class='negative'>-</span> Call (k3)</div>",
                        unsafe_allow_html=True,
                    )
                    price3 = st.number_input(
                        "",
                        value=bs_model(s, k3, rf, tau, sigma, y, "c"),
                        step=0.01,
                        format="%.2f",
                        key="rev_jade_lizard_call_price",
                        label_visibility="collapsed",
                    )
                    price4 = None

                elif strategy == "Condor":
                    if option_type == "CALL":
                        sign1 = "+" if side == "LONG" else "-"
                        css_class1 = "positive" if side == "LONG" else "negative"
                        st.markdown(
                            f"<div class='option-label'><span class='{css_class1}'>{sign1}</span> Call (k1)</div>",
                            unsafe_allow_html=True,
                        )
                        price1 = st.number_input(
                            "",
                            value=bs_model(s, k1, rf, tau, sigma, y, "c"),
                            step=0.01,
                            format="%.2f",
                            key="condor_call_price1",
                            label_visibility="collapsed",
                        )

                        sign2 = "+" if side == "SHORT" else "-"
                        css_class2 = "positive" if side == "SHORT" else "negative"
                        st.markdown(
                            f"<div class='option-label'><span class='{css_class2}'>{sign2}</span> Call (k2)</div>",
                            unsafe_allow_html=True,
                        )
                        price2 = st.number_input(
                            "",
                            value=bs_model(s, k2, rf, tau, sigma, y, "c"),
                            step=0.01,
                            format="%.2f",
                            key="condor_call_price2",
                            label_visibility="collapsed",
                        )

                        sign3 = "+" if side == "SHORT" else "-"
                        css_class3 = "positive" if side == "SHORT" else "negative"
                        st.markdown(
                            f"<div class='option-label'><span class='{css_class3}'>{sign3}</span> Call (k3)</div>",
                            unsafe_allow_html=True,
                        )
                        price3 = st.number_input(
                            "",
                            value=bs_model(s, k3, rf, tau, sigma, y, "c"),
                            step=0.01,
                            format="%.2f",
                            key="condor_call_price3",
                            label_visibility="collapsed",
                        )

                        sign4 = "+" if side == "LONG" else "-"
                        css_class4 = "positive" if side == "LONG" else "negative"
                        st.markdown(
                            f"<div class='option-label'><span class='{css_class4}'>{sign4}</span> Call (k4)</div>",
                            unsafe_allow_html=True,
                        )
                        price4 = st.number_input(
                            "",
                            value=bs_model(s, k4, rf, tau, sigma, y, "c"),
                            step=0.01,
                            format="%.2f",
                            key="condor_call_price4",
                            label_visibility="collapsed",
                        )
                    else:
                        sign1 = "+" if side == "LONG" else "-"
                        css_class1 = "positive" if side == "LONG" else "negative"
                        st.markdown(
                            f"<div class='option-label'><span class='{css_class1}'>{sign1}</span> Put (k1)</div>",
                            unsafe_allow_html=True,
                        )
                        price1 = st.number_input(
                            "",
                            value=bs_model(s, k1, rf, tau, sigma, y, "p"),
                            step=0.01,
                            format="%.2f",
                            key="condor_put_price1",
                            label_visibility="collapsed",
                        )

                        sign2 = "+" if side == "SHORT" else "-"
                        css_class2 = "positive" if side == "SHORT" else "negative"
                        st.markdown(
                            f"<div class='option-label'><span class='{css_class2}'>{sign2}</span> Put (k2)</div>",
                            unsafe_allow_html=True,
                        )
                        price2 = st.number_input(
                            "",
                            value=bs_model(s, k2, rf, tau, sigma, y, "p"),
                            step=0.01,
                            format="%.2f",
                            key="condor_put_price2",
                            label_visibility="collapsed",
                        )

                        sign3 = "+" if side == "SHORT" else "-"
                        css_class3 = "positive" if side == "SHORT" else "negative"
                        st.markdown(
                            f"<div class='option-label'><span class='{css_class3}'>{sign3}</span> Put (k3)</div>",
                            unsafe_allow_html=True,
                        )
                        price3 = st.number_input(
                            "",
                            value=bs_model(s, k3, rf, tau, sigma, y, "p"),
                            step=0.01,
                            format="%.2f",
                            key="condor_put_price3",
                            label_visibility="collapsed",
                        )

                        sign4 = "+" if side == "LONG" else "-"
                        css_class4 = "positive" if side == "LONG" else "negative"
                        st.markdown(
                            f"<div class='option-label'><span class='{css_class4}'>{sign4}</span> Put (k4)</div>",
                            unsafe_allow_html=True,
                        )
                        price4 = st.number_input(
                            "",
                            value=bs_model(s, k4, rf, tau, sigma, y, "p"),
                            step=0.01,
                            format="%.2f",
                            key="condor_put_price4",
                            label_visibility="collapsed",
                        )

                elif strategy == "Covered":
                    if option_type == "CALL":
                        st.markdown(
                            "<div class='option-label'><span style='font-weight:bold;'>S</span> <span class='negative'>-</span> Call (k)</div>",
                            unsafe_allow_html=True,
                        )
                        price1 = st.number_input(
                            "",
                            value=bs_model(s, k1, rf, tau, sigma, y, "c"),
                            step=0.01,
                            format="%.2f",
                            key="covered_call_price",
                            label_visibility="collapsed",
                        )
                        price2, price3, price4 = s, None, None
                    else:
                        st.markdown(
                            "<div class='option-label'><span class='negative'>-</span><span style='font-weight:bold;'>S</span> <span class='negative'>-</span> Put (k)</div>",
                            unsafe_allow_html=True,
                        )
                        price1 = st.number_input(
                            "",
                            value=bs_model(s, k1, rf, tau, sigma, y, "p"),
                            step=0.01,
                            format="%.2f",
                            key="covered_put_price",
                            label_visibility="collapsed",
                        )
                        price2, price3, price4 = s, None, None

                elif strategy == "Protective":
                    if option_type == "CALL":
                        st.markdown(
                            "<div class='option-label'><span class='negative'>-</span><span style='font-weight:bold;'>S</span> <span class='positive'>+</span> Call (k)</div>",
                            unsafe_allow_html=True,
                        )
                        price1 = st.number_input(
                            "",
                            value=bs_model(s, k1, rf, tau, sigma, y, "c"),
                            step=0.01,
                            format="%.2f",
                            key="protective_call_price",
                            label_visibility="collapsed",
                        )
                        price2, price3, price4 = s, None, None
                    else:
                        st.markdown(
                            "<div class='option-label'><span style='font-weight:bold;'>S</span> <span class='positive'>+</span> Put (k)</div>",
                            unsafe_allow_html=True,
                        )
                        price1 = st.number_input(
                            "",
                            value=bs_model(s, k1, rf, tau, sigma, y, "p"),
                            step=0.01,
                            format="%.2f",
                            key="protective_put_price",
                            label_visibility="collapsed",
                        )
                        price2, price3, price4 = s, None, None

            # 5번째 컬럼은 비워둠 (필요한 경우 추가 정보 표시 가능)
            with param_cols[4]:
                # 이전에 가지고 있던 CSS 코드는 이제 위에서 처리됨
                pass

            st.markdown("---")

            st.subheader("Option Strategy Plot")

            if st.button("Show Plot"):
                with st.spinner("Calculating..."):
                    try:
                        df, strategy_info = calculate_payoff_df(
                            strategy=strategy,
                            side=side,
                            option_type=option_type,
                            s=s,
                            k1=k1,
                            k2=k2,
                            k3=k3,
                            k4=k4,
                            price1=price1,
                            price2=price2,
                            price3=price3,
                            price4=price4,
                            size=size,
                            tau=tau,
                            rf=rf,
                            sigma=sigma,
                            y=y,
                        )

                        col_info1, col_info2, col_info3 = st.columns(3)

                        with col_info1:
                            st.markdown(
                                f"### {side} {option_type if strategy not in ['Straddle', 'Strangle', 'Strip', 'Strap', 'Jade Lizard', 'Reverse Jade Lizard'] else ''} {strategy}"
                            )

                            strategy_color = {
                                "Bullish": "#f05f3e",
                                "Bearish": "#34b4eb",
                                "Neutral": "#59d9b5",
                            }.get(strategy_info["strategy_type"], "#808080")

                            risk_color = {
                                "High Risk": "#f03ec6",
                                "Moderate Risk": "#37ad8c",
                                "Low Risk": "#3ebaf0",
                            }.get(strategy_info["risk_level"], "#808080")

                            strategy_text = f"<code style='color:{strategy_color};'>{strategy_info['strategy_type']}</code>"
                            risk_text = f"<code style='color:{risk_color};'>{strategy_info['risk_level']}</code>"

                            st.markdown(
                                f"{strategy_text} | {risk_text}", unsafe_allow_html=True
                            )

                        fig = plot_option_strategy(df, s, greeks, strategy_info)
                        st.plotly_chart(fig, use_container_width=True)

                        st.subheader("Strategy Performance")
                        max_profit_text = (
                            "+∞"
                            if np.isinf(strategy_info["max_profit"])
                            else f"${strategy_info['max_profit']:.2f}"
                        )
                        min_profit_text = (
                            "-∞"
                            if np.isinf(strategy_info["min_profit"])
                            else f"${strategy_info['min_profit']:.2f}"
                        )
                        idx = np.abs(df["x"] - s).argmin()
                        current_pl = df["y"].iloc[idx].round(2)
                        col_p1, col_p2, col_p3, col_p4 = st.columns(4)
                        with col_p1:
                            st.metric("P/L (@S)", f"${current_pl:.2f}")
                        with col_p2:
                            st.metric("Max Profit", max_profit_text)
                        with col_p3:
                            st.metric("Max Loss", min_profit_text)
                        with col_p4:
                            bep_text = (
                                f"${strategy_info['bep1']:.2f}"
                                if strategy_info["bep1"] is not None
                                else "N/A"
                            )
                            if strategy_info["bep2"] is not None:
                                bep_text += f", ${strategy_info['bep2']:.2f}"
                            st.metric("Break-Even Point(s)", bep_text)

                        st.markdown("<br>", unsafe_allow_html=True)

                        st.subheader("Greeks at Current Price")
                        idx = np.abs(df["x"] - s).argmin()
                        delta = df["Delta"].iloc[idx]
                        gamma = df["Gamma"].iloc[idx]
                        vega = df["Vega"].iloc[idx]
                        theta = df["Theta"].iloc[idx]
                        rho = df["Rho"].iloc[idx]
                        col_g1, col_g2, col_g3, col_g4, col_g5 = st.columns(5)
                        with col_g1:
                            st.metric("Delta", f"{delta:.4f}")
                        with col_g2:
                            st.metric("Gamma", f"{gamma:.4f}")
                        with col_g3:
                            st.metric("Vega", f"{vega:.4f}")
                        with col_g4:
                            st.metric("Theta", f"{theta:.4f}")
                        with col_g5:
                            st.metric("Rho", f"{rho:.4f}")

                    except Exception as e:
                        st.error(f"Error creating plot: {str(e)}")
        else:
            st.info("Please enter a ticker symbol and fetch data in the sidebar first.")

    with tab3:
        st.subheader("Option Chain")

        if "data" in st.session_state:
            data = st.session_state.data

            expiry = st.selectbox(
                "Select Expiry Date",
                options=data["expiry_dates"] if "expiry_dates" in data else [],
            )

            if expiry:
                with st.spinner("Fetching option chain..."):
                    calls, puts = get_option_chain(ticker, expiry)
                    if calls is not None and puts is not None:
                        st.subheader("Call Option Chain")
                        st.dataframe(calls)
                        st.subheader("Put Option Chain")
                        st.dataframe(puts)
                    else:
                        st.warning(
                            "Option chain data not available for the selected expiry date."
                        )
            else:
                st.info("Please select an expiry date to view the option chain.")
        else:
            st.info("Please enter a ticker symbol and fetch data in the sidebar first.")

    with tab2:
        st.header("About Option Pricing Models")

        st.markdown(
            """
        ### Black-Scholes Model
        
        The theoretical option price was calculated using the **Black-Scholes model**. The Black-Scholes model is the most widely used option pricing model and was developed by Fisher Black and Myron Scholes to derive European option prices based on Einstein's Brownian motion equations.
        
        The formula used is shown below:
        
        $$d_1 = \\frac{\ln(S_0/K) + (r_f - y + 0.5 \\sigma^2) \\tau}{\\sigma \\sqrt{\\tau}}$$
        
        $$d_2 = d_1 - \\sigma \\sqrt{\\tau}$$
        
        **Call price:**
        
        $$C(S_0, \\tau) = S_0 N(d_1) e^{-y \\tau} - K e^{-r_f \\tau} N(d_2)$$
        
        **Put price:**
        
        $$P(S_0, \\tau) = K e^{-r_f \\tau} N(-d_2) - S_0 N(-d_1) e^{-y \\tau}$$
        
        *where,*
        - $S_0$ : Underlying Price
        - $K$ : Strike Price
        - $r_f$ : Risk-free rate
        - $y$ : Dividend yield
        - $τ$ : Time to maturity
        - $N(x)$ : Standard normal cumulative distribution function
        """
        )

        st.markdown(
            """
        ### Option Greeks
        
        Option Greeks are key measures that assess an option's price sensitivity to factors like volatility and the price of the underlying asset. They are crucial for analyzing options portfolios and are widely used by investors to make informed trading decisions.
        
        **Delta:**
        
        Delta measures how much an option's price will change for every 1 dollar movement in the underlying asset. A Delta of 0.40 means the option price will move 0.40 dollar for each $1 change, and suggests a 40% chance the option will expire in the money (ITM).
        
        Call options have a Delta between 0.00 and 1.00, with at-the-money options near 0.50, increasing toward 1.00 as they move deeper ITM or approach expiration, and decreasing toward 0.00 if they are out-of-the-money.
        
        Put options have a Delta between 0.00 and –1.00, with at-the-money options near –0.50, decreasing toward –1.00 as they move deeper ITM or approach expiration, and approaching 0.00 if they are out-of-the-money.
        
        $$\\text{Call: } \\Delta_c = e^{-y\\tau}N(d_1)$$
        
        $$\\text{Put: } \\Delta_p = e^{-y\\tau}[N(d_1)-1]$$
        
        **Gamma:**
        
        Gamma measures the rate of change in an option's Delta for every $1 move in the underlying asset, much like acceleration compared to speed. As Delta increases with a stock price move, Gamma reflects how much Delta shifts.
        
        $$\\Gamma = \\frac{e^{-y\\tau}}{S_0\\sigma\\sqrt{\\tau}}N'(d_1)$$
        
        **Vega:**
        
        Vega measures the change in an option's price for a one-percentage-point change in the implied volatility of the underlying asset.
        
        $$\\nu = S_0N'(d_1)\\sqrt{\\tau}$$
        
        **Theta:**
        
        Theta measures the daily decrease in an option's price as it approaches expiration, reflecting time decay.
        
        $$\\text{Call: } \\theta_c = 1/T(-(\\frac{S_0\\sigma e^{-y\\tau}}{2\\sqrt{\\tau}}N'(d_1)) - r_fKe^{r_f\\tau} N(d_2) + yS_0e^{-y\\tau}N(d_1))$$
        
        $$\\text{Put: } \\theta_p = 1/T(-(\\frac{S_0\\sigma e^{-y\\tau}}{2\\sqrt{\\tau}}N'(d_1)) + r_fKe^{r_f\\tau} N(-d_2) - yS_0e^{-y\\tau}N(-d_1))$$
        
        **Rho:**
        
        Rho measures the change in an option's price for a one-percentage-point shift in interest rates.
        
        $$\\text{Call: } \\rho_c = K\\tau e^{-r_f\\tau}N(d_2)$$
        
        $$\\text{Put: } \\rho_p = -K\\tau e^{-r_f\\tau}N(-d_2)$$
        """
        )

        st.markdown(
            """
        ### Option Strategies
        
        This calculator supports several option strategies:
        
        - **Single**: A basic long or short position in a call or put option.
        - **Covered**: Covered call (long stock + short call) or covered put (short stock + short put).
        - **Protective**: Protective put (long stock + long put) or protective call (short stock + long call).
        - **Spread**: Bull/bear spread using calls or puts with different strike prices.
        - **Straddle**: Long/short position in both a call and put with the same strike and expiration.
        - **Strangle**: Long/short position in both a call and put with different strikes but same expiration.
        - **Strip**: Long 1 call and 2 puts at the same strike (bearish strategy).
        - **Strap**: Long 2 calls and 1 put at the same strike (bullish strategy).
        - **Butterfly**: A three-leg strategy with limited risk and reward.
        - **Ladder**: A multi-leg strategy with strike prices at regular intervals.
        - **Jade Lizard**: Short put + short call spread (bullish strategy).
        - **Reverse Jade Lizard**: Long put spread + short call (bearish strategy).
        - **Condor**: A four-leg strategy with limited risk and reward.

        If you have any questions or suggestions, please contact: pmh621@naver.com 
        """
        )


if __name__ == "__main__":
    main()
