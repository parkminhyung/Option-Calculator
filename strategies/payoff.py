import numpy as np
import pandas as pd
from models.greeks import option_greeks


def calculate_call_payoff(spot_price, strike, premium):
    """콜 옵션 페이오프 계산"""
    return np.where(spot_price <= strike, -premium, (spot_price - strike) - premium)


def calculate_put_payoff(spot_price, strike, premium):
    """풋 옵션 페이오프 계산"""
    return np.where(spot_price <= strike, (strike - spot_price) - premium, -premium)


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
    """
    옵션 전략에 따른 손익 및 그릭스 계산
    
    Parameters:
    -----------
    strategy : str
        옵션 전략 이름
    side : str
        'LONG' 또는 'SHORT'
    option_type : str
        'CALL' 또는 'PUT'
    s : float
        기초자산 가격
    k1, k2, k3, k4 : float
        각 포지션의 행사가격
    price1, price2, price3, price4 : float
        각 포지션의 옵션 가격
    size : int
        포지션 크기
    tau : int
        만기까지 남은 일수
    rf : float
        무위험 이자율 (%)
    sigma : float
        변동성 (%)
    y : float
        배당수익률 (%)
    option_chain : DataFrame, optional
        옵션 체인 데이터
        
    Returns:
    --------
    tuple
        (페이오프 데이터프레임, 전략 정보 딕셔너리)
    """
    k_values = [k for k in [k1, k2, k3, k4] if k is not None]
    if not k_values:
        return None

    k_min, k_max = min(k_values), max(k_values)
    price_range_min = max(0.1, s * 0.7, k_min * 0.7)
    price_range_max = max(s * 1.3, k_max * 1.3)

    x = np.arange(price_range_min, price_range_max, 0.1)
    df = pd.DataFrame({"x": x})

    # 그릭스 벡터 초기화
    call_delta1, call_delta2, call_delta3, call_delta4 = np.zeros(len(x)), np.zeros(len(x)), np.zeros(len(x)), np.zeros(len(x))
    put_delta1, put_delta2, put_delta3, put_delta4 = np.zeros(len(x)), np.zeros(len(x)), np.zeros(len(x)), np.zeros(len(x))
    gamma1, gamma2, gamma3, gamma4 = np.zeros(len(x)), np.zeros(len(x)), np.zeros(len(x)), np.zeros(len(x))
    vega1, vega2, vega3, vega4 = np.zeros(len(x)), np.zeros(len(x)), np.zeros(len(x)), np.zeros(len(x))
    call_theta1, call_theta2, call_theta3, call_theta4 = np.zeros(len(x)), np.zeros(len(x)), np.zeros(len(x)), np.zeros(len(x))
    put_theta1, put_theta2, put_theta3, put_theta4 = np.zeros(len(x)), np.zeros(len(x)), np.zeros(len(x)), np.zeros(len(x))
    call_rho1, call_rho2, call_rho3, call_rho4 = np.zeros(len(x)), np.zeros(len(x)), np.zeros(len(x)), np.zeros(len(x))
    put_rho1, put_rho2, put_rho3, put_rho4 = np.zeros(len(x)), np.zeros(len(x)), np.zeros(len(x)), np.zeros(len(x))

    # 각 행사가격에 대한 그릭스 계산
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

    # 각 행사가격에 대한 페이오프 계산
    c1 = calculate_call_payoff(x, k1, price1) if k1 is not None and price1 is not None else 0
    p1 = calculate_put_payoff(x, k1, price1) if k1 is not None and price1 is not None else 0

    c2 = calculate_call_payoff(x, k2, price2) if k2 is not None and price2 is not None else 0
    p2 = calculate_put_payoff(x, k2, price2) if k2 is not None and price2 is not None else 0

    c3 = calculate_call_payoff(x, k3, price3) if k3 is not None and price3 is not None else 0
    p3 = calculate_put_payoff(x, k3, price3) if k3 is not None and price3 is not None else 0

    c4 = calculate_call_payoff(x, k4, price4) if k4 is not None and price4 is not None else 0
    p4 = calculate_put_payoff(x, k4, price4) if k4 is not None and price4 is not None else 0

    # 전략별 손익 및 그릭스 계산
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

    # 포지션 크기 적용
    y_values = y_values * size
    delta = delta * size
    gamma = gamma * size
    vega = vega * size
    theta = theta * size
    rho = rho * size

    # 손익 데이터 저장
    df["y"] = y_values
    df["profit"] = np.where(y_values >= 0, y_values, np.nan)
    df["loss"] = np.where(y_values < 0, y_values, np.nan)
    df["Delta"] = delta
    df["Gamma"] = gamma
    df["Vega"] = vega
    df["Theta"] = theta
    df["Rho"] = rho

    # 손익 계산
    calculated_max_profit = np.max(y_values)
    calculated_min_profit = np.min(y_values)

    # 손익분기점 계산
    sign_changes = np.where(np.diff(np.signbit(y_values)))[0]
    bep1 = x[sign_changes[0]] if len(sign_changes) > 0 else None
    bep2 = x[sign_changes[1]] if len(sign_changes) > 1 else None

    # 전략별 최대/최소 손익 계산
    # (무한대 처리 등 기존 코드 유지)
    if strategy == "Single":
        if side == "LONG":
            max_profit = float("inf")
            min_profit = calculated_min_profit
        elif side == "SHORT":
            max_profit = calculated_max_profit
            min_profit = float("-inf")
    
    # ... 다른 전략들에 대한 손익 계산 ...
    
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

    else:
        max_profit = calculated_max_profit
        min_profit = calculated_min_profit

    # 전략 정보 반환
    strategy_info = {
        "strategy_type": strategy_type,
        "risk_level": risk_level,
        "bep1": bep1,
        "bep2": bep2,
        "max_profit": max_profit,
        "min_profit": min_profit,
    }

    return df, strategy_info
