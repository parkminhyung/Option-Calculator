import plotly.graph_objects as go
import numpy as np
from strategies.probability import calculate_expiry_range, calculate_probability_curve, calculate_price_probability


def plot_option_strategy(df, s, greeks, strategy_info, tau=30, sigma=30, y=0):
    """
    옵션 전략의 손익 및 그릭스 그래프 생성
    
    Parameters:
    -----------
    df : DataFrame
        페이오프 및 그릭스 데이터
    s : float
        현재 기초자산 가격
    greeks : str
        표시할 그릭스
    strategy_info : dict
        전략 정보
    tau : int, optional
        만기까지 남은 일수
    sigma : float, optional
        변동성 (%)
    y : float, optional
        배당수익률 (%)
        
    Returns:
    --------
    Figure
        Plotly 그래프 객체
    """
    fig = go.Figure()

    # 이익/손실 영역 계산 및 표시 (원래 색상으로 복원)
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

    # 현재 가격 지점 찾기
    x_vals = df["x"].values
    y_vals = df["y"].values
    current_idx = np.abs(x_vals - s).argmin()
    current_pl = y_vals[current_idx]
    
    # 현재 P/L 포인트 추가 (파란색 점)
    fig.add_trace(
        go.Scatter(
            x=[s],
            y=[current_pl],
            mode="markers",
            name="Current P/L",
            marker=dict(color="blue", size=8),
            showlegend=True
        )
    )

    # 그릭스 표시
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
                line=dict(dash="dot", color="orange", width=1.5),
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
        line=dict(color="green", width=1.5, dash="dash"),
    )

    # 확률 곡선 추가 (파란색 점선 - 더 얇게)
    try:
        # 가격 범위에 대한 확률 곡선 계산
        prob_curve = calculate_probability_curve(s, x_vals, tau, sigma, y)
        
        # 시각화를 위해 스케일링
        prob_curve = (prob_curve / np.max(prob_curve)) * (y_max - y_min) * 0.4 + y_min
        
        # 확률 곡선 추가 (더 얇은 선으로)
        fig.add_trace(
            go.Scatter(
                x=x_vals,
                y=prob_curve,
                mode="lines",
                name="Probability",
                line=dict(color="blue", width=1, dash="dash"),  # 두께를 1로 줄임
                hoverinfo="x+y",
            )
        )
        
        # 확률 범위 (68% 신뢰구간)
        lower_bound, upper_bound = calculate_expiry_range(s, tau, sigma, confidence=0.68, dividend_yield=y)
        
        # 확률 범위 주석 대신 호버 정보로 제공
        prob_above = 100 * (1 - (s - lower_bound) / (upper_bound - lower_bound))
        prob_below = 100 - prob_above
        
        # 현재 가격 주석에 확률 정보 추가
        fig.add_annotation(
            x=s,
            y=y_max * 0.9,
            text=f"Current: {s:.2f}<br>↓ {prob_below:.1f}% • {prob_above:.1f}% ↑",
            showarrow=False,
            bgcolor="rgba(242, 242, 242, 0.7)",
            font=dict(color="black"),
            borderwidth=0,
            borderpad=5,
            yanchor="top"
        )
    except Exception as e:
        print(f"Error displaying probability curve: {e}")
        # 오류가 발생해도 계속 진행

    # BEP 표시 변경: 점으로만 표시
    if strategy_info["bep1"] is not None:
        # BEP 점으로 표시
        fig.add_trace(
            go.Scatter(
                x=[strategy_info["bep1"]],
                y=[0],  # BEP에서 y값은 0
                mode="markers",
                name="Break-Even",
                marker=dict(
                    color="gray",
                    size=10,
                    symbol="circle"
                ),
                showlegend=True
            )
        )

    if strategy_info["bep2"] is not None:
        # BEP 점으로 표시 (두 번째는 범례에 표시하지 않음)
        fig.add_trace(
            go.Scatter(
                x=[strategy_info["bep2"]],
                y=[0],  # BEP에서 y값은 0
                mode="markers",
                name="Break-Even",
                marker=dict(
                    color="gray",
                    size=10,
                    symbol="circle"
                ),
                showlegend=False
            )
        )

    # 호버 템플릿 설정
    hover_template = "<b>Price:</b> %{x:.2f}<br><b>P/L:</b> %{y:.2f}"
    for trace in fig.data:
        if 'hovertemplate' in trace:
            trace.hovertemplate = hover_template

    # 그래프 레이아웃 설정
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