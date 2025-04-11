import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st


def create_volatility_surface(vol_data):
    """
    3D 변동성 표면 생성
    
    Parameters:
    -----------
    vol_data : list of dict
        변동성 데이터 포인트
        
    Returns:
    --------
    Figure
        Plotly 3D 표면 그래프 객체
    """
    if not vol_data or len(vol_data) <= 1:
        return None
        
    try:
        # 데이터프레임으로 변환
        vol_df = pd.DataFrame(vol_data)

        # 콜 옵션 데이터프레임
        call_df = vol_df[vol_df["type"] == "Call"]
        # 풋 옵션 데이터프레임
        put_df = vol_df[vol_df["type"] == "Put"]

        # 충분한 데이터가 있는지 확인
        if len(call_df) <= 1 and len(put_df) <= 1:
            return None

        # 개별 표면 그래프로 변경
        fig = make_subplots(
            rows=1,
            cols=2,
            specs=[[{"type": "surface"}, {"type": "surface"}]],
            subplot_titles=(
                "Call Options Volatility Surface",
                "Put Options Volatility Surface",
            ),
        )

        # 충분한 데이터가 있는 경우에만 표면 추가
        if len(call_df) > 1:
            try:
                # 콜 옵션 데이터를 표면에 적합하게 그리드화
                call_pivoted = (
                    call_df.pivot_table(
                        values="iv",
                        index="days",
                        columns="strike",
                        aggfunc="mean",
                    )
                    .fillna(method="ffill")
                    .fillna(method="bfill")
                )

                fig.add_trace(
                    go.Surface(
                        z=call_pivoted.values,
                        x=call_pivoted.columns.tolist(),  # strike
                        y=call_pivoted.index.tolist(),  # days
                        colorscale="Blues",
                        opacity=0.8,
                        name="Call Options",
                        showscale=False,
                    ),
                    row=1,
                    col=1,
                )
            except Exception as e:
                st.warning(f"Error creating call surface: {e}")

        if len(put_df) > 1:
            try:
                # 풋 옵션 데이터를 표면에 적합하게 그리드화
                put_pivoted = (
                    put_df.pivot_table(
                        values="iv",
                        index="days",
                        columns="strike",
                        aggfunc="mean",
                    )
                    .fillna(method="ffill")
                    .fillna(method="bfill")
                )

                fig.add_trace(
                    go.Surface(
                        z=put_pivoted.values,
                        x=put_pivoted.columns.tolist(),  # strike
                        y=put_pivoted.index.tolist(),  # days
                        colorscale="Reds",
                        opacity=0.8,
                        name="Put Options",
                        showscale=True,
                        colorbar=dict(title="IV (%)", x=1.0, y=0.5),
                    ),
                    row=1,
                    col=2,
                )
            except Exception as e:
                st.warning(f"Error creating put surface: {e}")

        # 그래프 레이아웃 업데이트
        fig.update_layout(
            title="Option Volatility Surface",
            height=600,
            width=800,
            scene=dict(
                xaxis_title="Strike Price",
                yaxis_title="Days to Expiry",
                zaxis_title="Implied Volatility (%)",
                camera=dict(eye=dict(x=1.5, y=1.5, z=1.0)),
            ),
            scene2=dict(
                xaxis_title="Strike Price",
                yaxis_title="Days to Expiry",
                zaxis_title="Implied Volatility (%)",
                camera=dict(eye=dict(x=1.5, y=1.5, z=1.0)),
            ),
            margin=dict(l=65, r=50, b=65, t=90),
        )

        return fig
    except Exception as e:
        st.error(f"Error generating volatility surface: {e}")
        return None


def create_volatility_smile(vol_data, s, option_type="CALL"):
    """
    변동성 스마일/스큐 그래프 생성
    
    Parameters:
    -----------
    vol_data : list of dict
        변동성 데이터 포인트
    s : float
        현재 기초자산 가격
    option_type : str, optional
        "CALL" 또는 "PUT"
        
    Returns:
    --------
    Figure
        Plotly 변동성 스마일 그래프 객체
    """
    if not vol_data:
        return None
        
    try:
        # 2D smile plot
        fig = go.Figure()

        # 각 만기일에 대한 옵션 변동성 스마일 추가
        for data in vol_data:
            fig.add_trace(
                go.Scatter(
                    x=data["x"],
                    y=data["y"],
                    mode="lines+markers",
                    name=f"{data['name']} ({data['days']} days)",
                    line=dict(color=data["color"]),
                )
            )

        # 현재 주가에 수직선 추가
        fig.add_vline(
            x=s, line_width=1, line_dash="dash", line_color="green"
        )

        fig.update_layout(
            title=f"{option_type} Options Volatility Smile/Skew",
            xaxis_title="Strike Price",
            yaxis_title="Implied Volatility (%)",
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1,
            ),
            width=800,
            height=500,
            margin=dict(l=65, r=50, b=65, t=90),
        )

        return fig
    except Exception as e:
        st.error(f"Error generating volatility smile: {e}")
        return None


def create_volume_chart(calls, puts):
    """
    콜옵션과 풋옵션의 거래량 차트 생성
    
    Parameters:
    -----------
    calls : DataFrame
        콜옵션 체인 데이터
    puts : DataFrame
        풋옵션 체인 데이터
        
    Returns:
    --------
    tuple
        (Plotly 그래프 객체, 콜 총거래량, 풋 총거래량, 풋콜비율)
    """
    if calls is None or puts is None:
        return None, 0, 0, 0
        
    try:
        # 데이터 준비
        call_strikes = calls['strike'].tolist()
        put_strikes = puts['strike'].tolist()
        call_volumes = calls['volume'].tolist()
        put_volumes = puts['volume'].tolist()

        # 거래량 차트 생성
        volume_fig = go.Figure()

        # 콜 거래량 바
        volume_fig.add_trace(go.Bar(
            x=call_strikes, 
            y=call_volumes, 
            name='Calls', 
            marker_color='red'
        ))
        
        # 풋 거래량 바
        volume_fig.add_trace(go.Bar(
            x=put_strikes, 
            y=put_volumes, 
            name='Puts', 
            marker_color='blue'
        ))

        volume_fig.update_layout(
            title="Call and Put Volume by Strike Price",
            xaxis_title="Strike Price",
            yaxis_title="Volume",
            height=400,
            margin=dict(l=40, r=40, t=40, b=40),
        )

        # 통계 계산
        call_volume_total = sum(v for v in call_volumes if isinstance(v, (int, float)) and v >= 0)
        put_volume_total = sum(v for v in put_volumes if isinstance(v, (int, float)) and v >= 0)
        
        put_call_ratio = put_volume_total / call_volume_total if call_volume_total > 0 else 0

        return volume_fig, call_volume_total, put_volume_total, put_call_ratio
    except Exception as e:
        st.error(f"Error creating volume chart: {e}")
        return None, 0, 0, 0
