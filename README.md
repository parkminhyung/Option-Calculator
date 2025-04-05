# Option Calculator / 옵션 계산기 / 期权计算器

<details>
<summary>English</summary>

Option Calculator is a Streamlit-based web application that allows users to visualize and analyze the profit/loss and Greeks of various option trading strategies.

<img width="1680" alt="image" src="https://github.com/user-attachments/assets/04b3fa7c-c478-46be-bb84-03cdf3bec08f" />

<img width="1680" alt="image" src="https://github.com/user-attachments/assets/9ce20a5d-6155-468e-a5bc-bc2ef7444250" />

<img width="1680" alt="image" src="https://github.com/user-attachments/assets/40cbccd3-770c-4943-ac4b-220fccc28e1a" />


## Features

- Real-time stock information and option chain lookup
- Theoretical option price calculation using the Black-Scholes model
- Visualization of profit/loss curves for various option strategies
- Analysis of option Greeks (Delta, Gamma, Vega, Theta, Rho)
- Support for multiple option strategies:
  - Single option
  - Covered Call/Put
  - Protective Put/Call
  - Spread
  - Straddle
  - Strangle
  - Strip
  - Strap
  - Butterfly
  - Ladder
  - Jade Lizard
  - Reverse Jade Lizard
  - Condor

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/option-calculator.git
cd option-calculator
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate  # Windows
```

3. Install the required packages:
```bash
pip install -r requirements.txt
```

## Usage

1. Run the application:
```bash
streamlit run app.py
```

2. A web browser will automatically open with the application running. If it doesn't, enter the URL displayed in your terminal into your browser.

3. How to use:
   - Enter a stock ticker symbol in the left sidebar and click the 'Fetch Data' button.
   - Set option parameters (underlying price, expiry date, volatility, etc.).
   - Select an option strategy and position.
   - Click the 'Show Plot' button to view the profit/loss curve and Greeks.
   
4. Special features:
   - Option prices are automatically calculated based on the Black-Scholes model when parameters are entered. Users can modify these prices if needed.
   - The "+" and "-" signs before call and put in the option price section indicate buying and selling positions: "+" for buying (long) and "-" for selling (short).
   - The volatility shown is the 52-week historical volatility, which users can modify.
   - Strategy selection is made through the side, option type, and strategy selection fields. For example, selecting side: short, option type: call, and strategy: single creates a call option short strategy.
   - The risk-free rate uses the U.S. 10-year Treasury yield as a benchmark.

## Project Structure

```
option-calculator/
│
├── app.py                  # Main application file
├── utils/
│   ├── __init__.py
│   ├── data_utils.py       # Data-related utility functions
│   ├── option_pricing.py   # Option pricing functions
│   ├── payoff.py           # Profit/loss calculation functions
│   └── plotting.py         # Visualization functions
│
├── styles.py               # Style definitions
├── requirements.txt        # List of required packages
└── README.md               # Project description
```

## Required Packages

Main packages:
- streamlit
- pandas
- numpy
- yfinance
- plotly
- scipy

For a complete list, see the requirements.txt file.

## Reference

### Black-Scholes Model

The Black-Scholes model is the most widely used option pricing model:

```
d₁ = [ln(S₀/K) + (rf - y + 0.5σ²)τ] / (σ√τ)
d₂ = d₁ - σ√τ

Call price: C(S₀, τ) = S₀N(d₁)e^(-yτ) - Ke^(-rfτ)N(d₂)
Put price: P(S₀, τ) = Ke^(-rfτ)N(-d₂) - S₀N(-d₁)e^(-yτ)
```

where,
- S₀: Underlying asset price
- K: Strike price
- rf: Risk-free interest rate
- y: Dividend yield
- τ: Time to expiration
- N(x): Standard normal cumulative distribution function

## License

MIT License

## Contact

If you have any questions or suggestions, please contact: pmh621@naver.com
</details>

<details>
<summary>한국어</summary>

옵션 계산기는 다양한 옵션 거래 전략의 손익과 그릭스를 시각화하고 분석할 수 있는 Streamlit 기반 웹 애플리케이션입니다.

<img width="1680" alt="image" src="https://github.com/user-attachments/assets/04b3fa7c-c478-46be-bb84-03cdf3bec08f" />

<img width="1680" alt="image" src="https://github.com/user-attachments/assets/9ce20a5d-6155-468e-a5bc-bc2ef7444250" />

<img width="1680" alt="image" src="https://github.com/user-attachments/assets/40cbccd3-770c-4943-ac4b-220fccc28e1a" />

## 기능

- 실시간 주식 정보 및 옵션 체인 조회
- 블랙-숄즈 모델을 이용한 이론적 옵션 가격 계산
- 다양한 옵션 전략의 손익 곡선 시각화
- 옵션 그릭스 (Delta, Gamma, Vega, Theta, Rho) 분석
- 다양한 옵션 전략 지원:
  - Single (단일 옵션)
  - Covered Call/Put (커버드 콜/풋)
  - Protective Put/Call (보호적 풋/콜)
  - Spread (스프레드)
  - Straddle (스트래들)
  - Strangle (스트랭글)
  - Strip (스트립)
  - Strap (스트랩)
  - Butterfly (버터플라이)
  - Ladder (래더)
  - Jade Lizard (제이드 리저드)
  - Reverse Jade Lizard (리버스 제이드 리저드)
  - Condor (콘도르)

## 설치 방법

1. 저장소를 클론합니다:
```bash
git clone https://github.com/yourusername/option-calculator.git
cd option-calculator
```

2. 가상 환경을 생성하고 활성화합니다:
```bash
python -m venv venv
source venv/bin/activate  # 리눅스/맥
# 또는
venv\Scripts\activate  # 윈도우
```

3. 필요한 패키지를 설치합니다:
```bash
pip install -r requirements.txt
```

## 사용 방법

1. 애플리케이션을 실행합니다:
```bash
streamlit run app.py
```

2. 웹 브라우저가 자동으로 열리고 애플리케이션이 실행됩니다. 실행되지 않는 경우, 터미널에 표시된 URL을 브라우저에 입력하세요.

3. 사용 방법:
   - 왼쪽 사이드바에 주식 티커 심볼을 입력하고 'Fetch Data' 버튼을 클릭합니다.
   - 옵션 파라미터를 설정합니다 (기초 자산 가격, 만기일, 변동성 등).
   - 옵션 전략과 포지션을 선택합니다.
   - 'Show Plot' 버튼을 클릭하여 손익 곡선과 그릭스를 확인합니다.

4. 특별 기능:
   - 옵션 가격은 사용자가 파라미터를 입력하면 블랙-숄즈 모형을 기반으로 자동으로 계산되어 입력창에 나타납니다. 사용자가 필요에 따라 이 가격을 수정할 수 있습니다.
   - 옵션 가격 섹션에서 콜과 풋 앞의 "+"와 "-" 기호는 매수와 매도 포지션을 나타냅니다: "+"는 매수(롱), "-"는 매도(숏)를 의미합니다.
   - 표시되는 변동성은 52주 역사적 변동성이며, 사용자가 수정할 수 있습니다.
   - 전략 선택은 사이드, 옵션 유형 및 전략 선택 필드를 통해 이루어집니다. 예를 들어, 사이드: short, 옵션 유형: call, 전략: single을 선택하면 콜 옵션 숏 전략을 생성합니다.
   - 무위험 금리는 미국 10년 국채 수익률을 기준으로 사용합니다.

## 프로젝트 구조

```
option-calculator/
│
├── app.py                  # 메인 애플리케이션 파일
├── utils/
│   ├── __init__.py
│   ├── data_utils.py       # 데이터 관련 유틸리티 함수
│   ├── option_pricing.py   # 옵션 가격 계산 함수
│   ├── payoff.py           # 손익 계산 함수
│   └── plotting.py         # 시각화 함수
│
├── styles.py               # 스타일 정의
├── requirements.txt        # 필요한 패키지 목록
└── README.md               # 프로젝트 설명
```

## 필요한 패키지

주요 패키지:
- streamlit
- pandas
- numpy
- yfinance
- plotly
- scipy

자세한 목록은 requirements.txt 파일을 참조하세요.

## 참고 자료

### 블랙-숄즈 모델

블랙-숄즈 모델은 가장 널리 사용되는 옵션 가격 결정 모델입니다:

```
d₁ = [ln(S₀/K) + (rf - y + 0.5σ²)τ] / (σ√τ)
d₂ = d₁ - σ√τ

콜 가격: C(S₀, τ) = S₀N(d₁)e^(-yτ) - Ke^(-rfτ)N(d₂)
풋 가격: P(S₀, τ) = Ke^(-rfτ)N(-d₂) - S₀N(-d₁)e^(-yτ)
```

여기서,
- S₀: 기초 자산 가격
- K: 행사가
- rf: 무위험 이자율
- y: 배당 수익률
- τ: 만기까지 남은 시간
- N(x): 표준 정규 누적 분포 함수

## 라이센스

MIT License

## 연락처

질문이나 제안이 있으시면 연락해 주세요: pmh621@naver.com
</details>

<details>
<summary>中文</summary>

期权计算器是一个基于Streamlit的网络应用程序，用户可以可视化和分析各种期权交易策略的盈亏和希腊字母值。

<img width="1680" alt="image" src="https://github.com/user-attachments/assets/04b3fa7c-c478-46be-bb84-03cdf3bec08f" />

<img width="1680" alt="image" src="https://github.com/user-attachments/assets/9ce20a5d-6155-468e-a5bc-bc2ef7444250" />

<img width="1680" alt="image" src="https://github.com/user-attachments/assets/40cbccd3-770c-4943-ac4b-220fccc28e1a" />


## 功能

- 实时股票信息和期权链查询
- 使用布莱克-斯科尔斯模型计算理论期权价格
- 可视化各种期权策略的盈亏曲线
- 分析期权希腊字母（Delta, Gamma, Vega, Theta, Rho）
- 支持多种期权策略：
  - 单一期权（Single）
  - 备兑看涨/看跌期权（Covered Call/Put）
  - 保护性看跌/看涨期权（Protective Put/Call）
  - 价差策略（Spread）
  - 跨式策略（Straddle）
  - 宽跨式策略（Strangle）
  - 多头看跌偏置策略（Strip）
  - 多头看涨偏置策略（Strap）
  - 蝶式策略（Butterfly）
  - 阶梯策略（Ladder）
  - 玉蜥蜴策略（Jade Lizard）
  - 反向玉蜥蜴策略（Reverse Jade Lizard）
  - 秃鹰策略（Condor）

## 安装方法

1. 克隆仓库：
```bash
git clone https://github.com/yourusername/option-calculator.git
cd option-calculator
```

2. 创建并激活虚拟环境：
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 或者
venv\Scripts\activate  # Windows
```

3. 安装所需的包：
```bash
pip install -r requirements.txt
```

## 使用方法

1. 运行应用程序：
```bash
streamlit run app.py
```

2. 网络浏览器将自动打开并运行应用程序。如果没有自动打开，请在浏览器中输入终端显示的URL。

3. 使用说明：
   - 在左侧边栏输入股票代码，并点击"Fetch Data"按钮。
   - 设置期权参数（标的资产价格、到期日、波动率等）。
   - 选择期权策略和持仓方向。
   - 点击"Show Plot"按钮查看盈亏曲线和希腊字母值。

4. 特殊功能：
   - 输入参数后，期权价格会根据布莱克-斯科尔斯模型自动计算并显示在输入框中。用户可以根据需要修改这些价格。
   - 期权价格部分中看涨和看跌期权前的"+"和"-"符号表示买入和卖出持仓："+"表示买入（做多），"-"表示卖出（做空）。
   - 显示的波动率是52周历史波动率，用户可以修改。
   - 策略选择通过方向（side）、期权类型（option type）和策略（strategy）选择字段完成。例如，选择方向：short，期权类型：call，策略：single将创建看涨期权卖空策略。
   - 无风险利率使用美国10年期国债收益率作为基准。

## 项目结构

```
option-calculator/
│
├── app.py                  # 主应用程序文件
├── utils/
│   ├── __init__.py
│   ├── data_utils.py       # 数据相关的实用函数
│   ├── option_pricing.py   # 期权定价函数
│   ├── payoff.py           # 盈亏计算函数
│   └── plotting.py         # 可视化函数
│
├── styles.py               # 样式定义
├── requirements.txt        # 所需包列表
└── README.md               # 项目描述
```

## 所需包

主要包：
- streamlit
- pandas
- numpy
- yfinance
- plotly
- scipy

完整列表请参见requirements.txt文件。

## 参考资料

### 布莱克-斯科尔斯模型

布莱克-斯科尔斯模型是最广泛使用的期权定价模型：

```
d₁ = [ln(S₀/K) + (rf - y + 0.5σ²)τ] / (σ√τ)
d₂ = d₁ - σ√τ

看涨期权价格: C(S₀, τ) = S₀N(d₁)e^(-yτ) - Ke^(-rfτ)N(d₂)
看跌期权价格: P(S₀, τ) = Ke^(-rfτ)N(-d₂) - S₀N(-d₁)e^(-yτ)
```

其中，
- S₀: 标的资产价格
- K: 行权价格
- rf: 无风险利率
- y: 股息收益率
- τ: 到期时间
- N(x): 标准正态累积分布函数

## 许可证

MIT许可证

## 联系方式

如有任何问题或建议，请联系：pmh621@naver.com
</details>
