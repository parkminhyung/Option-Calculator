# Options Calculator

<div align="center">
  
[한국어](#한국어) | [English](#english) | [中文](#中文) | [日本語](#日本語)

</div>

---

<a id="한국어"></a>
## 한국어

# 옵션 계산기

금융 투자자를 위한 강력한 옵션 전략 시뮬레이션 및 분석 도구

<img width="1671" alt="image" src="https://github.com/user-attachments/assets/4b458818-ff41-4510-a9cd-af7a28f6cbb9" />

<img width="1671" alt="image" src="https://github.com/user-attachments/assets/44535153-5417-4d12-b9d8-d080582c0719" />

<img width="1673" alt="image" src="https://github.com/user-attachments/assets/2483ddce-1df5-4aae-9c14-10543384181f" />

<img width="1668" alt="image" src="https://github.com/user-attachments/assets/ef85e353-f66f-40a4-b79d-a5282f7c1d82" />

<img width="1671" alt="image" src="https://github.com/user-attachments/assets/aa6cc085-0502-4394-8f74-bbca63f865a9" />

## 주요 기능

- **실시간 주식 데이터 가져오기**: 주식 티커 심볼을 입력하여 실시간 주가 및 기본 정보 확인
- **옵션 전략 시뮬레이션**: 다양한 옵션 전략의 손익 구조를 시각화
- **그릭스 분석**: 델타, 감마, 베가, 세타, 로 등 옵션 그릭스 계산 및 시각화
- **내재 변동성 분석**: 변동성 스마일/스큐 및 3D 변동성 표면 생성
- **옵션 체인 조회**: 특정 만기일에 대한 전체 옵션 체인 데이터 확인 (yfinance 패키지 사용)

## 사용 가능한 옵션 전략

1. **단일(Single)**: 기본 콜/풋 옵션 매수/매도
2. **커버드(Covered)**: 커버드 콜(주식 매수 + 콜 매도) 또는 커버드 풋(주식 매도 + 풋 매도)
3. **보호(Protective)**: 보호 풋(주식 매수 + 풋 매수) 또는 보호 콜(주식 매도 + 콜 매수)
4. **스프레드(Spread)**: 콜/풋을 사용한 불/베어 스프레드
5. **스트래들(Straddle)**: 동일 행사가의 콜과 풋 동시 매수/매도
6. **스트랭글(Strangle)**: 서로 다른 행사가의 콜과 풋 동시 매수/매도
7. **스트립(Strip)**: 동일 행사가에서 콜 1개, 풋 2개 매수 (베어리시 전략)
8. **스트랩(Strap)**: 동일 행사가에서 콜 2개, 풋 1개 매수 (불리시 전략)
9. **버터플라이(Butterfly)**: 제한된 위험과 보상의 3단계 전략
10. **래더(Ladder)**: 일정 간격의 행사가를 가진 다중 단계 전략
11. **제이드 리저드(Jade Lizard)**: 풋 매도 + 콜 스프레드 매도 (불리시 전략)
12. **리버스 제이드 리저드(Reverse Jade Lizard)**: 풋 스프레드 매수 + 콜 매도 (베어리시 전략)
13. **콘도르(Condor)**: 제한된 위험과 보상의 4단계 전략

## 설치 및 실행

```bash
# 저장소 복제
git clone https://github.com/parkminhyung/Option-Calculator.git
cd Option-Calculator

# 필요한 패키지 설치
pip install streamlit pandas numpy yfinance plotly scipy

# 애플리케이션 실행
streamlit run app.py
```

## 사용 방법

1. 사이드바에 주식 티커 심볼을 입력하고 "Fetch Data" 버튼을 클릭합니다.
2. 원하는 옵션 전략, 행사가, 만기일 등의 파라미터를 설정합니다.
3. "Show Plot" 버튼을 클릭하여 전략의 손익 구조와 그릭스를 시각화합니다.
4. "Option Chain" 탭에서는 특정 만기일에 대한 옵션 체인 정보를 확인할 수 있습니다.

## 참고 사항

- Option Chain 정보는 yfinance 패키지에 의존하므로 일부 시장이나 종목에서는 데이터가 제한될 수 있습니다.
- Option Prices 하단에 나타나는 가격은 블랙숄즈모형을 기반으로 도출한 이론가격이며, 옵션 앞 +는 매수, -는 매도를 나타냅니다. 예를 들어 - call은 콜매도를 의미하며, +2x call은 콜을 2배로 매수한다는 뜻입니다.

---

<a id="english"></a>
## English

# Options Calculator

A powerful option strategy simulation and analysis tool for financial investors

<img width="1671" alt="image" src="https://github.com/user-attachments/assets/4b458818-ff41-4510-a9cd-af7a28f6cbb9" />

<img width="1671" alt="image" src="https://github.com/user-attachments/assets/44535153-5417-4d12-b9d8-d080582c0719" />

<img width="1673" alt="image" src="https://github.com/user-attachments/assets/2483ddce-1df5-4aae-9c14-10543384181f" />

<img width="1668" alt="image" src="https://github.com/user-attachments/assets/ef85e353-f66f-40a4-b79d-a5282f7c1d82" />

<img width="1671" alt="image" src="https://github.com/user-attachments/assets/aa6cc085-0502-4394-8f74-bbca63f865a9" />

## Key Features

- **Real-time Stock Data Fetching**: Input a stock ticker symbol to get real-time prices and basic information
- **Option Strategy Simulation**: Visualize the profit/loss structure of various option strategies
- **Greeks Analysis**: Calculate and visualize option Greeks including Delta, Gamma, Vega, Theta, and Rho
- **Implied Volatility Analysis**: Generate volatility smiles/skews and 3D volatility surfaces
- **Option Chain Lookup**: View complete option chain data for specific expiry dates (using yfinance package)

## Available Option Strategies

1. **Single**: Basic long/short positions in call/put options
2. **Covered**: Covered call (long stock + short call) or covered put (short stock + short put)
3. **Protective**: Protective put (long stock + long put) or protective call (short stock + long call)
4. **Spread**: Bull/bear spreads using calls or puts
5. **Straddle**: Long/short positions in both a call and put with the same strike
6. **Strangle**: Long/short positions in both a call and put with different strikes
7. **Strip**: Long 1 call and 2 puts at the same strike (bearish strategy)
8. **Strap**: Long 2 calls and 1 put at the same strike (bullish strategy)
9. **Butterfly**: A three-leg strategy with limited risk and reward
10. **Ladder**: A multi-leg strategy with strike prices at regular intervals
11. **Jade Lizard**: Short put + short call spread (bullish strategy)
12. **Reverse Jade Lizard**: Long put spread + short call (bearish strategy)
13. **Condor**: A four-leg strategy with limited risk and reward

## Installation and Running

```bash
# Clone the repository
git clone https://github.com/parkminhyung/Option-Calculator.git
cd Option-Calculator

# Install required packages
pip install streamlit pandas numpy yfinance plotly scipy

# Run the application
streamlit run app.py
```

## How to Use

1. Enter a stock ticker symbol in the sidebar and click the "Fetch Data" button.
2. Set your desired parameters including option strategy, strike prices, expiry date, etc.
3. Click the "Show Plot" button to visualize the strategy's profit/loss structure and Greeks.
4. Use the "Option Chain" tab to view option chain information for specific expiry dates.

## Notes

- Option Chain information depends on the yfinance package, so data may be limited for some markets or securities.
- The prices shown under Option Prices are theoretical prices based on the Black-Scholes model. The + sign in front of options indicates buying, while - indicates selling. For example, - call means selling a call, and +2x call means buying two call options.

---

<a id="中文"></a>
## 中文

# 期权计算器

为金融投资者提供的强大期权策略模拟和分析工具

<img width="1671" alt="image" src="https://github.com/user-attachments/assets/4b458818-ff41-4510-a9cd-af7a28f6cbb9" />

<img width="1671" alt="image" src="https://github.com/user-attachments/assets/44535153-5417-4d12-b9d8-d080582c0719" />

<img width="1673" alt="image" src="https://github.com/user-attachments/assets/2483ddce-1df5-4aae-9c14-10543384181f" />

<img width="1668" alt="image" src="https://github.com/user-attachments/assets/ef85e353-f66f-40a4-b79d-a5282f7c1d82" />

<img width="1671" alt="image" src="https://github.com/user-attachments/assets/aa6cc085-0502-4394-8f74-bbca63f865a9" />

## 主要功能

- **实时股票数据获取**：输入股票代码获取实时价格和基本信息
- **期权策略模拟**：可视化各种期权策略的盈亏结构
- **希腊字母分析**：计算并可视化期权希腊字母，包括Delta、Gamma、Vega、Theta和Rho
- **隐含波动率分析**：生成波动率微笑/偏斜和3D波动率曲面
- **期权链查询**：查看特定到期日的完整期权链数据（使用yfinance包）

## 可用期权策略

1. **单一(Single)**：期权多空基本持仓
2. **备兑(Covered)**：备兑认购期权（买入股票+卖出认购期权）或备兑认沽期权（卖出股票+卖出认沽期权）
3. **保护(Protective)**：保护性认沽期权（买入股票+买入认沽期权）或保护性认购期权（卖出股票+买入认购期权）
4. **价差(Spread)**：使用认购期权或认沽期权的牛市/熊市价差策略
5. **跨式(Straddle)**：同时买入/卖出相同行权价的认购期权和认沽期权
6. **宽跨式(Strangle)**：同时买入/卖出不同行权价的认购期权和认沽期权
7. **看跌组合(Strip)**：在相同行权价买入1个认购期权和2个认沽期权（看跌策略）
8. **看涨组合(Strap)**：在相同行权价买入2个认购期权和1个认沽期权（看涨策略）
9. **蝶式(Butterfly)**：具有有限风险和回报的三腿策略
10. **阶梯(Ladder)**：具有等间距行权价的多腿策略
11. **翡翠蜥蜴(Jade Lizard)**：卖出认沽期权+卖出认购期权价差（看涨策略）
12. **反翡翠蜥蜴(Reverse Jade Lizard)**：买入认沽期权价差+卖出认购期权（看跌策略）
13. **秃鹰(Condor)**：具有有限风险和回报的四腿策略

## 安装和运行

```bash
# 克隆仓库
git clone https://github.com/parkminhyung/Option-Calculator.git
cd Option-Calculator

# 安装所需包
pip install streamlit pandas numpy yfinance plotly scipy

# 运行应用
streamlit run app.py
```

## 使用方法

1. 在侧边栏输入股票代码并点击"Fetch Data"按钮。
2. 设置所需参数，包括期权策略、行权价、到期日等。
3. 点击"Show Plot"按钮可视化策略的盈亏结构和希腊字母。
4. 使用"Option Chain"标签页查看特定到期日的期权链信息。

## 注意事项

- 期权链信息依赖于yfinance包，因此某些市场或证券的数据可能有限。
- Option Prices下显示的价格是基于Black-Scholes模型的理论价格。期权前的+号表示买入，-号表示卖出。例如，-call表示卖出认购期权，+2x call表示买入两个认购期权。

---

<a id="日本語"></a>
## 日本語

# オプション計算機

金融投資家のための強力なオプション戦略シミュレーションおよび分析ツール

<img width="1671" alt="image" src="https://github.com/user-attachments/assets/4b458818-ff41-4510-a9cd-af7a28f6cbb9" />

<img width="1671" alt="image" src="https://github.com/user-attachments/assets/44535153-5417-4d12-b9d8-d080582c0719" />

<img width="1673" alt="image" src="https://github.com/user-attachments/assets/2483ddce-1df5-4aae-9c14-10543384181f" />

<img width="1668" alt="image" src="https://github.com/user-attachments/assets/ef85e353-f66f-40a4-b79d-a5282f7c1d82" />

<img width="1671" alt="image" src="https://github.com/user-attachments/assets/aa6cc085-0502-4394-8f74-bbca63f865a9" />

## 主な機能

- **リアルタイム株式データ取得**：株式ティッカーシンボルを入力してリアルタイム価格と基本情報を取得
- **オプション戦略シミュレーション**：様々なオプション戦略の損益構造を視覚化
- **ギリシャ指標分析**：デルタ、ガンマ、ベガ、シータ、ローなどのオプションギリシャ指標を計算・視覚化
- **インプライドボラティリティ分析**：ボラティリティスマイル/スキューおよび3Dボラティリティサーフェスを生成
- **オプションチェーン検索**：特定の満期日の完全なオプションチェーンデータを表示（yfinanceパッケージ使用）

## 利用可能なオプション戦略

1. **シングル(Single)**：コール/プットオプションの基本的なロング/ショートポジション
2. **カバード(Covered)**：カバードコール（株式ロング+コールショート）またはカバードプット（株式ショート+プットショート）
3. **プロテクティブ(Protective)**：プロテクティブプット（株式ロング+プットロング）またはプロテクティブコール（株式ショート+コールロング）
4. **スプレッド(Spread)**：コールまたはプットを使用したブル/ベアスプレッド
5. **ストラドル(Straddle)**：同じ行使価格のコールとプットの両方をロング/ショート
6. **ストラングル(Strangle)**：異なる行使価格のコールとプットの両方をロング/ショート
7. **ストリップ(Strip)**：同じ行使価格でコール1つとプット2つをロング（ベアリッシュ戦略）
8. **ストラップ(Strap)**：同じ行使価格でコール2つとプット1つをロング（ブリッシュ戦略）
9. **バタフライ(Butterfly)**：限定されたリスクとリワードを持つ3レッグ戦略
10. **ラダー(Ladder)**：一定間隔の行使価格を持つマルチレッグ戦略
11. **ジェイドリザード(Jade Lizard)**：プットショート+コールスプレッドショート（ブリッシュ戦略）
12. **リバースジェイドリザード(Reverse Jade Lizard)**：プットスプレッドロング+コールショート（ベアリッシュ戦略）
13. **コンドル(Condor)**：限定されたリスクとリワードを持つ4レッグ戦略

## インストールと実行

```bash
# リポジトリをクローン
git clone https://github.com/parkminhyung/Option-Calculator.git
cd Option-Calculator

# 必要なパッケージをインストール
pip install streamlit pandas numpy yfinance plotly scipy

# アプリケーションを実行
streamlit run app.py
```

## 使用方法

1. サイドバーに株式ティッカーシンボルを入力し、「Fetch Data」ボタンをクリックします。
2. オプション戦略、行使価格、満期日などの希望するパラメータを設定します。
3. 「Show Plot」ボタンをクリックして、戦略の損益構造とギリシャ指標を視覚化します。
4. 「Option Chain」タブを使用して、特定の満期日のオプションチェーン情報を表示します。

## 注意事項

- オプションチェーン情報はyfinanceパッケージに依存しているため、一部の市場や証券ではデータが制限される場合があります。
- Option Prices下に表示される価格はブラック・ショールズモデルに基づく理論価格です。オプションの前の+は買い、-は売りを示します。例えば、- callはコールの売りを意味し、+2x callはコールを2倍買うことを意味します。
