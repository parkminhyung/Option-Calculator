# 🚀 Options Calculator

<div align="center">
  
### 💹 금융 시장을 위한 최고의 옵션 분석 도구 💹

[![Stars](https://img.shields.io/badge/Stars-⭐⭐⭐⭐⭐-yellow)](https://github.com/parkminhyung/Option-Calculator)
[![Version](https://img.shields.io/badge/Version-1.0.0-blue)](https://github.com/parkminhyung/Option-Calculator/releases)
[![License](https://img.shields.io/badge/License-MIT-green)](https://opensource.org/licenses/MIT)

**[한국어](#한국어) | [English](#english) | [中文](#中文) | [日本語](#日本語)**

</div>

---

<a id="한국어"></a>

<div align="center">
  
# 🇰🇷 옵션 계산기

### 금융 투자자를 위한 강력한 옵션 전략 시뮬레이션 및 분석 도구
  
</div>

<p align="center">
  <img width="800" alt="메인 화면" src="https://github.com/user-attachments/assets/4b458818-ff41-4510-a9cd-af7a28f6cbb9" />
</p>

<details open>
<summary><b>📊 스크린샷 더 보기</b></summary>
<p align="center">
  <img width="800" alt="화면 2" src="https://github.com/user-attachments/assets/44535153-5417-4d12-b9d8-d080582c0719" />
  <br><br>
  <img width="800" alt="화면 3" src="https://github.com/user-attachments/assets/2483ddce-1df5-4aae-9c14-10543384181f" />
  <br><br>
  <img width="800" alt="화면 4" src="https://github.com/user-attachments/assets/ef85e353-f66f-40a4-b79d-a5282f7c1d82" />
  <br><br>
  <img width="800" alt="화면 5" src="https://github.com/user-attachments/assets/aa6cc085-0502-4394-8f74-bbca63f865a9" />
</p>
</details>

## ✨ 주요 기능

| 기능 | 설명 |
|------|------|
| 📈 **실시간 주식 데이터** | 주식 티커 심볼을 입력하여 실시간 주가 및 기본 정보 확인 |
| 📊 **옵션 전략 시뮬레이션** | 다양한 옵션 전략의 손익 구조를 시각화 |
| 🔄 **그릭스 분석** | 델타, 감마, 베가, 세타, 로 등 옵션 그릭스 계산 및 시각화 |
| 📉 **내재 변동성 분석** | 변동성 스마일/스큐 및 3D 변동성 표면 생성 |
| 🔍 **옵션 체인 조회** | 특정 만기일에 대한 전체 옵션 체인 데이터 확인 (yfinance 패키지 사용) |

## 📋 사용 가능한 옵션 전략

<table>
  <tr>
    <td><b>🔹 단일(Single)</b></td>
    <td>기본 콜/풋 옵션 매수/매도</td>
  </tr>
  <tr>
    <td><b>🔹 커버드(Covered)</b></td>
    <td>커버드 콜(주식 매수 + 콜 매도) 또는 커버드 풋(주식 매도 + 풋 매도)</td>
  </tr>
  <tr>
    <td><b>🔹 보호(Protective)</b></td>
    <td>보호 풋(주식 매수 + 풋 매수) 또는 보호 콜(주식 매도 + 콜 매수)</td>
  </tr>
  <tr>
    <td><b>🔹 스프레드(Spread)</b></td>
    <td>콜/풋을 사용한 불/베어 스프레드</td>
  </tr>
  <tr>
    <td><b>🔹 스트래들(Straddle)</b></td>
    <td>동일 행사가의 콜과 풋 동시 매수/매도</td>
  </tr>
  <tr>
    <td><b>🔹 스트랭글(Strangle)</b></td>
    <td>서로 다른 행사가의 콜과 풋 동시 매수/매도</td>
  </tr>
  <tr>
    <td><b>🔹 스트립(Strip)</b></td>
    <td>동일 행사가에서 콜 1개, 풋 2개 매수 (베어리시 전략)</td>
  </tr>
  <tr>
    <td><b>🔹 스트랩(Strap)</b></td>
    <td>동일 행사가에서 콜 2개, 풋 1개 매수 (불리시 전략)</td>
  </tr>
  <tr>
    <td><b>🔹 버터플라이(Butterfly)</b></td>
    <td>제한된 위험과 보상의 3단계 전략</td>
  </tr>
  <tr>
    <td><b>🔹 래더(Ladder)</b></td>
    <td>일정 간격의 행사가를 가진 다중 단계 전략</td>
  </tr>
  <tr>
    <td><b>🔹 제이드 리저드(Jade Lizard)</b></td>
    <td>풋 매도 + 콜 스프레드 매도 (불리시 전략)</td>
  </tr>
  <tr>
    <td><b>🔹 리버스 제이드 리저드(Reverse Jade Lizard)</b></td>
    <td>풋 스프레드 매수 + 콜 매도 (베어리시 전략)</td>
  </tr>
  <tr>
    <td><b>🔹 콘도르(Condor)</b></td>
    <td>제한된 위험과 보상의 4단계 전략</td>
  </tr>
</table>

## 🛠️ 설치 및 실행

```bash
# 저장소 복제
git clone https://github.com/parkminhyung/Option-Calculator.git
cd Option-Calculator

# 필요한 패키지 설치
pip install streamlit pandas numpy yfinance plotly scipy

# 애플리케이션 실행
streamlit run app.py
```

## 📝 사용 방법

1. 사이드바에 주식 티커 심볼을 입력하고 "Fetch Data" 버튼을 클릭합니다.

2. 원하는 옵션 전략, 행사가, 만기일 등의 파라미터를 설정합니다.

3. "Show Plot" 버튼을 클릭하여 전략의 손익 구조와 그릭스를 시각화합니다.

4. "Option Chain" 탭에서는 특정 만기일에 대한 옵션 체인 정보를 확인할 수 있습니다.

## 📌 참고 사항

> ⚠️ **옵션 체인 정보는 yfinance 패키지에 의존하므로 일부 시장이나 종목에서는 데이터가 제한될 수 있습니다.**

> 💡 **Option Prices 하단에 나타나는 가격은 블랙숄즈모형을 기반으로 도출한 이론가격이며, 옵션 앞 +는 매수, -는 매도를 나타냅니다. 예를 들어 - call은 콜매도를 의미하며, +2x call은 콜을 2배로 매수한다는 뜻입니다.**

---

<a id="english"></a>

<div align="center">
  
# 🇺🇸 Options Calculator

### A powerful option strategy simulation and analysis tool for financial investors
  
</div>

<p align="center">
  <img width="800" alt="Main Screen" src="https://github.com/user-attachments/assets/4b458818-ff41-4510-a9cd-af7a28f6cbb9" />
</p>

<details>
<summary><b>📊 More Screenshots</b></summary>
<p align="center">
  <img width="800" alt="Screen 2" src="https://github.com/user-attachments/assets/44535153-5417-4d12-b9d8-d080582c0719" />
  <br><br>
  <img width="800" alt="Screen 3" src="https://github.com/user-attachments/assets/2483ddce-1df5-4aae-9c14-10543384181f" />
  <br><br>
  <img width="800" alt="Screen 4" src="https://github.com/user-attachments/assets/ef85e353-f66f-40a4-b79d-a5282f7c1d82" />
  <br><br>
  <img width="800" alt="Screen 5" src="https://github.com/user-attachments/assets/aa6cc085-0502-4394-8f74-bbca63f865a9" />
</p>
</details>

## ✨ Key Features

| Feature | Description |
|---------|-------------|
| 📈 **Real-time Stock Data** | Input a stock ticker symbol to get real-time prices and basic information |
| 📊 **Option Strategy Simulation** | Visualize the profit/loss structure of various option strategies |
| 🔄 **Greeks Analysis** | Calculate and visualize option Greeks including Delta, Gamma, Vega, Theta, and Rho |
| 📉 **Implied Volatility Analysis** | Generate volatility smiles/skews and 3D volatility surfaces |
| 🔍 **Option Chain Lookup** | View complete option chain data for specific expiry dates (using yfinance package) |

## 📋 Available Option Strategies

<table>
  <tr>
    <td><b>🔹 Single</b></td>
    <td>Basic long/short positions in call/put options</td>
  </tr>
  <tr>
    <td><b>🔹 Covered</b></td>
    <td>Covered call (long stock + short call) or covered put (short stock + short put)</td>
  </tr>
  <tr>
    <td><b>🔹 Protective</b></td>
    <td>Protective put (long stock + long put) or protective call (short stock + long call)</td>
  </tr>
  <tr>
    <td><b>🔹 Spread</b></td>
    <td>Bull/bear spreads using calls or puts</td>
  </tr>
  <tr>
    <td><b>🔹 Straddle</b></td>
    <td>Long/short positions in both a call and put with the same strike</td>
  </tr>
  <tr>
    <td><b>🔹 Strangle</b></td>
    <td>Long/short positions in both a call and put with different strikes</td>
  </tr>
  <tr>
    <td><b>🔹 Strip</b></td>
    <td>Long 1 call and 2 puts at the same strike (bearish strategy)</td>
  </tr>
  <tr>
    <td><b>🔹 Strap</b></td>
    <td>Long 2 calls and 1 put at the same strike (bullish strategy)</td>
  </tr>
  <tr>
    <td><b>🔹 Butterfly</b></td>
    <td>A three-leg strategy with limited risk and reward</td>
  </tr>
  <tr>
    <td><b>🔹 Ladder</b></td>
    <td>A multi-leg strategy with strike prices at regular intervals</td>
  </tr>
  <tr>
    <td><b>🔹 Jade Lizard</b></td>
    <td>Short put + short call spread (bullish strategy)</td>
  </tr>
  <tr>
    <td><b>🔹 Reverse Jade Lizard</b></td>
    <td>Long put spread + short call (bearish strategy)</td>
  </tr>
  <tr>
    <td><b>🔹 Condor</b></td>
    <td>A four-leg strategy with limited risk and reward</td>
  </tr>
</table>

## 🛠️ Installation and Running

```bash
# Clone the repository
git clone https://github.com/parkminhyung/Option-Calculator.git
cd Option-Calculator

# Install required packages
pip install streamlit pandas numpy yfinance plotly scipy

# Run the application
streamlit run app.py
```

## 📝 How to Use

1. Enter a stock ticker symbol in the sidebar and click the "Fetch Data" button.

2. Set your desired parameters including option strategy, strike prices, expiry date, etc.

3. Click the "Show Plot" button to visualize the strategy's profit/loss structure and Greeks.

4. Use the "Option Chain" tab to view option chain information for specific expiry dates.

## 📌 Notes

> ⚠️ **Option Chain information depends on the yfinance package, so data may be limited for some markets or securities.**

> 💡 **The prices shown under Option Prices are theoretical prices based on the Black-Scholes model. The + sign in front of options indicates buying, while - indicates selling. For example, - call means selling a call, and +2x call means buying two call options.**

---

<a id="中文"></a>

<div align="center">
  
# 🇨🇳 期权计算器

### 为金融投资者提供的强大期权策略模拟和分析工具
  
</div>

<p align="center">
  <img width="800" alt="主屏幕" src="https://github.com/user-attachments/assets/4b458818-ff41-4510-a9cd-af7a28f6cbb9" />
</p>

<details>
<summary><b>📊 更多截图</b></summary>
<p align="center">
  <img width="800" alt="屏幕 2" src="https://github.com/user-attachments/assets/44535153-5417-4d12-b9d8-d080582c0719" />
  <br><br>
  <img width="800" alt="屏幕 3" src="https://github.com/user-attachments/assets/2483ddce-1df5-4aae-9c14-10543384181f" />
  <br><br>
  <img width="800" alt="屏幕 4" src="https://github.com/user-attachments/assets/ef85e353-f66f-40a4-b79d-a5282f7c1d82" />
  <br><br>
  <img width="800" alt="屏幕 5" src="https://github.com/user-attachments/assets/aa6cc085-0502-4394-8f74-bbca63f865a9" />
</p>
</details>

## ✨ 主要功能

| 功能 | 描述 |
|------|------|
| 📈 **实时股票数据获取** | 输入股票代码获取实时价格和基本信息 |
| 📊 **期权策略模拟** | 可视化各种期权策略的盈亏结构 |
| 🔄 **希腊字母分析** | 计算并可视化期权希腊字母，包括Delta、Gamma、Vega、Theta和Rho |
| 📉 **隐含波动率分析** | 生成波动率微笑/偏斜和3D波动率曲面 |
| 🔍 **期权链查询** | 查看特定到期日的完整期权链数据（使用yfinance包） |

## 📋 可用期权策略

<table>
  <tr>
    <td><b>🔹 单一(Single)</b></td>
    <td>期权多空基本持仓</td>
  </tr>
  <tr>
    <td><b>🔹 备兑(Covered)</b></td>
    <td>备兑认购期权（买入股票+卖出认购期权）或备兑认沽期权（卖出股票+卖出认沽期权）</td>
  </tr>
  <tr>
    <td><b>🔹 保护(Protective)</b></td>
    <td>保护性认沽期权（买入股票+买入认沽期权）或保护性认购期权（卖出股票+买入认购期权）</td>
  </tr>
  <tr>
    <td><b>🔹 价差(Spread)</b></td>
    <td>使用认购期权或认沽期权的牛市/熊市价差策略</td>
  </tr>
  <tr>
    <td><b>🔹 跨式(Straddle)</b></td>
    <td>同时买入/卖出相同行权价的认购期权和认沽期权</td>
  </tr>
  <tr>
    <td><b>🔹 宽跨式(Strangle)</b></td>
    <td>同时买入/卖出不同行权价的认购期权和认沽期权</td>
  </tr>
  <tr>
    <td><b>🔹 看跌组合(Strip)</b></td>
    <td>在相同行权价买入1个认购期权和2个认沽期权（看跌策略）</td>
  </tr>
  <tr>
    <td><b>🔹 看涨组合(Strap)</b></td>
    <td>在相同行权价买入2个认购期权和1个认沽期权（看涨策略）</td>
  </tr>
  <tr>
    <td><b>🔹 蝶式(Butterfly)</b></td>
    <td>具有有限风险和回报的三腿策略</td>
  </tr>
  <tr>
    <td><b>🔹 阶梯(Ladder)</b></td>
    <td>具有等间距行权价的多腿策略</td>
  </tr>
  <tr>
    <td><b>🔹 翡翠蜥蜴(Jade Lizard)</b></td>
    <td>卖出认沽期权+卖出认购期权价差（看涨策略）</td>
  </tr>
  <tr>
    <td><b>🔹 反翡翠蜥蜴(Reverse Jade Lizard)</b></td>
    <td>买入认沽期权价差+卖出认购期权（看跌策略）</td>
  </tr>
  <tr>
    <td><b>🔹 秃鹰(Condor)</b></td>
    <td>具有有限风险和回报的四腿策略</td>
  </tr>
</table>

## 🛠️ 安装和运行

```bash
# 克隆仓库
git clone https://github.com/parkminhyung/Option-Calculator.git
cd Option-Calculator

# 安装所需包
pip install streamlit pandas numpy yfinance plotly scipy

# 运行应用
streamlit run app.py
```

## 📝 使用方法

1. 在侧边栏输入股票代码并点击"Fetch Data"按钮。

2. 设置所需参数，包括期权策略、行权价、到期日等。

3. 点击"Show Plot"按钮可视化策略的盈亏结构和希腊字母。

4. 使用"Option Chain"标签页查看特定到期日的期权链信息。

## 📌 注意事项

> ⚠️ **期权链信息依赖于yfinance包，因此某些市场或证券的数据可能有限。**

> 💡 **Option Prices下显示的价格是基于Black-Scholes模型的理论价格。期权前的+号表示买入，-号表示卖出。例如，-call表示卖出认购期权，+2x call表示买入两个认购期权。**

---

<a id="日本語"></a>

<div align="center">
  
# 🇯🇵 オプション計算機

### 金融投資家のための強力なオプション戦略シミュレーションおよび分析ツール
  
</div>

<p align="center">
  <img width="800" alt="メイン画面" src="https://github.com/user-attachments/assets/4b458818-ff41-4510-a9cd-af7a28f6cbb9" />
</p>

<details>
<summary><b>📊 その他のスクリーンショット</b></summary>
<p align="center">
  <img width="800" alt="画面 2" src="https://github.com/user-attachments/assets/44535153-5417-4d12-b9d8-d080582c0719" />
  <br><br>
  <img width="800" alt="画面 3" src="https://github.com/user-attachments/assets/2483ddce-1df5-4aae-9c14-10543384181f" />
  <br><br>
  <img width="800" alt="画面 4" src="https://github.com/user-attachments/assets/ef85e353-f66f-40a4-b79d-a5282f7c1d82" />
  <br><br>
  <img width="800" alt="画面 5" src="https://github.com/user-attachments/assets/aa6cc085-0502-4394-8f74-bbca63f865a9" />
</p>
</details>

## ✨ 主な機能

| 機能 | 説明 |
|------|------|
| 📈 **リアルタイム株式データ取得** | 株式ティッカーシンボルを入力してリアルタイム価格と基本情報を取得 |
| 📊 **オプション戦略シミュレーション** | 様々なオプション戦略の損益構造を視覚化 |
| 🔄 **ギリシャ指標分析** | デルタ、ガンマ、ベガ、シータ、ローなどのオプションギリシャ指標を計算・視覚化 |
| 📉 **インプライドボラティリティ分析** | ボラティリティスマイル/スキューおよび3Dボラティリティサーフェスを生成 |
| 🔍 **オプションチェーン検索** | 特定の満期日の完全なオプションチェーンデータを表示（yfinanceパッケージ使用） |

## 📋 利用可能なオプション戦略

<table>
  <tr>
    <td><b>🔹 シングル(Single)</b></td>
    <td>コール/プットオプションの基本的なロング/ショートポジション</td>
  </tr>
  <tr>
    <td><b>🔹 カバード(Covered)</b></td>
    <td>カバードコール（株式ロング+コールショート）またはカバードプット（株式ショート+プットショート）</td>
  </tr>
  <tr>
    <td><b>🔹 プロテクティブ(Protective)</b></td>
    <td>プロテクティブプット（株式ロング+プットロング）またはプロテクティブコール（株式ショート+コールロング）</td>
  </tr>
  <tr>
    <td><b>🔹 スプレッド(Spread)</b></td>
    <td>コールまたはプットを使用したブル/ベアスプレッド</td>
  </tr>
  <tr>
    <td><b>🔹 ストラドル(Straddle)</b></td>
    <td>同じ行使価格のコールとプットの両方をロング/ショート</td>
  </tr>
  <tr>
    <td><b>🔹 ストラングル(Strangle)</b></td>
    <td>異なる行使価格のコールとプットの両方をロング/ショート</td>
  </tr>
  <tr>
    <td><b>🔹 ストリップ(Strip)</b></td>
    <td>同じ行使価格でコール1つとプット2つをロング（ベアリッシュ戦略）</td>
  </tr>
  <tr>
    <td><b>🔹 ストラップ(Strap)</b></td>
    <td>同じ行使価格でコール2つとプット1つをロング（ブリッシュ戦略）</td>
  </tr>
  <tr>
    <td><b>🔹 バタフライ(Butterfly)</b></td>
    <td>限定されたリスクとリワードを持つ3レッグ戦略</td>
  </tr>
  <tr>
    <td><b>🔹 ラダー(Ladder)</b></td>
    <td>一定間隔の行使価格を持つマルチレッグ戦略</td>
  </tr>
  <tr>
    <td><b>🔹 ジェイドリザード(Jade Lizard)</b></td>
    <td>プットショート+コールスプレッドショート（ブリッシュ戦略）</td>
  </tr>
  <tr>
    <td><b>🔹 リバースジェイドリザード(Reverse Jade Lizard)</b></td>
    <td>プットスプレッドロング+コールショート（ベアリッシュ戦略）</td>
  </tr>
  <tr>
    <td><b>🔹 コンドル(Condor)</b></td>
    <td>限定されたリスクとリワードを持つ4レッグ戦略</td>
  </tr>
</table>

## 🛠️ インストールと実行

```bash
# リポジトリをクローン
git clone https://github.com/parkminhyung/Option-Calculator.git
cd Option-Calculator

# 必要なパッケージをインストール
pip install streamlit pandas numpy yfinance plotly scipy

# アプリケーションを実行
streamlit run app.py
```

## 📝 使用方法

1. サイドバーに株式ティッカーシンボルを入力し、「Fetch Data」ボタンをクリックします。

2. オプション戦略、行使価格、満期日などの希望するパラメータを設定します。

3. 「Show Plot」ボタンをクリックして、戦略の損益構造とギリシャ指標を視覚化します。

4. 「Option Chain」タブを使用して、特定の満期日のオプションチェーン情報を表示します。

## 📌 注意事項

> ⚠️ **オプションチェーン情報はyfinanceパッケージに依存しているため、一部の市場や証券ではデータが制限される場合があります。**

> 💡 **Option Prices下に表示される価格はブラック・ショールズモデルに基づく理論価格です。オプションの前の+は買い、-は売りを示します。例えば、- callはコールの売りを意味し、+2x callはコールを2倍買うことを意味します。**
