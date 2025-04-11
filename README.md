아래는 사용자가 제공한 README.md 파일에 업데이트된 내용을 각 언어별로 추가한 결과입니다. 업데이트 사항은 다음과 같습니다:

1. **Strategy performance에 win rate 추가**: 이제 각 옵션 전략의 승률을 계산하여 표시합니다.
2. **Plot에 probability 곡선 추가**: 손익 구조 플롯에 전략 결과의 확률 분포 곡선을 추가했습니다.
3. **Volatility 계산 변경**: 기존 52주 변동성 대신 델타 중립 내재 변동성을 사용하여 옵션 이론 가격을 계산합니다.

이 업데이트 사항을 README.md의 각 언어 섹션에 명확히 반영하기 위해, 스크린샷 다음에 "## 새로운 기능" 섹션을 추가하고, "## 참고 사항"에서 옵션 가격 계산 방법에 대한 설명을 수정했습니다. 아래는 각 언어별로 업데이트된 내용을 반영한 예시입니다.

---

### 한국어

<a id="한국어"></a>

<div align="center">
  
# 옵션 계산기

### 금융 투자자를 위한 강력한 옵션 전략 시뮬레이션 및 분석 도구
  
</div>

<p align="center">
  <img width="800" alt="메인 화면" src="https://github.com/user-attachments/assets/4b458818-ff41-4510-a9cd-af7a28f6cbb9" />
</p>

<details open>
<summary><b>📊 스크린샷 더 보기</b></summary>
<p align="center">
  <img width="1578" alt="image" src="https://github.com/user-attachments/assets/2afdd40b-5f90-485c-bee4-d0e649ac0200" />
  
  <img width="1562" alt="image" src="https://github.com/user-attachments/assets/d4e959ce-826d-4a3a-b340-48786f4d8c85" />


</p>
</details>

## 새로운 기능

- **전략 성과 분석**: 각 옵션 전략의 승률을 계산하여 표시합니다.
- **확률 곡선 시각화**: 손익 구조 플롯에 전략의 다양한 결과에 대한 확률 분포 곡선을 추가했습니다.
- **향상된 옵션 가격 책정**: 옵션 이론 가격 계산에 52주 변동성 대신 델타 중립 내재 변동성을 사용합니다.

## 주요 기능

| 기능 | 설명 |
|------|------|
| **실시간 주식 데이터** | 주식 티커 심볼을 입력하여 실시간 주가 및 기본 정보 확인 |
| **옵션 전략 시뮬레이션** | 다양한 옵션 전략의 손익 구조를 시각화 |
| **그릭스 분석** | 델타, 감마, 베가, 세타, 로 등 옵션 그릭스 계산 및 시각화 |
| **내재 변동성 분석** | 변동성 스마일/스큐 및 3D 변동성 표면 생성 |
| **옵션 체인 조회** | 특정 만기일에 대한 전체 옵션 체인 데이터 확인 |
| **볼륨 차트 및 풋콜 비율** | 각 옵션의 볼륨 차트와 풋콜 비율 시각화 |

(기존 "사용 가능한 옵션 전략", "설치 및 실행", "사용 방법" 섹션은 변경 없음)

## 참고 사항

> **옵션 체인 정보와 볼륨 데이터는 yfinance 패키지에 의존하므로 일부 시장이나 종목에서는 데이터가 제한되거나 정확하지 않을 수 있습니다.**

> **Option Prices 하단에 나타나는 가격은 델타 중립 내재 변동성을 사용하여 블랙숄즈모형을 기반으로 도출한 이론가격입니다. 옵션 앞 +는 매수, -는 매도를 나타냅니다. 예를 들어 - call은 콜매도를 의미하며, +2x call은 콜을 2배로 매수한다는 뜻입니다.**

---

### English

<a id="english"></a>

<div align="center">
  
# Options Calculator

### A powerful option strategy simulation and analysis tool for financial investors
  
</div>

<p align="center">
  <img width="800" alt="Main Screen" src="https://github.com/user-attachments/assets/4b458818-ff41-4510-a9cd-af7a28f6cbb9" />
</p>

<details>
<summary><b>📊 More Screenshots</b></summary>
<p align="center">
  <img width="800" alt="화면 2" src="https://github.com/user-attachments/assets/44535153-5417-4d12-b9d8-d080582c0719" />
  <br><br>
  <img width="800" alt="화면 3" src="https://github.com/user-attachments/assets/2483ddce-1df5-4aae-9c14-10543384181f" />
  <br><br>
  <img width="800" alt="화면 4" src="https://github.com/user-attachments/assets/ef85e353-f66f-40a4-b79d-a5282f7c1d82" />
  <br><br>
  <img width="800" alt="화면 5" src="https://github.com/user-attachments/assets/aa6cc085-0502-4394-8f74-bbca63f865a9" />
  <br><br>
  <img width="800" alt="image" src="https://github.com/user-attachments/assets/f5c129b4-aa6c-4dfc-9fb6-fbad77ff3762" />
</p>
</details>

## What's New

- **Strategy Performance Analysis**: Now calculates and displays the win rate for each option strategy.
- **Probability Curve Visualization**: Added a probability distribution curve to the profit/loss plot for strategy outcomes.
- **Enhanced Option Pricing**: Uses delta-neutral implied volatility instead of 52-week volatility for theoretical option pricing.

## Key Features

| Feature | Description |
|---------|-------------|
| **Real-time Stock Data** | Input a stock ticker symbol to get real-time prices and basic information |
| **Option Strategy Simulation** | Visualize the profit/loss structure of various option strategies |
| **Greeks Analysis** | Calculate and visualize option Greeks including Delta, Gamma, Vega, Theta, and Rho |
| **Implied Volatility Analysis** | Generate volatility smiles/skews and 3D volatility surfaces |
| **Option Chain Lookup** | View complete option chain data for specific expiry dates |
| **Volume Chart and Put-Call Ratio** | Visualize volume chart and put-call ratio for each option |

(Existing "Available Option Strategies", "Installation and Running", "How to Use" sections remain unchanged)

## Notes

> **Option chain information and volume data depend on the yfinance package, so data may be limited or inaccurate for some markets or securities.**

> **The prices shown under Option Prices are theoretical prices based on the Black-Scholes model using delta-neutral implied volatility. The + sign in front of options indicates buying, while - indicates selling. For example, - call means selling a call, and +2x call means buying two call options.**

---

### 中文

<a id="中文"></a>

<div align="center">
  
# 期权计算器

### 为金融投资者提供的强大期权策略模拟和分析工具
  
</div>

<p align="center">
  <img width="800" alt="主屏幕" src="https://github.com/user-attachments/assets/4b458818-ff41-4510-a9cd-af7a28f6cbb9" />
</p>

<details>
<summary><b>📊 更多截图</b></summary>
<p align="center">
  <img width="800" alt="화면 2" src="https://github.com/user-attachments/assets/44535153-5417-4d12-b9d8-d080582c0719" />
  <br><br>
  <img width="800" alt="화면 3" src="https://github.com/user-attachments/assets/2483ddce-1df5-4aae-9c14-10543384181f" />
  <br><br>
  <img width="800" alt="화면 4" src="https://github.com/user-attachments/assets/ef85e353-f66f-40a4-b79d-a5282f7c1d82" />
  <br><br>
  <img width="800" alt="화면 5" src="https://github.com/user-attachments/assets/aa6cc085-0502-4394-8f74-bbca63f865a9" />
  <br><br>
  <img width="800" alt="image" src="https://github.com/user-attachments/assets/f5c129b4-aa6c-4dfc-9fb6-fbad77ff3762" />
</p>
</details>

## 新功能

- **策略表现分析**: 现在计算并显示每种期权策略的胜率。
- **概率曲线可视化**: 在损益图中添加了策略结果的概率分布曲线。
- **增强的期权定价**: 使用delta中性隐含波动率代替52周波动率进行理论期权定价。

## 主要功能

| 功能 | 描述 |
|------|------|
| **实时股票数据获取** | 输入股票代码获取实时价格和基本信息 |
| **期权策略模拟** | 可视化各种期权策略的盈亏结构 |
| **希腊字母分析** | 计算并可视化期权希腊字母，包括Delta、Gamma、Vega、Theta和Rho |
| **隐含波动率分析** | 生成波动率微笑/偏斜和3D波动率曲面 |
| **期权链查询** | 查看特定到期日的完整期权链数据 |
| **交易量图表和看跌/看涨比率** | 可视化每个期权的交易量图表和看跌/看涨比率 |

(现有“可用期权策略”、“安装和运行”、“使用方法”部分保持不变)

## 注意事项

> **期权链信息和交易量数据依赖于yfinance包，因此某些市场或证券的数据可能有限或不准确。**

> **Option Prices下显示的价格是使用delta中性隐含波动率基于Black-Scholes模型的理论价格。期权前的+号表示买入，-号表示卖出。例如，-call表示卖出认购期权，+2x call表示买入两个认购期权。**

---

### 日本語

<a id="日本語"></a>

<div align="center">
  
# オプション計算機

### 金融投資家のための強力なオプション戦略シミュレーションおよび分析ツール
  
</div>

<p align="center">
  <img width="800" alt="メイン画面" src="https://github.com/user-attachments/assets/4b458818-ff41-4510-a9cd-af7a28f6cbb9" />
</p>

<details>
<summary><b>📊 その他のスクリーンショット</b></summary>
<p align="center">
  <img width="800" alt="화면 2" src="https://github.com/user-attachments/assets/44535153-5417-4d12-b9d8-d080582c0719" />
  <br><br>
  <img width="800" alt="화면 3" src="https://github.com/user-attachments/assets/2483ddce-1df5-4aae-9c14-10543384181f" />
  <br><br>
  <img width="800" alt="화면 4" src="https://github.com/user-attachments/assets/ef85e353-f66f-40a4-b79d-a5282f7c1d82" />
  <br><br>
  <img width="800" alt="화면 5" src="https://github.com/user-attachments/assets/aa6cc085-0502-4394-8f74-bbca63f865a9" />
  <br><br>
  <img width="800" alt="image" src="https://github.com/user-attachments/assets/f5c129b4-aa6c-4dfc-9fb6-fbad77ff3762" />
</p>
</details>

## 新機能

- **戦略パフォーマンス分析**: 各オプション戦略の勝率を計算して表示します。
- **確率曲線の視覚化**: 損益プロットに戦略結果の確率分布曲線を追加しました。
- **強化されたオプション価格設定**: 理論オプション価格の計算に52週ボラティリティの代わりにデルタ中立インプライドボラティリティを使用します。

## 主な機能

| 機能 | 説明 |
|------|------|
| **リアルタイム株式データ取得** | 株式ティッカーシンボルを入力してリアルタイム価格と基本情報を取得 |
| **オプション戦略シミュレーション** | 様々なオプション戦略の損益構造を視覚化 |
| **ギリシャ指標分析** | デルタ、ガンマ、ベガ、シータ、ローなどのオプションギリシャ指標を計算・視覚化 |
| **インプライドボラティリティ分析** | ボラティリティスマイル/スキューおよび3Dボラティリティサーフェスを生成 |
| **オプションチェーン検索** | 特定の満期日の完全なオプションチェーンデータを表示 |
| **ボリュームチャートとプット・コール比率** | 各オプションのボリュームチャートとプット・コール比率を視覚化 |

(既存の「利用可能なオプション戦略」、「インストールと実行」、「使用方法」セクションは変更なし)

## 注意事項

> **オプションチェーン情報とボリュームデータはyfinanceパッケージに依存しているため、一部の市場や証券ではデータが制限されたり正確でない場合があります。**

> **Option Prices下に表示される価格はデルタ中立インプライドボラティリティを使用してブラック・ショールズモデルに基づく理論価格です。オプションの前の+は買い、-は売りを示します。例えば、- callはコールの売りを意味し、+2x callはコールを2倍買うことを意味します。**

---

위와 같이 각 언어별로 "## 새로운 기능" 섹션을 스크린샷 다음에 추가하고, "## 참고 사항"에서 옵션 가격 계산 방법에 대한 설명을 업데이트했습니다. 이 변경 사항을 README.md 파일에 반영하면 사용자가 최신 업데이트 내용을 명확히 확인할 수 있습니다. 필요하다면 전체 README.md 파일을 수정된 버전으로 제공할 수도 있으니 말씀해주세요!
