# ポーカーのミニゲーム

ポーカーの一種であるテキサスホールデムは、\*情報集合数が多く、手法やアルゴリズムを試すのに時間的コストや計算機コストがかかります。

そこで、手法の有効性を試す時には、情報集合数の少ないポーカーのミニゲームを用います。

この資料では、Kuhn Poker (クーン・ポーカー) と Leduc Poker (ルダック・ポーカー) を使用します。

表 1 に各ゲームの情報集合数を記載しています。[1]

また、情報集合数は、各プレイヤから区別できない状態の集合の個数です。例えば、Kuhn Poker の例ですと自分が Q を持っていた時に、相手が J を持っているか K を持っているかが自分からは判別できないため(Q, J), (Q, K) という状態は同じ集合になります。(Kuhn Poker の詳細な説明は後述)

表 1 :　各ゲームの情報集合数

| Kuhn Poker | Leduc Poker | Heads-up Limit<br>Texas Hod'em | Heads-up No Limit<br>Texas Hod'em |
| ---------- | ----------- | ------------------------------ | --------------------------------- |
| 10^1 (12)  | 10^2 (288)  | 10^13                          | 10^160                            |

ここから、Kuhn Poker と Leduc Poker のルール説明をしてきます。
説明の順番ですが、まず 2 人対戦の Kuhn Poker と Leduc Poker の説明を行った後に、多人数対戦への拡張の仕方を説明します。

# Kuhn Poker

<p align="center">
  <img src="https://user-images.githubusercontent.com/63486375/168191528-df281145-02c6-48ea-bd17-b8686ca7f06e.jpeg">
<p>

[準備]

準備 0. J,Q,K (強さ: J < Q <K) を 1 枚ずつ用意し、各プレイヤに 2 枚のチップを渡す

準備 1. 1 枚ずつカードを配り、チップ 1 枚ずつポットに置く

![a](https://user-images.githubusercontent.com/63486375/168206301-b9417f92-ced0-42df-9c73-f100613de5b0.png)

[ゲーム]

ゲーム 1. プレイヤ A が Pass か Bet (チップ 1 枚) を選択

![b](https://user-images.githubusercontent.com/63486375/168206313-4b36afca-769a-43ed-a8d8-21af82210b4f.png)

ゲーム 2. プレイヤ B が Pass か Bet (チップ 1 枚) を選択

![c](https://user-images.githubusercontent.com/63486375/168206316-d89090cc-3c8f-4050-9ea8-ae10b27cb038.png)

ゲーム 3. Pass, Bet と行動が続いた場合、プレイヤ A が Pass か bet (チップ 1 枚) を選択

![d](https://user-images.githubusercontent.com/63486375/168206318-c36be7ba-a97a-4464-90e1-07b5ec7d1c48.png)

[終了条件]

終了条件 1. 両プレイヤが Bet をした or Pass をした場合、手札を見せ合い、強い方がポットを獲得

![e](https://user-images.githubusercontent.com/63486375/168206322-5e5ef9a8-e474-4cbe-808c-aac5e7566ead.png)

終了条件 2. 片方が Bet をしている状態で、片方が Pass を選択した場合、Bet したプレイヤがポットを獲得

![f](https://user-images.githubusercontent.com/63486375/168206324-f113a08d-b3e2-41d4-96db-cfc1ae092317.png)

# 引用文献

[1] 河村圭悟 et al. 未知環境における多人数不完全情報ゲームの戦略計算. ゲームプログラミングワーク ショップ 2017 論文集. p.80-87. 2017.

<p align="center">
    <table border="1">
        <tr>
            <td><a href="https://github.com/yu5uke-1024/poker_and_game_theory/blob/main/Doc/Chapter1.md">previous chapter</a></td>
            <td><a>♤</a></td>
            <td><a href="https://github.com/yu5uke-1024/poker_and_game_theory/blob/main/Doc/Chapter2.md">next chapter</a></td>
        </tr>
    </table>
<p>
