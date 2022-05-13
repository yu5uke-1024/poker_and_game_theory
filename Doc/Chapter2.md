# ポーカーのミニゲーム

ポーカーの一種であるテキサスホールデムは、\*情報集合数が多く、手法やアルゴリズムを試すのに時間的コストや計算機コストがかかります。

そこで、手法の有効性を試す時には、情報集合数の少ないポーカーのミニゲームを用います。

この資料では、Kuhn Poker (クーン・ポーカー) と Leduc Poker (ルダック・ポーカー) を使用します。

表 1 に各ゲームの情報集合数を記載しています。[1]

また、情報集合数は、各プレイヤから区別できない状態の集合の個数です。例えば、Kuhn Poker の例ですと自分が Q を持っていた時に、相手が J を持っているか K を持っているかが自分からは判別できないため(Q, J), (Q, K) という状態は同じ集合になります。(Kuhn Poker の詳細な説明は後述)

<img width="1164" alt="ScreenShot 2022-05-13 9 44 23" src="https://user-images.githubusercontent.com/63486375/168189513-4d97e616-929b-4ae6-ba62-d3dd3f139d95.png">

ここから、Kuhn Poker と Leduc Poker のルール説明をしてきます。
説明の順番ですが、まず 2 人対戦の Kuhn Poker と Leduc Poker の説明を行った後に、多人数対戦への拡張の仕方を説明します。

# Kuhn Poker

<p align="center">
  <img src="https://user-images.githubusercontent.com/63486375/168191528-df281145-02c6-48ea-bd17-b8686ca7f06e.jpeg">
<p>

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
