# KDEL 学位論文用 LaTeX テンプレート

OCU KDEL 学位論文および公聴会レジュメ用 LaTeX テンプレート

## 構成
| File / Directory | Description                                |
| :--------------- | :----------------------------------------- |
| resume           | 公聴会レジュメ用テンプレート               |
| thesis           | 学位論文用テンプレート                     |
| sample/resume    | 公聴会レジュメサンプル                     |
| sample/thesis    | 学位論文サンプル                           |
| .latexmkrc       | latexmk 用設定ファイル（OverLeaf 用）      |
| README.md        | 本ドキュメント                             |

resume，thesis はそれぞれレジュメおよび論文のテンプレートのみ．  
sample/resume および sample/thesis はテンプレートを用いた動作確認用のサンプルになっている（2017 年度加守田修論レジュメ，2016 年度小林修論）．    



## インストール

### 環境 (OS, TeX / LaTeX ディストリビューション)

| OS       | Distribution                         |
| :------- | :----------------------------------- |
| Windows  | TeX Live 2017, 2018                  |
| macOS    | MacTeX 2017, 2018                    |
| Linux    | TeX Live 2017, 2018 (manual install) |
| OverLeaf | V2 (TeX Live 2017 based)             |

インストール方法については各自 TeX Wiki ([Windows](
https://texwiki.texjp.org/?TeX%20Live%2FWindows) / [macOS](https://texwiki.texjp.org/?MacTeX) / [Linux](https://texwiki.texjp.org/?Linux#texlive)) などで確認すること．  
なお．[OverLeaf](https://www.overleaf.com) V2 で LuaLaTeX を用いる場合，`New File` → `From External URL` で最新版の [lltjext.sty](http://mirrors.ctan.org/macros/luatex/generic/luatexja/src/lltjext.sty) を追加する必要がある．


### テンプレート取得
ホストされているリポジトリからローカルに `git clone <url>` すればよい．  

*自身に権限のあるリモートリポジトリを設定してバージョン管理することを勧める．*  
*（fork したものを編集するか，fork せず clone したものに別途用意したリモートリポジトリを `git remote add <name> <url>` で追加してそちらで管理するか，など）*


## 使い方


### レジュメ用テンプレート

構成は以下の通り．

| File / Directory | Description                                |
| :--------------- | :----------------------------------------- |
| resume.tex       | レジュメ本体（コンパイル対象）             |
| preamble.tex     | レジュメ設定ファイル（基本的に変更しない） |
| .latexmkrc       | latexmk 設定ファイル（基本的に変更しない） |


#### タイトル・著者などの情報の入力
resume.tex 冒頭の \title や \author などのマクロに，自身の論文のタイトルなどを設定する．

#### 本文の編集  
resume.tex に内容を記述する．  

必要なパッケージの導入やマクロの定義はこのファイルのプレアンブル（`\documentclass{...}` から `\begin{document}` の間）で行うこと． 
*`\usepackage` や `\newcommand`, `\renewcommand` をする場合，preamble.tex 内に同じ引数のものがないか確認すること．*  
*ただし，既に preamble.tex で `\usepackage` 等されているが調整のためにオプション引数を少し変えたい，といった場合には preamble.tex 側の該当箇所を編集する．*

#### コンパイル   
resume ディレクトリにて下記のコマンドを実行．

```sh
latexmk resume.tex 
```

これにより resume.pdf が生成される．
生成されない場合はエラーがあるため，これを修正し再度上記コマンドでコンパイル．    
エラーの内容・場所は，標準出力および .log ファイルから確認できる．  
エラーを修正しても latexmk が異常終了する場合は，下記コマンドで一時ファイルなどを一度削除してみるとよい．

```sh
latexmk -C
```


### 学位論文用テンプレート

構成は以下の通り．

|  File / Directory   |                Description                 |
| :------------------ | :----------------------------------------- |
| thesis.tex          | 論文全体用（コンパイル対象）               |
| title.tex           | 表紙・背表紙用                             |
| preamble.tex        | 論文設定ファイル（基本的に変更しない）     |
| preamble_user.tex   | 独自設定記述用ファイル                     |
| abstract.tex        | 概要記述用                                 |
| chapter**.tex       | 各章記述用                                 |
| appendix*.tex       | 付録記述用                                 |
| acknowledgement.tex | 謝辞記述用                                 |
| bibliography.tex    | 参考文献記述用                             |
| achievement.tex     | 業績記述用                                 |
| .latexmkrc          | latexmk 設定ファイル（基本的に変更しない） |


#### タイトル・著者などの情報の入力
preamble_user.tex 冒頭の \title や \author などのマクロに，自身の論文のタイトルなどを設定する．

#### 各ソースファイル編集
各章・概要・謝辞・参考文献などは，chapter**.tex など個別のファイルに分けて記述する．  
thesis.tex がこれらをまとめる役割をしており，順番通りに `\include` することで論文全体を構成する（なお thesis.tex の `\usepackage{docmute}` は消さないこと）．  
タイトル設定やパッケージ導入，マクロ定義などは preamble_user.tex で行う．
*`\usepackage` や `\newcommand`, `\renewcommand` をする場合，preamble.tex 内に同じ引数のものがないか確認すること．*  
*ただし，既に preamble.tex で `\usepackage` 等されているが調整のためにオプション引数を少し変えたい，といった場合には preamble.tex 側の該当箇所を編集する．*

#### 論文本体のコンパイル  
thesis.tex をコンパイルすることで論文全体がコンパイルされる．
```sh
latexmk thesis.tex
```
なお，章ごとにコンパイルすることも可能（ただし相互参照は 未解決になる）．
```sh
latexmk chapter01.tex
```

#### 表紙・背表紙のコンパイル  
title.tex は，論文を綴じるファイルに貼り付ける表紙と背表紙を出力するためのもの．  
以下のコマンドでコンパイル．

```sh
latexmk title.tex
```

## TeX / LaTeX に慣れている人向け
### 対応する TeX / LaTeX エンジン
現状のテンプレート（プレアンブルの内容）は pLaTeX，upLaTeX，LuaLaTeX および XeTeX に対応している．  

ただし，thesis テンプレートは背表紙出力の際に部分的に縦書きをしている関係で XeLaTeX ではコンパイルエラーとなる．  
（pLaTeX における plext パッケージに該当するものが見当たらないため互換性がなくなっている）  
TODO: XeLaTeX 対応

### .latexmkrc
.latexmkrc にはこれらのエンジン向けのものを設定済み（なお pdfLaTeX，`dvips` + `ps2pdf` 用の設定もしているが，動作未確認）．  
デフォルトでは upLaTeX が利用されるが，`latexmk` のコマンド引数などで切替可能．  

### 文書の分割とドキュメントクラス設定の共有  
thesis.tex において docmute パッケージを読み込むことで実現している．  
従来の `\include` は，その場所に読み込み先ファイルの内容を *そのまま* 展開する．  
docmute パッケージは，`\include` の仕様を変更するパッケージで，読み込み先ファイル内のドキュメントクラス設定およびプレアンブルを削除したものを展開する．  
なお docmute パッケージを用いない場合，`\include` するファイルにはドキュメントクラス設定およびプレアンブルを書けず，したがって chapter**.tex などを個別でコンパイルすることはできない．

### OverLeaf V2 における LuaLaTeX の使用について
同梱の .latexmkrc をプロジェクトのトップディレクトリに配置していれば，`Menu` → `Setting` → `Compiler` で LuaLaTeX を選択しておくことで利用できる．  
ただし，TeX Live 2017 時点では LuaTeX-ja の付属パッケージ lltjext.sty （背表紙出力に使用するため読み込んでいる）にバグが存在し，array など一部の環境での出力が 90 度回転してしまう．  
このバグは最新版では修正されており，CTAN より最新版を取得して読み込ませておくことで対応可能．  
