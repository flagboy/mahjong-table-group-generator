# 麻雀卓組み生成プログラム / Mahjong Table Group Generator

麻雀大会用の卓組み（テーブル配置）を自動生成するプログラムです。参加者が均等に対戦し、待機回数も公平になるよう設計されています。

## 特徴

- 🎯 **公平な組み合わせ**: 同じペアの対戦回数を最小化
- ⏱️ **均等な待機時間**: 全員の待機回数を均等に配分
- 🖥️ **Webインターフェース**: ブラウザから簡単に操作可能
- 📊 **表形式の結果表示**: 縦軸に卓番号、横軸に節を配置した見やすい表示
- 🎲 **5人打ち対応**: オプションで5人打ちも可能
- 📈 **統計情報**: ペア対戦回数や待機回数の統計を表示

## 必要環境

- Python 3.12以上
- Flask 3.0.0

## インストール

1. リポジトリをクローン
```bash
git clone https://github.com/flagboy/mahjong-table-group-generator.git
cd mahjong-table-group-generator
```

2. 依存関係をインストール
```bash
pip install -r requirements.txt
```

## 使い方

### Webインターフェース版

1. アプリケーションを起動
```bash
python app.py
```
または
```bash
./run.sh
```

2. ブラウザで http://127.0.0.1:5000 にアクセス

3. 参加者名を1行に1人ずつ入力

4. 節数（ラウンド数）を入力

5. 必要に応じて「5人打ちを許可」にチェック

6. 「卓組を生成」ボタンをクリック

### コマンドライン版

基本的な使い方：
```bash
python table_group.py [参加人数] [節数]
```

5人打ちを許可する場合：
```bash
python table_group.py [参加人数] [節数] --five
```

例：
```bash
# 9人で9節の卓組を生成
python table_group.py 9 9

# 13人で5節、5人打ちありの卓組を生成
python table_group.py 13 5 --five
```

### 最適化版（大規模大会向け）

線形計画法を使用した高度な最適化版：
```bash
python table_group_optimized.py [参加人数] [節数]
```

## 結果の見方

### Web版の表示

結果は以下の形式で表示されます：

| 卓 | 第1節 | 第2節 | 第3節 | ... |
|---|-------|-------|-------|-----|
| 卓1 | 田中<br>鈴木<br>佐藤<br>高橋 | 山田<br>伊藤<br>渡辺<br>中村 | ... | ... |
| 卓2 | 山田<br>伊藤<br>渡辺<br>中村 | 田中<br>鈴木<br>佐藤<br>高橋 | ... | ... |
| 待機 | 小林 | 加藤 | ... | ... |

### 統計情報

- **最大同卓回数**: 同じペアが最大何回同じ卓になったか
- **同卓回数分布**: 各回数ごとのペア数
- **待機回数統計**: 各プレイヤーの待機回数（5人で5人打ちなしの場合）

## アルゴリズム

### 基本アルゴリズム
1. 各節で待機回数が最も少ないプレイヤーを優先的に待機に割り当て
2. 同じ待機回数の場合は、ラウンドローテーション方式で決定的に選択
3. プレイヤーのペア対戦回数を最小化するよう組み合わせを最適化

### 最適化版アルゴリズム
- PuLP（線形計画法ライブラリ）を使用
- ペア対戦回数の最大値を最小化する目的関数
- 大規模な参加者数でも効率的に処理

## ファイル構成

- `app.py` - Flask Webアプリケーション
- `table_group_web.py` - Web版用の卓組生成ロジック
- `table_group.py` - コマンドライン版
- `table_group_optimized.py` - 最適化版（線形計画法使用）
- `templates/index.html` - WebインターフェースのHTMLテンプレート
- `static/js/app.js` - フロントエンドJavaScript
- `static/css/style.css` - スタイルシート

## 本番環境へのデプロイ

### Gunicornを使用したデプロイ

1. 依存関係をインストール（Gunicornを含む）
```bash
pip install -r requirements.txt
```

2. Gunicornで起動
```bash
./run_production.sh
```
または
```bash
gunicorn --config gunicorn_config.py wsgi:app
```

### Nginxとの連携

`nginx.conf.example`を参考にNginxを設定してください：

```bash
sudo cp nginx.conf.example /etc/nginx/sites-available/mahjong-table-group
sudo ln -s /etc/nginx/sites-available/mahjong-table-group /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl reload nginx
```

### Systemdサービスとして実行

`systemd.service.example`を参考にサービスファイルを作成：

```bash
sudo cp systemd.service.example /etc/systemd/system/mahjong-table-group.service
sudo systemctl daemon-reload
sudo systemctl enable mahjong-table-group
sudo systemctl start mahjong-table-group
```

### 環境変数

本番環境では以下の環境変数を設定することを推奨：

```bash
export FLASK_ENV=production
export SECRET_KEY=your-secret-key-here
```

## ライセンス

このプロジェクトはMITライセンスの下で公開されています。

## 貢献

バグ報告や機能提案は[Issues](https://github.com/flagboy/mahjong-table-group-generator/issues)からお願いします。

プルリクエストも歓迎します！

## 作者

[@flagboy](https://github.com/flagboy)