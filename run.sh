#!/bin/bash

# 仮想環境が存在しない場合は作成
if [ ! -d "venv" ]; then
    echo "仮想環境を作成しています..."
    python -m venv venv
fi

# 仮想環境をアクティベート
source venv/bin/activate

# 必要なパッケージをインストール
pip install -r requirements.txt

# Flaskアプリケーションを起動
echo "アプリケーションを起動しています..."
echo "ブラウザで http://localhost:5000 にアクセスしてください"
python app.py