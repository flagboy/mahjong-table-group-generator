#!/usr/bin/env python3
"""麻雀卓組生成Webアプリケーション"""

from flask import Flask, render_template, request, jsonify
from table_group_web_universal import TableGroupGenerator

app = Flask(__name__)

@app.route('/')
def index():
    """トップページ"""
    return render_template('index.html')

@app.route('/generate', methods=['POST'])
def generate():
    """卓組を生成"""
    try:
        data = request.json
        player_names = data.get('players', [])
        rounds = int(data.get('rounds', 1))
        allow_five = data.get('allow_five', False)
        
        if len(player_names) < 4:
            return jsonify({'error': '参加者は4人以上必要です'}), 400
        
        if rounds < 1:
            return jsonify({'error': '回数は1以上必要です'}), 400
        
        generator = TableGroupGenerator(player_names, rounds, allow_five)
        results = generator.generate()
        formatted_results = generator.format_results(results)
        
        return jsonify({
            'success': True,
            'results': formatted_results
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # 開発サーバーの場合
    app.run(debug=False, host='0.0.0.0', port=5001)