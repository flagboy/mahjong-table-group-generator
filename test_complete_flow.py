#!/usr/bin/env python3
"""完全なフローをテスト（デバッグ情報付き）"""

import requests
import json
from collections import defaultdict
from itertools import combinations


def test_complete_flow():
    """Webアプリケーションの完全なフローをテスト"""
    
    print("=== 12人6回戦の完全なフローテスト ===\n")
    
    # テストデータ
    players = ["P1", "P2", "P3", "P4", "P5", "P6", 
               "P7", "P8", "P9", "P10", "P11", "P12"]
    
    # リクエスト送信
    url = "http://localhost:5001/generate"
    data = {
        "players": players,
        "rounds": 6,
        "allow_five": False
    }
    
    print("1. リクエスト送信")
    print(f"   URL: {url}")
    print(f"   データ: {json.dumps(data, ensure_ascii=False)}")
    
    response = requests.post(url, json=data)
    result = response.json()
    
    if not result.get('success'):
        print(f"\nエラー: {result}")
        return
    
    # レスポンスの詳細を表示
    print("\n2. レスポンス受信")
    formatted = result['results']
    stats = formatted['statistics']['pair_statistics']
    
    print(f"\n3. 統計情報（APIからの返答）:")
    print(f"   最小同卓回数: {stats['min_count']}回")
    print(f"   最大同卓回数: {stats['max_count']}回")
    print(f"   カバレッジ: {stats['coverage']:.1f}%")
    print(f"   分布: {stats['distribution']}")
    
    # 手動で再計算して検証
    print(f"\n4. 手動検証:")
    pair_count = defaultdict(int)
    
    for round_data in formatted['rounds_data']:
        for table_data in round_data['tables']:
            players_in_table = table_data['players']
            # プレイヤー名をインデックスに変換
            indices = [players.index(p) for p in players_in_table]
            
            for i in range(len(indices)):
                for j in range(i+1, len(indices)):
                    pair = tuple(sorted([indices[i], indices[j]]))
                    pair_count[pair] += 1
    
    # 全ペアの統計を計算
    all_pairs = list(combinations(range(12), 2))
    all_counts = [pair_count.get(pair, 0) for pair in all_pairs]
    
    manual_min = min(all_counts)
    manual_max = max(all_counts)
    manual_coverage = sum(1 for c in all_counts if c > 0) / len(all_pairs) * 100
    
    # 分布を計算
    manual_distribution = defaultdict(int)
    for count in all_counts:
        manual_distribution[count] += 1
    
    print(f"   手動計算の最小: {manual_min}回")
    print(f"   手動計算の最大: {manual_max}回")
    print(f"   手動計算のカバレッジ: {manual_coverage:.1f}%")
    print(f"   手動計算の分布: {dict(manual_distribution)}")
    
    # 一致確認
    print(f"\n5. 検証結果:")
    if stats['min_count'] == manual_min:
        print("   ✓ 最小同卓回数が一致")
    else:
        print(f"   ✗ 最小同卓回数が不一致: API={stats['min_count']}, 手動={manual_min}")
    
    if stats['max_count'] == manual_max:
        print("   ✓ 最大同卓回数が一致")
    else:
        print(f"   ✗ 最大同卓回数が不一致: API={stats['max_count']}, 手動={manual_max}")
    
    # 0回同卓のペアを特定
    if manual_min == 0:
        print(f"\n6. 0回同卓のペアの詳細:")
        zero_pairs = []
        for pair in all_pairs:
            if pair_count.get(pair, 0) == 0:
                i, j = pair
                zero_pairs.append((players[i], players[j]))
        
        print(f"   0回同卓のペア: {zero_pairs}")
        print(f"   合計: {len(zero_pairs)}ペア")
    
    # JavaScriptで表示される内容を確認
    print(f"\n7. ブラウザでの表示内容:")
    print(f"   distribution オブジェクト: {stats['distribution']}")
    print(f"   min_count: {stats['min_count']}")
    
    if stats['min_count'] == 0 and '0' not in stats['distribution']:
        print("\n   ⚠️ 問題発見: min_countは0だが、distributionに'0'キーが存在しない")
        print("   これがユーザーが報告している問題の原因の可能性があります")


if __name__ == "__main__":
    test_complete_flow()