#!/usr/bin/env python3
"""本番環境のAPIをテスト"""

import requests
import json
from collections import defaultdict
from itertools import combinations


def test_production_api():
    """本番環境のAPIに12人6回戦のリクエストを送信"""
    
    url = "https://mahjong-table-group-generator.onrender.com/generate"
    
    # 12人のプレイヤー
    data = {
        "players": ["P1", "P2", "P3", "P4", "P5", "P6", 
                   "P7", "P8", "P9", "P10", "P11", "P12"],
        "rounds": 6,
        "allow_five": False
    }
    
    print("本番環境APIテスト（12人6回戦）")
    print("="*60)
    print(f"URL: {url}")
    print(f"データ: {json.dumps(data, ensure_ascii=False)}")
    
    try:
        # POSTリクエストを送信
        response = requests.post(url, json=data, timeout=30)
        
        if response.status_code == 200:
            result = response.json()
            
            if result.get('success'):
                formatted = result['results']
                
                print("\n成功！結果を受信しました。")
                
                # 統計を表示
                stats = formatted['statistics']['pair_statistics']
                print(f"\n統計（APIレスポンス）:")
                print(f"  最小同卓回数: {stats['min_count']}回")
                print(f"  最大同卓回数: {stats['max_count']}回")
                print(f"  カバレッジ: {stats['coverage']:.1f}%")
                print(f"  分布: {stats['distribution']}")
                
                # 各ラウンドの詳細を確認
                print("\n各ラウンドの詳細:")
                pair_count = defaultdict(int)
                
                for round_data in formatted['rounds_data']:
                    round_num = round_data['round_num']
                    print(f"\n第{round_num}回戦:")
                    
                    for table_data in round_data['tables']:
                        table_num = table_data['table_num']
                        players = table_data['players']
                        print(f"  卓{table_num}: {', '.join(players)}")
                        
                        # ペアをカウント（手動で再計算）
                        player_indices = [data['players'].index(p) for p in players]
                        for i in range(len(player_indices)):
                            for j in range(i+1, len(player_indices)):
                                pair = tuple(sorted([player_indices[i], player_indices[j]]))
                                pair_count[pair] += 1
                
                # 手動計算の統計
                all_pairs = list(combinations(range(12), 2))
                all_counts = [pair_count.get(pair, 0) for pair in all_pairs]
                
                manual_min = min(all_counts)
                manual_max = max(all_counts)
                manual_coverage = sum(1 for c in all_counts if c > 0) / len(all_pairs) * 100
                
                manual_distribution = defaultdict(int)
                for count in all_counts:
                    manual_distribution[count] += 1
                
                print(f"\n手動計算の統計:")
                print(f"  最小: {manual_min}回")
                print(f"  最大: {manual_max}回")
                print(f"  カバレッジ: {manual_coverage:.1f}%")
                print(f"  分布: {dict(manual_distribution)}")
                
                # 0回同卓のペアを特定
                if manual_min == 0:
                    print(f"\n⚠️ 0回同卓のペアが存在します！")
                    zero_pairs = []
                    for i, j in all_pairs:
                        if pair_count.get((i, j), 0) == 0:
                            zero_pairs.append((data['players'][i], data['players'][j]))
                    print(f"0回同卓のペア: {zero_pairs}")
                    print(f"合計: {len(zero_pairs)}ペア")
                
                # APIの統計と手動計算が一致するか確認
                print(f"\n検証結果:")
                if stats['min_count'] != manual_min:
                    print(f"⚠️ 統計の不一致: API最小={stats['min_count']}, 手動={manual_min}")
                    print("これが問題の原因です！")
                else:
                    print("✓ 統計が一致しています")
                
            else:
                print(f"\nエラー: {result}")
        else:
            print(f"\nHTTPエラー: {response.status_code}")
            print(response.text)
            
    except Exception as e:
        print(f"\n接続エラー: {str(e)}")
        print("注: Renderの無料プランは初回アクセス時に起動に時間がかかる場合があります")


if __name__ == "__main__":
    test_production_api()