#!/usr/bin/env python3
"""Webサービスの12人6回戦を直接テスト"""

from table_group_web_universal import TableGroupGenerator
from collections import defaultdict
from itertools import combinations


def test_web_service():
    """Webサービスの実際の動作をテスト"""
    
    # 12人の名前リスト
    player_names = [f"Player{i}" for i in range(1, 13)]
    
    # TableGroupGeneratorを作成
    generator = TableGroupGenerator(player_names, 6, allow_five=False)
    
    # 卓組を生成
    results = generator.generate()
    
    print("Webサービスの12人6回戦テスト")
    print("="*60)
    
    # ペアカウント
    pair_count = defaultdict(int)
    
    # 各ラウンドを表示
    for round_num, round_tables in enumerate(results, 1):
        print(f"\n第{round_num}回戦:")
        for table_num, table in enumerate(round_tables, 1):
            # プレイヤーID（0-indexed）を表示
            players_str = ", ".join(f"P{p+1}" for p in sorted(table))
            print(f"  卓{table_num}: {players_str}")
            
            # ペアをカウント
            for p1, p2 in combinations(table, 2):
                pair = tuple(sorted([p1, p2]))
                pair_count[pair] += 1
    
    # 全ペアの統計
    all_pairs = list(combinations(range(12), 2))
    all_counts = [pair_count.get(pair, 0) for pair in all_pairs]
    
    # 分布を計算
    count_distribution = defaultdict(int)
    for count in all_counts:
        count_distribution[count] += 1
    
    min_count = min(all_counts)
    max_count = max(all_counts)
    coverage = sum(1 for c in all_counts if c > 0) / len(all_pairs) * 100
    
    print(f"\n統計:")
    print(f"  総ペア数: {len(all_pairs)}")
    print(f"  最小同卓回数: {min_count}回")
    print(f"  最大同卓回数: {max_count}回")
    print(f"  カバレッジ: {coverage:.1f}%")
    print(f"  分布: {dict(count_distribution)}")
    
    # 0回同卓のペアを特定
    if min_count == 0:
        print(f"\n0回同卓のペア:")
        zero_pairs = []
        for pair in all_pairs:
            if pair_count.get(pair, 0) == 0:
                p1, p2 = pair
                zero_pairs.append((p1+1, p2+1))
                print(f"  P{p1+1} - P{p2+1}")
        
        print(f"\n0回同卓のペア数: {len(zero_pairs)}")
    
    # フォーマット済み結果も確認
    formatted = generator.format_results(results)
    print(f"\nフォーマット済み統計:")
    print(f"  最小: {formatted['statistics']['pair_statistics']['min_count']}回")
    print(f"  最大: {formatted['statistics']['pair_statistics']['max_count']}回")
    print(f"  カバレッジ: {formatted['statistics']['pair_statistics']['coverage']:.1f}%")
    
    return min_count > 0


if __name__ == "__main__":
    has_full_coverage = test_web_service()
    
    if not has_full_coverage:
        print("\n" + "="*60)
        print("問題: 0回同卓のペアが存在します！")
        print("12人6回戦の特別処理が正しく動作していない可能性があります。")