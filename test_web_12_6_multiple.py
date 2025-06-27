#!/usr/bin/env python3
"""Webサービスの12人6回戦を複数回テスト"""

from table_group_web_universal import TableGroupGenerator
from collections import defaultdict
from itertools import combinations


def test_multiple_runs():
    """複数回実行して結果の一貫性を確認"""
    
    player_names = [f"Player{i}" for i in range(1, 13)]
    
    print("Webサービスの12人6回戦を10回テスト")
    print("="*60)
    
    results_summary = []
    
    for run in range(10):
        generator = TableGroupGenerator(player_names, 6, allow_five=False)
        results = generator.generate()
        
        # ペアカウント
        pair_count = defaultdict(int)
        
        for round_tables in results:
            for table in round_tables:
                for p1, p2 in combinations(table, 2):
                    pair = tuple(sorted([p1, p2]))
                    pair_count[pair] += 1
        
        # 統計
        all_pairs = list(combinations(range(12), 2))
        all_counts = [pair_count.get(pair, 0) for pair in all_pairs]
        
        min_count = min(all_counts)
        max_count = max(all_counts)
        coverage = sum(1 for c in all_counts if c > 0) / len(all_pairs) * 100
        
        count_distribution = defaultdict(int)
        for count in all_counts:
            count_distribution[count] += 1
        
        results_summary.append({
            'run': run + 1,
            'min': min_count,
            'max': max_count,
            'coverage': coverage,
            'distribution': dict(count_distribution),
            'has_zero': min_count == 0
        })
        
        print(f"\n実行 {run + 1}:")
        print(f"  最小: {min_count}回, 最大: {max_count}回")
        print(f"  カバレッジ: {coverage:.1f}%")
        print(f"  分布: {dict(count_distribution)}")
        
        if min_count == 0:
            print("  ⚠️ 0回同卓のペアあり！")
            # 0回同卓のペアを表示
            zero_pairs = []
            for pair in all_pairs:
                if pair_count.get(pair, 0) == 0:
                    p1, p2 = pair
                    zero_pairs.append((p1+1, p2+1))
            print(f"  0回同卓のペア: {zero_pairs}")
    
    # サマリー
    print("\n" + "="*60)
    print("サマリー:")
    
    zero_count_runs = sum(1 for r in results_summary if r['has_zero'])
    print(f"  0回同卓のペアが発生した実行回数: {zero_count_runs}/10")
    
    if zero_count_runs > 0:
        print("\n問題: 12人6回戦の特別処理が一貫して動作していません！")
    else:
        print("\n✓ 全ての実行で100%カバレッジを達成")
    
    # 結果が同じかチェック
    first_dist = results_summary[0]['distribution']
    all_same = all(r['distribution'] == first_dist for r in results_summary)
    
    if all_same:
        print("✓ 全ての実行で同じ結果（決定的アルゴリズム）")
    else:
        print("⚠️ 実行毎に異なる結果（非決定的アルゴリズム）")


if __name__ == "__main__":
    test_multiple_runs()