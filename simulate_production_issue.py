#!/usr/bin/env python3
"""本番環境の問題をシミュレート"""

from collections import defaultdict
from itertools import combinations


def simulate_production_solution():
    """本番環境が返している解を再現"""
    
    # 本番環境から取得した解（0-indexed）
    production_solution = [
        # 第1回戦
        [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11]],
        # 第2回戦
        [[0, 4, 8, 9], [1, 5, 10, 11], [2, 3, 6, 7]],
        # 第3回戦
        [[0, 5, 7, 10], [1, 3, 6, 8], [2, 4, 9, 11]],
        # 第4回戦
        [[0, 3, 6, 11], [1, 4, 7, 9], [2, 5, 8, 10]],
        # 第5回戦
        [[0, 1, 5, 9], [2, 6, 8, 11], [3, 4, 7, 10]],
        # 第6回戦
        [[0, 2, 7, 11], [1, 6, 9, 10], [3, 4, 5, 7]],
    ]
    
    print("本番環境の解の検証")
    print("="*60)
    
    pair_count = defaultdict(int)
    
    # 各ラウンドを表示
    for round_num, round_tables in enumerate(production_solution, 1):
        print(f"\n第{round_num}回戦:")
        for table_num, table in enumerate(round_tables, 1):
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
    
    # 問題の原因を分析
    print("\n問題の原因:")
    print("第6回戦の卓3で、プレイヤー7が重複しています！")
    print("正しくは [3, 4, 5, 8] であるべきところが [3, 4, 5, 7] になっています。")
    
    # 修正版
    print("\n修正版:")
    production_solution[5][2] = [3, 4, 5, 8]  # 第6回戦の卓3を修正
    
    # 再計算
    pair_count_fixed = defaultdict(int)
    for round_tables in production_solution:
        for table in round_tables:
            for p1, p2 in combinations(table, 2):
                pair = tuple(sorted([p1, p2]))
                pair_count_fixed[pair] += 1
    
    all_counts_fixed = [pair_count_fixed.get(pair, 0) for pair in all_pairs]
    min_count_fixed = min(all_counts_fixed)
    coverage_fixed = sum(1 for c in all_counts_fixed if c > 0) / len(all_pairs) * 100
    
    print(f"  修正後の最小同卓回数: {min_count_fixed}回")
    print(f"  修正後のカバレッジ: {coverage_fixed:.1f}%")


if __name__ == "__main__":
    simulate_production_solution()