#!/usr/bin/env python3
"""12人6回戦の事前計算解の検証スクリプト"""

from itertools import combinations
from collections import defaultdict

def verify_solution():
    """12人6回戦の解を検証"""
    # _get_12_6_optimal_solutionから取得した解
    solution = [
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
        [[0, 2, 7, 11], [1, 6, 9, 10], [3, 4, 5, 8]],
    ]
    
    # ペアカウントを計算
    pair_count = defaultdict(int)
    
    for round_num, round_tables in enumerate(solution, 1):
        print(f"\n第{round_num}回戦:")
        for table_num, table in enumerate(round_tables, 1):
            print(f"  卓{table_num}: {table}")
            # この卓でのペアを記録
            for p1, p2 in combinations(table, 2):
                pair = tuple(sorted([p1, p2]))
                pair_count[pair] += 1
    
    # 全ペアをチェック
    print("\n=== ペア同卓回数の検証 ===")
    total_pairs = 12 * 11 // 2  # 66ペア
    
    # 0回同卓のペアを見つける
    all_pairs = set()
    for i in range(12):
        for j in range(i + 1, 12):
            all_pairs.add((i, j))
    
    met_pairs = set(pair_count.keys())
    never_met_pairs = all_pairs - met_pairs
    
    print(f"\n全ペア数: {total_pairs}")
    print(f"少なくとも1回同卓したペア数: {len(met_pairs)}")
    print(f"0回同卓のペア数: {len(never_met_pairs)}")
    print(f"カバレッジ: {len(met_pairs) / total_pairs * 100:.1f}%")
    
    if never_met_pairs:
        print("\n0回同卓のペア:")
        for p1, p2 in sorted(never_met_pairs):
            print(f"  ({p1}, {p2})")
    
    # 同卓回数の分布
    count_distribution = defaultdict(int)
    for count in pair_count.values():
        count_distribution[count] += 1
    
    if never_met_pairs:
        count_distribution[0] = len(never_met_pairs)
    
    print("\n同卓回数の分布:")
    for count in sorted(count_distribution.keys()):
        print(f"  {count}回: {count_distribution[count]}ペア")
    
    # 最小・最大同卓回数
    all_counts = list(pair_count.values())
    if never_met_pairs:
        all_counts.extend([0] * len(never_met_pairs))
    
    print(f"\n最小同卓回数: {min(all_counts)}")
    print(f"最大同卓回数: {max(all_counts)}")

if __name__ == "__main__":
    verify_solution()