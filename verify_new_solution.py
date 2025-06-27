#!/usr/bin/env python3
"""新しい12人6回戦解の検証"""

from itertools import combinations
from collections import defaultdict

def verify_solution():
    """新しい解を検証"""
    # 更新された_get_12_6_optimal_solutionの解
    solution = [
        # 第1回戦
        [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11]],
        # 第2回戦  
        [[0, 4, 8, 11], [1, 5, 9, 6], [2, 3, 7, 10]],
        # 第3回戦
        [[0, 5, 10, 6], [1, 3, 8, 11], [2, 4, 9, 7]],
        # 第4回戦
        [[0, 3, 9, 7], [1, 4, 10, 8], [2, 5, 11, 6]],
        # 第5回戦
        [[0, 1, 10, 7], [2, 5, 8, 9], [3, 4, 6, 11]],
        # 第6回戦
        [[0, 2, 4, 9], [1, 3, 5, 10], [6, 7, 8, 11]],
    ]
    
    # ペアカウントを計算
    pair_count = defaultdict(int)
    
    for round_num, round_tables in enumerate(solution, 1):
        print(f"\n第{round_num}回戦:")
        for table_num, table in enumerate(round_tables, 1):
            print(f"  卓{table_num}: {sorted(table)}")
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
    
    # 各プレイヤーの待機回数を確認（全員0回のはず）
    print("\n各プレイヤーの参加状況:")
    for player in range(12):
        rounds_played = sum(1 for round_tables in solution for table in round_tables if player in table)
        print(f"  プレイヤー{player}: {rounds_played}回参加")

if __name__ == "__main__":
    verify_solution()