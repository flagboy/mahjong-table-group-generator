#!/usr/bin/env python3
"""12人6回戦の各解を比較"""

from itertools import combinations
from collections import defaultdict

def verify_solution(solution, name):
    """解を検証して統計を返す"""
    print(f"\n{'='*60}")
    print(f"{name}")
    print('='*60)
    
    # 0-indexedか1-indexedかを自動判定
    min_player = min(min(table) for round_tables in solution for table in round_tables)
    is_one_indexed = min_player == 1
    
    pair_count = defaultdict(int)
    
    for round_num, round_tables in enumerate(solution, 1):
        print(f"\n第{round_num}回戦:")
        for table_num, table in enumerate(round_tables, 1):
            print(f"  卓{table_num}: {table}")
            for p1, p2 in combinations(table, 2):
                pair = tuple(sorted([p1, p2]))
                pair_count[pair] += 1
    
    # 全ペアをチェック
    if is_one_indexed:
        all_pairs = set(combinations(range(1, 13), 2))
    else:
        all_pairs = set(combinations(range(12), 2))
    
    met_pairs = set(pair_count.keys())
    never_met_pairs = all_pairs - met_pairs
    
    # 統計
    total_pairs = 66
    coverage = len(met_pairs) / total_pairs * 100
    
    print(f"\n統計:")
    print(f"  全ペア数: {total_pairs}")
    print(f"  同卓したペア数: {len(met_pairs)}")
    print(f"  0回同卓のペア数: {len(never_met_pairs)}")
    print(f"  カバレッジ: {coverage:.1f}%")
    
    # 同卓回数の分布
    count_distribution = defaultdict(int)
    for count in pair_count.values():
        count_distribution[count] += 1
    
    if never_met_pairs:
        count_distribution[0] = len(never_met_pairs)
    
    print(f"\n同卓回数の分布:")
    for count in sorted(count_distribution.keys()):
        print(f"  {count}回: {count_distribution[count]}ペア")
    
    if never_met_pairs and len(never_met_pairs) <= 10:
        print(f"\n0回同卓のペア:")
        for p1, p2 in sorted(never_met_pairs):
            print(f"  ({p1}, {p2})")
    
    return {
        'coverage': coverage,
        'never_met': len(never_met_pairs),
        'distribution': dict(count_distribution)
    }

def main():
    # 現在のtable_group_web_universal.pyの解（0-indexed）
    current_solution = [
        [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11]],
        [[0, 4, 8, 9], [1, 5, 10, 11], [2, 3, 6, 7]],
        [[0, 5, 7, 10], [1, 3, 6, 8], [2, 4, 9, 11]],
        [[0, 3, 6, 11], [1, 4, 7, 9], [2, 5, 8, 10]],
        [[0, 1, 5, 9], [2, 6, 8, 11], [3, 4, 7, 10]],
        [[0, 2, 7, 11], [1, 6, 9, 10], [3, 4, 5, 8]],
    ]
    
    # table_group_12_6_proven.pyの最適解1（1-indexed）
    proven_solution_1 = [
        [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]],
        [[1, 5, 9, 12], [2, 6, 10, 7], [3, 4, 8, 11]],
        [[1, 6, 11, 7], [2, 4, 9, 12], [3, 5, 10, 8]],
        [[1, 4, 10, 8], [2, 5, 11, 9], [3, 6, 12, 7]],
        [[1, 2, 11, 8], [3, 6, 9, 10], [4, 5, 7, 12]],
        [[1, 3, 5, 10], [2, 4, 6, 11], [7, 8, 9, 12]],
    ]
    
    # table_group_12_6_proven.pyの最適解2（1-indexed）
    proven_solution_2 = [
        [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]],
        [[1, 5, 9, 10], [2, 6, 11, 12], [3, 4, 7, 8]],
        [[1, 6, 8, 12], [2, 4, 9, 11], [3, 5, 7, 10]],
        [[1, 4, 7, 11], [2, 5, 8, 9], [3, 6, 10, 12]],
        [[1, 2, 7, 9], [3, 4, 5, 12], [6, 8, 10, 11]],
        [[1, 3, 6, 11], [2, 4, 8, 10], [5, 7, 9, 12]],
    ]
    
    # 各解を検証
    results = []
    results.append(('現在の解（table_group_web_universal.py）', 
                   verify_solution(current_solution, "現在の解（table_group_web_universal.py）")))
    results.append(('証明済み最適解1（table_group_12_6_proven.py）', 
                   verify_solution(proven_solution_1, "証明済み最適解1（table_group_12_6_proven.py）")))
    results.append(('証明済み最適解2（table_group_12_6_proven.py）', 
                   verify_solution(proven_solution_2, "証明済み最適解2（table_group_12_6_proven.py）")))
    
    # サマリー
    print(f"\n{'='*60}")
    print("比較サマリー")
    print('='*60)
    for name, stats in results:
        print(f"\n{name}:")
        print(f"  カバレッジ: {stats['coverage']:.1f}%")
        print(f"  0回同卓ペア数: {stats['never_met']}")
        print(f"  分布: {stats['distribution']}")

if __name__ == "__main__":
    main()