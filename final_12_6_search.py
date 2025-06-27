#!/usr/bin/env python3
"""12人6回戦の最大3回同卓解を集中探索"""

from collections import defaultdict
from itertools import combinations
import random


def evaluate_solution(solution):
    """解を評価"""
    pair_count = defaultdict(int)
    
    for round_tables in solution:
        for table in round_tables:
            for p1, p2 in combinations(table, 2):
                pair = tuple(sorted([p1, p2]))
                pair_count[pair] += 1
    
    all_pairs = list(combinations(range(12), 2))
    all_counts = [pair_count.get(pair, 0) for pair in all_pairs]
    
    min_count = min(all_counts)
    max_count = max(all_counts)
    coverage = sum(1 for c in all_counts if c > 0) / len(all_pairs) * 100
    
    count_distribution = defaultdict(int)
    for count in all_counts:
        count_distribution[count] += 1
    
    return {
        'min': min_count,
        'max': max_count,
        'coverage': coverage,
        'distribution': dict(count_distribution),
        'is_optimal': coverage == 100 and max_count <= 3
    }


def intensive_search():
    """集中的に最大3回同卓の解を探索"""
    
    print("12人6回戦の最大3回同卓解を集中探索")
    print("="*60)
    
    # 候補となる良い初期配置
    initial_rounds = [
        [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11]],
        [[0, 1, 4, 5], [2, 3, 6, 7], [8, 9, 10, 11]],
        [[0, 1, 8, 9], [2, 3, 10, 11], [4, 5, 6, 7]],
    ]
    
    # 全ての可能なラウンド構成
    all_tables = list(combinations(range(12), 4))
    valid_rounds = []
    
    for t1 in all_tables:
        remaining1 = set(range(12)) - set(t1)
        for t2 in combinations(remaining1, 4):
            remaining2 = remaining1 - set(t2)
            if len(remaining2) == 4:
                t3 = tuple(sorted(remaining2))
                round_config = [list(t1), list(t2), list(t3)]
                valid_rounds.append(round_config)
    
    print(f"可能なラウンド構成数: {len(valid_rounds)}")
    
    found_solutions = []
    
    for initial in initial_rounds:
        print(f"\n初期配置: {initial}")
        remaining = [r for r in valid_rounds if r != initial]
        
        for attempt in range(100000):
            if attempt % 10000 == 0 and attempt > 0:
                print(f"  試行 {attempt}...")
            
            # ランダムに5ラウンドを選択
            solution = [initial]
            solution.extend(random.sample(remaining, 5))
            
            stats = evaluate_solution(solution)
            
            if stats['is_optimal']:
                found_solutions.append((solution, stats))
                print(f"\n✓ 最大3回同卓の解を発見！（試行 {attempt + 1}）")
                print(f"  分布: {stats['distribution']}")
                break
    
    return found_solutions


def main():
    """メイン実行"""
    random.seed(123)
    
    solutions = intensive_search()
    
    if solutions:
        print(f"\n{'='*60}")
        print(f"合計 {len(solutions)} 個の最大3回同卓解を発見！")
        
        # 最良の解を選択（1回同卓のペア数が最も少ない）
        best_solution, best_stats = min(solutions, 
            key=lambda x: x[1]['distribution'].get(1, 0))
        
        print(f"\n最良解の統計:")
        print(f"  カバレッジ: {best_stats['coverage']}%")
        print(f"  最大同卓回数: {best_stats['max']}回")
        print(f"  分布: {best_stats['distribution']}")
        
        print("\n# 最大3回同卓の解（0-indexed）:")
        print("OPTIMAL_12_6_MAX3 = [")
        for round_tables in best_solution:
            print(f"    {round_tables},")
        print("]")
    else:
        print(f"\n{'='*60}")
        print("残念ながら最大3回同卓の解は見つかりませんでした。")
        print("現在の実装（最大4回）が実用的な最良解と考えられます。")


if __name__ == "__main__":
    main()