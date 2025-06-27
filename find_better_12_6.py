#!/usr/bin/env python3
"""12人6回戦のより良い解を探索"""

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
    
    # 階層的スコア
    score = (
        min_count * 1000000 +
        -max_count * 10000 +
        -count_distribution[min_count] * 100 +
        -count_distribution[max_count]
    )
    
    return {
        'min': min_count,
        'max': max_count,
        'coverage': coverage,
        'distribution': dict(count_distribution),
        'score': score
    }


def search_better_solution():
    """現在の解より良い解を探索"""
    
    # 現在の解
    current_solution = [
        [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11]],
        [[0, 4, 8, 11], [1, 5, 9, 6], [2, 3, 7, 10]],
        [[0, 5, 10, 6], [1, 3, 8, 11], [2, 4, 9, 7]],
        [[0, 3, 9, 7], [1, 4, 10, 8], [2, 5, 11, 6]],
        [[0, 1, 10, 7], [2, 5, 8, 9], [3, 4, 6, 11]],
        [[0, 2, 4, 9], [1, 3, 5, 10], [6, 7, 8, 11]],
    ]
    
    current_stats = evaluate_solution(current_solution)
    print("現在の解:")
    print(f"  カバレッジ: {current_stats['coverage']:.1f}%")
    print(f"  最小: {current_stats['min']}回, 最大: {current_stats['max']}回")
    print(f"  分布: {current_stats['distribution']}")
    print(f"  スコア: {current_stats['score']}")
    
    # より良い解を探す（第1回戦は固定）
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
    
    print(f"\n可能なラウンド構成数: {len(valid_rounds)}")
    print("より良い解を探索中...")
    
    best_solution = current_solution
    best_stats = current_stats
    
    # 第1回戦を固定
    first_round = [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11]]
    remaining_rounds = [r for r in valid_rounds if r != first_round]
    
    for attempt in range(50000):
        if attempt % 5000 == 0:
            print(f"  試行 {attempt}...")
        
        # ランダムに5ラウンドを選択
        solution = [first_round]
        solution.extend(random.sample(remaining_rounds, 5))
        
        stats = evaluate_solution(solution)
        
        # より良い解かチェック
        if stats['coverage'] == 100 and stats['score'] > best_stats['score']:
            best_solution = solution
            best_stats = stats
            print(f"\nより良い解を発見！（試行 {attempt + 1}）")
            print(f"  カバレッジ: {stats['coverage']:.1f}%")
            print(f"  最小: {stats['min']}回, 最大: {stats['max']}回")
            print(f"  分布: {stats['distribution']}")
            print(f"  スコア: {stats['score']}")
            
            if stats['max'] <= 3:
                print("  ✓ 最大3回同卓を達成！")
                break
    
    return best_solution, best_stats


def main():
    """メイン実行"""
    print("12人6回戦のより良い解を探索")
    print("="*60)
    
    random.seed(42)
    best_solution, best_stats = search_better_solution()
    
    if best_stats['score'] > 0:
        print("\n" + "="*60)
        print("最良解:")
        
        for round_num, round_tables in enumerate(best_solution, 1):
            print(f"\n第{round_num}回戦:")
            for table_num, table in enumerate(round_tables, 1):
                print(f"  卓{table_num}: {table}")
        
        print(f"\n統計:")
        print(f"  カバレッジ: {best_stats['coverage']:.1f}%")
        print(f"  最小: {best_stats['min']}回, 最大: {best_stats['max']}回")
        print(f"  分布: {best_stats['distribution']}")
        
        if best_stats['coverage'] == 100 and best_stats['max'] <= 3:
            print("\n# Python形式の解（0-indexed）:")
            print("BETTER_12_6_SOLUTION = [")
            for round_tables in best_solution:
                print(f"    {round_tables},")
            print("]")


if __name__ == "__main__":
    main()