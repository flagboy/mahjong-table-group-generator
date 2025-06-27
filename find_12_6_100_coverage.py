#!/usr/bin/env python3
"""12人6回戦で100%カバレッジの解を探索"""

from itertools import combinations
from collections import defaultdict
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
        'pair_count': dict(pair_count)
    }


def generate_round_configurations():
    """全ての可能なラウンド構成を生成"""
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
    
    return valid_rounds


def find_100_coverage_solution():
    """100%カバレッジの解を探索"""
    valid_rounds = generate_round_configurations()
    print(f"可能なラウンド構成数: {len(valid_rounds)}")
    
    best_solution = None
    best_stats = None
    
    # 既知の良い初期配置から始める
    initial_round = [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11]]
    
    # 初期配置を含むラウンドを絞り込む
    rounds_with_initial = [r for r in valid_rounds if r == initial_round]
    if not rounds_with_initial:
        rounds_with_initial = valid_rounds
    
    print("\n100%カバレッジの解を探索中...")
    
    for attempt in range(1000000):
        if attempt % 10000 == 0:
            print(f"試行 {attempt}...")
            if best_stats:
                print(f"  現在の最良: カバレッジ {best_stats['coverage']:.1f}%, 最大 {best_stats['max']}回")
        
        # ランダムに6ラウンドを選択（最初は固定）
        solution = [initial_round]
        remaining_rounds = [r for r in valid_rounds if r != initial_round]
        solution.extend(random.sample(remaining_rounds, 5))
        
        # 評価
        stats = evaluate_solution(solution)
        
        # 100%カバレッジかチェック
        if stats['coverage'] == 100:
            if best_stats is None or stats['max'] < best_stats['max']:
                best_solution = solution
                best_stats = stats
                print(f"\n100%カバレッジの解を発見！（試行 {attempt + 1}）")
                print(f"  最小: {stats['min']}回, 最大: {stats['max']}回")
                print(f"  分布: {stats['distribution']}")
                
                if stats['max'] <= 3:
                    print("  ✓ 実用的な最良解です！")
                    return solution, stats
        
        # より良い解かチェック
        elif best_stats is None or (
            stats['coverage'] > best_stats['coverage'] or
            (stats['coverage'] == best_stats['coverage'] and stats['max'] < best_stats['max'])
        ):
            best_solution = solution
            best_stats = stats
    
    return best_solution, best_stats


def format_solution(solution):
    """解を表示用にフォーマット"""
    print("\n発見した解:")
    print("="*50)
    
    for i, round_tables in enumerate(solution, 1):
        print(f"\n第{i}回戦:")
        for j, table in enumerate(round_tables, 1):
            # 0-indexed表示
            print(f"  卓{j}: {table}")
    
    stats = evaluate_solution(solution)
    print(f"\n統計:")
    print(f"  カバレッジ: {stats['coverage']:.1f}%")
    print(f"  最小: {stats['min']}回, 最大: {stats['max']}回")
    print(f"  分布: {stats['distribution']}")
    
    if stats['coverage'] == 100 and stats['max'] <= 3:
        print("\n# Python形式の解（0-indexed）:")
        print("solution = [")
        for round_tables in solution:
            print(f"    {round_tables},")
        print("]")
    
    return stats


def main():
    """メイン実行"""
    print("12人6回戦で100%カバレッジの解を探索")
    print("="*60)
    
    # 乱数シードを設定
    random.seed(123)
    
    solution, stats = find_100_coverage_solution()
    
    if solution and stats['coverage'] == 100:
        format_solution(solution)
        
        if stats['max'] <= 3:
            print("\n✓ 100%カバレッジで最大3回同卓の実用的な解が見つかりました！")
        else:
            print(f"\n△ 100%カバレッジですが、最大{stats['max']}回同卓です。")
    else:
        print("\n100%カバレッジの解が見つかりませんでした。")
        if solution:
            format_solution(solution)


if __name__ == "__main__":
    main()