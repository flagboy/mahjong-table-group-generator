#!/usr/bin/env python3
"""12人6回戦の最適解を生成するプログラム"""

from itertools import combinations
from collections import defaultdict
import random


def find_12_6_optimal():
    """12人6回戦の最適解を探索"""
    
    # 全ての可能な卓の組み合わせ
    all_tables = list(combinations(range(1, 13), 4))
    print(f"可能な卓数: {len(all_tables)}")
    
    # 各ラウンドで可能な卓の組み合わせ（3卓で12人全員をカバー）
    valid_rounds = []
    for t1 in all_tables:
        remaining1 = set(range(1, 13)) - set(t1)
        for t2 in combinations(remaining1, 4):
            remaining2 = remaining1 - set(t2)
            if len(remaining2) == 4:
                t3 = tuple(sorted(remaining2))
                round_config = tuple(sorted([t1, t2, t3]))
                if round_config not in valid_rounds:
                    valid_rounds.append(round_config)
    
    print(f"可能なラウンド構成数: {len(valid_rounds)}")
    
    # 最適解を探索（ランダムサンプリング）
    best_solution = None
    best_distribution = None
    target_distribution = {1: 24, 2: 42}
    
    print("\n最適解を探索中...")
    
    for attempt in range(100000):
        if attempt % 10000 == 0:
            print(f"試行 {attempt}...")
        
        # ランダムに6ラウンドを選択
        selected_rounds = random.sample(valid_rounds, 6)
        
        # ペア回数をカウント
        pair_count = defaultdict(int)
        for round_tables in selected_rounds:
            for table in round_tables:
                for p1, p2 in combinations(table, 2):
                    pair = tuple(sorted([p1, p2]))
                    pair_count[pair] += 1
        
        # 分布を計算
        all_pairs = list(combinations(range(1, 13), 2))
        counts = [pair_count.get(pair, 0) for pair in all_pairs]
        distribution = defaultdict(int)
        for c in counts:
            distribution[c] += 1
        
        # 目標分布と一致するかチェック
        if dict(distribution) == target_distribution:
            print(f"\n最適解を発見！（試行 {attempt + 1}）")
            best_solution = selected_rounds
            best_distribution = dict(distribution)
            break
        
        # より良い解かチェック
        if best_distribution is None or (
            distribution.get(0, 0) < best_distribution.get(0, 1000) or
            (distribution.get(0, 0) == best_distribution.get(0, 0) and
             max(counts) < max(pair_count.get(p, 0) for p in all_pairs))
        ):
            best_solution = selected_rounds
            best_distribution = dict(distribution)
    
    return best_solution, best_distribution


def format_solution(rounds):
    """解を読みやすい形式に変換"""
    formatted = []
    for round_tables in rounds:
        round_data = []
        for table in round_tables:
            round_data.append(list(table))
        formatted.append(round_data)
    return formatted


def verify_and_print(solution):
    """解を検証して表示"""
    print("\n発見した解:")
    print("="*50)
    
    pair_count = defaultdict(int)
    
    for i, round_tables in enumerate(solution, 1):
        print(f"\n第{i}回戦:")
        for j, table in enumerate(round_tables, 1):
            print(f"  卓{j}: {list(table)}")
            for p1, p2 in combinations(table, 2):
                pair = tuple(sorted([p1, p2]))
                pair_count[pair] += 1
    
    # 統計
    all_pairs = list(combinations(range(1, 13), 2))
    counts = [pair_count.get(pair, 0) for pair in all_pairs]
    distribution = defaultdict(int)
    for c in counts:
        distribution[c] += 1
    
    print(f"\n統計:")
    print(f"  最小: {min(counts)}回, 最大: {max(counts)}回")
    print(f"  分布: {dict(distribution)}")
    print(f"  目標: {{1: 24, 2: 42}}")
    
    if dict(distribution) == {1: 24, 2: 42}:
        print("  ✓ 完璧な最適解です！")
        
        # Python形式で出力
        print("\n# Python形式の解:")
        print("solution = [")
        for round_tables in solution:
            tables = [list(table) for table in round_tables]
            print(f"    {tables},")
        print("]")
    else:
        print("  ✗ 最適解ではありません")


def main():
    """メイン実行"""
    print("12人6回戦の最適解探索プログラム")
    print("="*50)
    
    solution, distribution = find_12_6_optimal()
    
    if solution:
        verify_and_print(solution)
    else:
        print("最適解が見つかりませんでした")


if __name__ == "__main__":
    # 乱数シードを設定（再現性のため）
    random.seed(42)
    main()