#!/usr/bin/env python3
"""現在のWebサービスの12人6回戦解を検証"""

from collections import defaultdict
from itertools import combinations


def verify_current_solution():
    """現在実装されている解を検証"""
    
    # table_group_web_universal.pyの現在の解（0-indexed）
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
    
    print("現在のWebサービスの12人6回戦解を検証")
    print("="*60)
    
    pair_count = defaultdict(int)
    
    # 各ラウンドを表示
    for round_num, round_tables in enumerate(solution, 1):
        print(f"\n第{round_num}回戦:")
        for table_num, table in enumerate(round_tables, 1):
            # 1-indexedで表示
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
    
    print(f"\n評価:")
    if coverage == 100:
        print("  ✓ 100%カバレッジ達成")
    else:
        print(f"  ✗ カバレッジが{coverage:.1f}%（100%未満）")
    
    if max_count <= 3:
        print("  ✓ 最大3回同卓で抑制")
    elif max_count <= 4:
        print("  △ 最大4回同卓（実用上は許容範囲）")
    else:
        print(f"  ✗ 最大{max_count}回同卓")
    
    return coverage == 100


if __name__ == "__main__":
    is_100_coverage = verify_current_solution()
    
    if not is_100_coverage:
        print("\n" + "="*60)
        print("警告: 現在の実装は100%カバレッジを達成していません！")
        print("修正が必要です。")