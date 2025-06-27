#!/usr/bin/env python3
"""12人6回戦の最適解を検証"""

from collections import defaultdict
from itertools import combinations


def verify_solution():
    """現在の12人6回戦の解を検証"""
    
    # table_group_web_universal.pyに実装された解（0-indexed）
    solution = [
        # 第1回戦
        [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11]],
        # 第2回戦
        [[0, 4, 8, 10], [1, 5, 9, 11], [2, 3, 6, 7]],
        # 第3回戦
        [[0, 5, 7, 9], [1, 3, 6, 10], [2, 4, 8, 11]],
        # 第4回戦
        [[0, 3, 6, 11], [1, 4, 7, 8], [2, 5, 9, 10]],
        # 第5回戦
        [[0, 1, 5, 8], [2, 6, 10, 11], [3, 4, 7, 9]],
        # 第6回戦
        [[0, 2, 7, 10], [1, 4, 6, 9], [3, 5, 8, 11]],
    ]
    
    print("12人6回戦の最適解検証")
    print("="*60)
    
    # ペアカウント
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
    
    print(f"\n統計結果:")
    print(f"  総ペア数: {len(all_pairs)}")
    print(f"  最小同卓回数: {min_count}回")
    print(f"  最大同卓回数: {max_count}回")
    print(f"  カバレッジ: {coverage:.1f}%")
    print(f"  分布: {dict(count_distribution)}")
    
    # 理論的最適解かチェック
    is_optimal = (dict(count_distribution) == {1: 24, 2: 42})
    
    print(f"\n評価:")
    if is_optimal:
        print("  ✓ 理論的最適解です！（1回×24ペア、2回×42ペア）")
        print("  ✓ 最大2回同卓を達成（理論的最適値）")
    else:
        print("  ✗ 理論的最適解ではありません")
        if min_count < 1:
            print(f"    - {count_distribution[0]}ペアが0回同卓")
        if max_count > 2:
            print(f"    - {count_distribution[max_count]}ペアが{max_count}回同卓")
    
    # 各プレイヤーの同卓パターンも確認
    if is_optimal:
        print("\n各プレイヤーの同卓パターン:")
        player_meetings = defaultdict(lambda: defaultdict(int))
        for (p1, p2), count in pair_count.items():
            player_meetings[p1][p2] = count
            player_meetings[p2][p1] = count
        
        for player in range(12):
            meetings = [player_meetings[player][other] for other in range(12) if other != player]
            ones = meetings.count(1)
            twos = meetings.count(2)
            print(f"  P{player+1}: 1回同卓×{ones}人, 2回同卓×{twos}人")
    
    return is_optimal


if __name__ == "__main__":
    is_optimal = verify_solution()
    
    if is_optimal:
        print("\n" + "="*60)
        print("結論: 12人6回戦の理論的最適解が実装されています！")
    else:
        print("\n" + "="*60)
        print("結論: さらなる改善が必要です。")