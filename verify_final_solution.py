#!/usr/bin/env python3
"""最終的な12人6回戦解の検証"""

from collections import defaultdict
from itertools import combinations


def verify_final_solution():
    """現在のtable_group_web_universal.pyの解を詳細に検証"""
    
    # 現在の実装（0-indexed）
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
    
    print("最終的な12人6回戦解の検証")
    print("="*60)
    
    # エラーチェック：各ラウンドで12人全員が配置されているか
    for round_num, round_tables in enumerate(solution, 1):
        players_in_round = []
        for table in round_tables:
            players_in_round.extend(table)
        
        if sorted(players_in_round) != list(range(12)):
            print(f"❌ エラー: 第{round_num}回戦で重複または欠落があります！")
            print(f"   配置されたプレイヤー: {sorted(players_in_round)}")
        else:
            print(f"✓ 第{round_num}回戦: 12人全員が正しく配置されています")
    
    # ペアカウント
    pair_count = defaultdict(int)
    
    print("\n各ラウンドの詳細:")
    for round_num, round_tables in enumerate(solution, 1):
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
    
    # 最終評価
    print(f"\n最終評価:")
    if coverage == 100:
        print("✓ 100%カバレッジを達成！")
    else:
        print(f"❌ カバレッジが{coverage:.1f}%（100%未満）")
    
    if max_count <= 3:
        print("✓ 最大3回同卓を達成！")
    elif max_count <= 4:
        print("△ 最大4回同卓（実用上は許容範囲）")
    else:
        print(f"❌ 最大{max_count}回同卓")


if __name__ == "__main__":
    verify_final_solution()