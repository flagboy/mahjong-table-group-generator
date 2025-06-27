#!/usr/bin/env python3
"""内部実装の修正をテスト"""

from table_group_universal import UniversalTableGroupGenerator


def test_internal_implementation():
    """内部実装を直接テスト"""
    
    print("内部実装（UniversalTableGroupGenerator）の修正テスト")
    print("="*80)
    
    test_cases = [
        (12, 6, "12人6回戦"),
        (9, 4, "9人4回戦（1人待機）"),
        (10, 4, "10人4回戦（2人待機）"),
        (11, 4, "11人4回戦（3人待機）"),
        (16, 6, "16人6回戦"),
        (13, 4, "13人4回戦（1人待機）"),
    ]
    
    for players, rounds, name in test_cases:
        print(f"\n{'='*60}")
        print(f"ケース: {name}")
        
        generator = UniversalTableGroupGenerator(
            players=players,
            rounds=rounds,
            allow_five=False
        )
        
        results = generator.generate()
        
        # 各ラウンドをチェック
        has_issue = False
        for round_num, round_tables in enumerate(results, 1):
            total_players = 0
            for table in round_tables:
                total_players += len(table)
            
            if total_players != players:
                print(f"❌ 第{round_num}回戦: {total_players}人配置（{players}人であるべき）")
                has_issue = True
                
                # 詳細を表示
                print(f"   卓構成: {[len(t) for t in round_tables]}")
                all_players = set(range(1, players + 1))
                placed = set()
                for table in round_tables:
                    placed.update(table)
                missing = all_players - placed
                if missing:
                    print(f"   未配置: {sorted(missing)}")
        
        if not has_issue:
            # 統計を計算
            from collections import defaultdict
            from itertools import combinations
            
            pair_count = defaultdict(int)
            for round_tables in results:
                for table in round_tables:
                    if len(table) >= 4:
                        for p1, p2 in combinations(table, 2):
                            pair = tuple(sorted([p1, p2]))
                            pair_count[pair] += 1
            
            all_pairs = list(combinations(range(1, players + 1), 2))
            all_counts = [pair_count.get(pair, 0) for pair in all_pairs]
            
            min_count = min(all_counts)
            max_count = max(all_counts)
            coverage = sum(1 for c in all_counts if c > 0) / len(all_pairs) * 100
            
            print(f"✓ 正常動作 - 最小{min_count}回、最大{max_count}回、カバレッジ{coverage:.1f}%")


if __name__ == "__main__":
    test_internal_implementation()