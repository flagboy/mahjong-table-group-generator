#!/usr/bin/env python3
"""問題の詳細なトレース"""

from table_group_universal import UniversalTableGroupGenerator


def trace_issue():
    """問題を詳細にトレース"""
    
    print("12人6回戦の問題をトレース")
    print("="*60)
    
    # UniversalTableGroupGeneratorのインスタンスを作成
    gen = UniversalTableGroupGenerator(players=12, rounds=6, allow_five=False)
    
    # _optimize_exact メソッドを確認
    print(f"\n戦略: {gen.strategy}")
    print(f"プレイヤー数: {gen.players}")
    print(f"ラウンド数: {gen.rounds}")
    
    # 各メソッドが何人用に卓を作るか確認
    if gen.players % 4 == 0:
        tables_per_round = gen.players // 4
        print(f"\n期待される卓数/ラウンド: {tables_per_round}")
    
    # generate()を呼び出して詳細を確認
    results = gen.generate()
    
    print("\n生成された結果:")
    for round_num, round_tables in enumerate(results, 1):
        total_players = sum(len(table) for table in round_tables)
        print(f"\n第{round_num}回戦:")
        print(f"  卓数: {len(round_tables)}")
        print(f"  配置人数: {total_players}")
        print(f"  各卓: {[len(t) for t in round_tables]}人")
        
        if total_players != 12:
            print(f"  ❌ エラー: {12 - total_players}人が未配置！")
            # 未配置のプレイヤーを特定
            all_players = set(range(1, 13))
            placed_players = set()
            for table in round_tables:
                placed_players.update(table)
            missing = all_players - placed_players
            print(f"  未配置プレイヤー: {sorted(missing)}")


if __name__ == "__main__":
    trace_issue()