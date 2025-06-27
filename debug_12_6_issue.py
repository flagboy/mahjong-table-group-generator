#!/usr/bin/env python3
"""12人6回戦の問題をデバッグ"""

from table_group_web_universal import TableGroupGenerator
from table_group_universal import UniversalTableGroupGenerator


def debug_issue():
    """問題をデバッグ"""
    
    print("=== 直接UniversalTableGroupGeneratorを使用 ===")
    universal_gen = UniversalTableGroupGenerator(players=12, rounds=6, allow_five=False)
    universal_results = universal_gen.generate()
    
    print("\nUniversalTableGroupGeneratorの結果（1-indexed）:")
    for round_num, round_tables in enumerate(universal_results, 1):
        print(f"\n第{round_num}回戦:")
        all_players = []
        for table_num, table in enumerate(round_tables, 1):
            print(f"  卓{table_num}: {table} ({len(table)}人)")
            all_players.extend(table)
        print(f"  配置された人数: {len(all_players)}人")
        print(f"  プレイヤー: {sorted(all_players)}")
    
    print("\n\n=== TableGroupGeneratorを使用 ===")
    player_names = [f"Player{i}" for i in range(1, 13)]
    web_gen = TableGroupGenerator(player_names, 6, allow_five=False)
    web_results = web_gen.generate()
    
    print("\nTableGroupGeneratorの結果（0-indexed）:")
    for round_num, round_tables in enumerate(web_results, 1):
        print(f"\n第{round_num}回戦:")
        all_players = []
        for table_num, table in enumerate(round_tables, 1):
            print(f"  卓{table_num}: {table} ({len(table)}人)")
            all_players.extend(table)
        print(f"  配置された人数: {len(all_players)}人")
        print(f"  プレイヤー: {sorted(all_players)}")
    
    # format_resultsの確認
    print("\n\n=== format_resultsの結果 ===")
    formatted = web_gen.format_results(web_results)
    
    for round_data in formatted['rounds_data']:
        round_num = round_data['round_num']
        tables = round_data['tables']
        waiting = round_data['waiting']
        
        print(f"\n第{round_num}回戦:")
        total_players = 0
        for table in tables:
            print(f"  卓{table['table_num']}: {table['players']} ({len(table['players'])}人)")
            total_players += len(table['players'])
        
        if waiting:
            print(f"  待機: {waiting} ({len(waiting)}人)")
            total_players += len(waiting)
        
        print(f"  合計: {total_players}人")


if __name__ == "__main__":
    debug_issue()