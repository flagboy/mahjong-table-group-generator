#!/usr/bin/env python3
"""12人6回戦で空の卓が発生する問題を調査"""

from table_group_web_universal import TableGroupGenerator
import json


def test_12_6_issue():
    """12人6回戦で空の卓が発生するかテスト"""
    
    player_names = [f"Player{i}" for i in range(1, 13)]
    generator = TableGroupGenerator(player_names, 6, allow_five=False)
    
    print("12人6回戦の問題調査")
    print("="*60)
    print(f"参加者: {len(player_names)}人")
    print(f"回数: 6回")
    print(f"5人打ち: なし")
    
    # 複数回テスト
    for test_num in range(5):
        print(f"\n\nテスト {test_num + 1}:")
        print("-" * 40)
        
        results = generator.generate()
        formatted = generator.format_results(results)
        
        # 各ラウンドをチェック
        has_empty_table = False
        for round_data in formatted['rounds_data']:
            round_num = round_data['round_num']
            tables = round_data['tables']
            
            print(f"\n第{round_num}回戦:")
            
            # 各卓の人数を確認
            total_players_seated = 0
            for table in tables:
                table_num = table['table_num']
                players = table['players']
                player_count = len(players)
                total_players_seated += player_count
                
                print(f"  卓{table_num}: {player_count}人 - {', '.join(players)}")
                
                if player_count == 0:
                    print(f"    ⚠️ 空の卓が発生！")
                    has_empty_table = True
                elif player_count < 4:
                    print(f"    ⚠️ 4人未満の卓！")
            
            # 待機者
            waiting = round_data['waiting']
            if waiting:
                print(f"  待機: {len(waiting)}人 - {', '.join(waiting)}")
            
            # 合計人数チェック
            total_players = total_players_seated + len(waiting)
            if total_players != 12:
                print(f"  ❌ エラー: 合計人数が{total_players}人（12人であるべき）")
            
        # 統計
        stats = formatted['statistics']['pair_statistics']
        print(f"\n統計:")
        print(f"  最小同卓回数: {stats['min_count']}回")
        print(f"  最大同卓回数: {stats['max_count']}回")
        print(f"  カバレッジ: {stats['coverage']:.1f}%")
        
        if has_empty_table:
            print("\n⚠️ このテストで空の卓が発生しました！")
            
            # 生の結果も確認
            print("\n生の結果（0-indexed）:")
            for round_num, round_tables in enumerate(results, 1):
                print(f"第{round_num}回戦: {round_tables}")


if __name__ == "__main__":
    test_12_6_issue()