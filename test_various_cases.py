#!/usr/bin/env python3
"""様々なケースで同様の問題が発生しないか確認"""

from table_group_web_universal import TableGroupGenerator
from collections import defaultdict


def test_case(players, rounds, name):
    """特定のケースをテスト"""
    print(f"\n{'='*60}")
    print(f"ケース: {name}")
    print(f"参加者: {players}人、回数: {rounds}回")
    
    player_names = [f"P{i}" for i in range(1, players + 1)]
    generator = TableGroupGenerator(player_names, rounds, allow_five=False)
    
    try:
        results = generator.generate()
        formatted = generator.format_results(results)
        
        # 各ラウンドをチェック
        has_issue = False
        for round_data in formatted['rounds_data']:
            round_num = round_data['round_num']
            tables = round_data['tables']
            waiting = round_data['waiting']
            
            # 配置された人数を計算
            total_seated = sum(len(table['players']) for table in tables)
            total_waiting = len(waiting)
            total_players = total_seated + total_waiting
            
            if total_players != players:
                print(f"\n❌ 第{round_num}回戦: エラー！")
                print(f"   配置: {total_seated}人、待機: {total_waiting}人")
                print(f"   合計: {total_players}人（{players}人であるべき）")
                has_issue = True
            
            # 空の卓をチェック
            for table in tables:
                if len(table['players']) == 0:
                    print(f"   ⚠️ 卓{table['table_num']}が空！")
                    has_issue = True
                elif len(table['players']) < 4 and not generator.allow_five:
                    print(f"   ⚠️ 卓{table['table_num']}が{len(table['players'])}人のみ！")
                    has_issue = True
        
        if not has_issue:
            # 統計情報
            stats = formatted['statistics']['pair_statistics']
            print(f"✓ 問題なし - 最小: {stats['min_count']}回、最大: {stats['max_count']}回、カバレッジ: {stats['coverage']:.1f}%")
        
    except Exception as e:
        print(f"❌ エラー発生: {e}")


def main():
    """メイン実行"""
    print("様々なケースでの問題チェック")
    print("="*80)
    
    # テストケース
    test_cases = [
        # 4の倍数のケース
        (8, 4, "8人4回戦"),
        (8, 8, "8人8回戦"),
        (12, 4, "12人4回戦"),
        (12, 6, "12人6回戦（既知の問題）"),
        (12, 8, "12人8回戦"),
        (16, 4, "16人4回戦"),
        (16, 6, "16人6回戦"),
        (16, 8, "16人8回戦"),
        (20, 4, "20人4回戦"),
        (20, 6, "20人6回戦"),
        (24, 4, "24人4回戦"),
        (24, 6, "24人6回戦"),
        
        # 4の倍数でないケース（待機者あり）
        (9, 4, "9人4回戦（1人待機）"),
        (10, 4, "10人4回戦（2人待機）"),
        (11, 4, "11人4回戦（3人待機）"),
        (13, 4, "13人4回戦（1人待機）"),
        (14, 4, "14人4回戦（2人待機）"),
        (15, 4, "15人4回戦（3人待機）"),
        (17, 4, "17人4回戦（1人待機）"),
        (18, 4, "18人4回戦（2人待機）"),
        (19, 4, "19人4回戦（3人待機）"),
        
        # 境界ケース
        (4, 3, "4人3回戦（最小）"),
        (40, 8, "40人8回戦（大規模）"),
    ]
    
    # 問題のあるケースをカウント
    issue_count = 0
    
    for players, rounds, name in test_cases:
        try:
            test_case(players, rounds, name)
        except Exception as e:
            print(f"\n❌ {name}: 実行エラー - {e}")
            issue_count += 1
    
    print(f"\n{'='*80}")
    print(f"テスト完了: {len(test_cases)}ケース中{issue_count}ケースで問題あり")


if __name__ == "__main__":
    main()