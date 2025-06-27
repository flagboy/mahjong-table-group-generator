#!/usr/bin/env python3
"""修正後の動作確認サマリー"""

from table_group_web_universal import TableGroupGenerator


def quick_test(players, rounds):
    """クイックテスト"""
    try:
        generator = TableGroupGenerator([f"P{i}" for i in range(1, players + 1)], rounds, False)
        results = generator.generate()
        formatted = generator.format_results(results)
        
        # 全ラウンドで人数チェック
        for round_data in formatted['rounds_data']:
            total = sum(len(t['players']) for t in round_data['tables']) + len(round_data['waiting'])
            if total != players:
                return f"❌ 人数不一致（{total}/{players}）"
        
        stats = formatted['statistics']['pair_statistics']
        return f"✓ OK（最小{stats['min_count']}回、最大{stats['max_count']}回、{stats['coverage']:.0f}%）"
    except Exception as e:
        return f"❌ エラー: {str(e)[:30]}..."


def main():
    """メイン実行"""
    print("修正後の動作確認サマリー")
    print("="*80)
    
    test_cases = [
        # 主要なケース
        (8, 4), (8, 6), (8, 8),
        (9, 4), (10, 4), (11, 4),
        (12, 4), (12, 6), (12, 8),
        (13, 4), (14, 4), (15, 4),
        (16, 4), (16, 6), (16, 8),
        (17, 4), (18, 4), (19, 4),
        (20, 4), (20, 6),
        (24, 4), (24, 6),
        (28, 4), (32, 4), (36, 4), (40, 4),
    ]
    
    print(f"{'人数':>4} {'回数':>4} {'結果':<50}")
    print("-" * 60)
    
    issue_count = 0
    for players, rounds in test_cases:
        result = quick_test(players, rounds)
        print(f"{players:4d} {rounds:4d} {result}")
        if "❌" in result:
            issue_count += 1
    
    print("\n" + "="*80)
    print(f"テスト結果: {len(test_cases)}ケース中{issue_count}ケースで問題")
    
    if issue_count == 0:
        print("✓ 全てのケースで正常動作を確認！")


if __name__ == "__main__":
    main()