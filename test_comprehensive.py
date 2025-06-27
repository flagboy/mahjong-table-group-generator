#!/usr/bin/env python3
"""包括的なテストスクリプト"""

from table_group_web_universal import TableGroupGenerator
from collections import defaultdict
from itertools import combinations
import time


def test_case(players: int, rounds: int, allow_five: bool = False, runs: int = 3):
    """特定のケースをテスト"""
    results = []
    
    for run in range(runs):
        start_time = time.time()
        
        # プレイヤー名を生成
        player_names = [f'P{i+1}' for i in range(players)]
        gen = TableGroupGenerator(player_names, rounds, allow_five)
        game_results = gen.generate()
        formatted = gen.format_results(game_results)
        
        # 統計情報を取得
        stats = formatted['statistics']['pair_statistics']
        
        # 待機統計も取得
        wait_stats = formatted['statistics'].get('wait_statistics', None)
        
        elapsed_time = time.time() - start_time
        
        results.append({
            'min_count': stats['min_count'],
            'max_count': stats['max_count'],
            'distribution': stats['distribution'],
            'coverage': stats['coverage'],
            'wait_stats': wait_stats,
            'time': elapsed_time
        })
    
    # 理論値を計算
    total_pairs = players * (players - 1) // 2
    tables_per_round = players // 4
    if allow_five and players % 4 == 1:
        # 5人卓1つの場合
        pairs_per_round = (players // 5) * 10 + ((players % 5) // 4) * 6
    else:
        pairs_per_round = tables_per_round * 6
    total_pair_meetings = pairs_per_round * rounds
    ideal_avg = total_pair_meetings / total_pairs if total_pairs > 0 else 0
    
    return {
        'players': players,
        'rounds': rounds,
        'allow_five': allow_five,
        'results': results,
        'theory': {
            'total_pairs': total_pairs,
            'pairs_per_round': pairs_per_round,
            'total_meetings': total_pair_meetings,
            'ideal_avg': ideal_avg,
            'ideal_min': int(ideal_avg),
            'ideal_max': int(ideal_avg) + (1 if ideal_avg % 1 > 0 else 0)
        }
    }


def print_test_results(test_data):
    """テスト結果を表示"""
    print(f"\n{'='*80}")
    print(f"テストケース: {test_data['players']}人 {test_data['rounds']}回戦", end="")
    if test_data['allow_five']:
        print(" (5人打ちあり)")
    else:
        print()
    print(f"{'='*80}")
    
    # 理論値
    theory = test_data['theory']
    print(f"\n理論的分析:")
    print(f"- 全ペア数: {theory['total_pairs']}")
    print(f"- 1ラウンドの最大ペア数: {theory['pairs_per_round']}")
    print(f"- {test_data['rounds']}ラウンドの最大ペア総数: {theory['total_meetings']}")
    print(f"- 理想的な平均同卓回数: {theory['ideal_avg']:.2f}")
    
    # 各実行結果
    print(f"\n実行結果 ({len(test_data['results'])}回):")
    for i, result in enumerate(test_data['results'], 1):
        print(f"\n実行{i}:")
        print(f"  最小: {result['min_count']}回, 最大: {result['max_count']}回")
        print(f"  カバレッジ: {result['coverage']:.1f}%")
        print(f"  分布: ", end="")
        for count in sorted(result['distribution'].keys()):
            print(f"{count}回×{result['distribution'][count]}ペア ", end="")
        print()
        print(f"  実行時間: {result['time']:.2f}秒")
        
        # 待機統計があれば表示
        if result['wait_stats']:
            wait_counts = defaultdict(int)
            for ws in result['wait_stats']:
                wait_counts[ws['count']] += 1
            if wait_counts:
                print(f"  待機回数分布: ", end="")
                for count in sorted(wait_counts.keys()):
                    print(f"{count}回待機×{wait_counts[count]}人 ", end="")
                print()
    
    # 平均値を計算
    if len(test_data['results']) > 1:
        avg_min = sum(r['min_count'] for r in test_data['results']) / len(test_data['results'])
        avg_max = sum(r['max_count'] for r in test_data['results']) / len(test_data['results'])
        avg_coverage = sum(r['coverage'] for r in test_data['results']) / len(test_data['results'])
        avg_time = sum(r['time'] for r in test_data['results']) / len(test_data['results'])
        
        print(f"\n平均値:")
        print(f"  最小: {avg_min:.1f}回, 最大: {avg_max:.1f}回")
        print(f"  カバレッジ: {avg_coverage:.1f}%")
        print(f"  実行時間: {avg_time:.2f}秒")
    
    # 評価
    print(f"\n評価:")
    best_result = min(test_data['results'], key=lambda r: (r['max_count'], -r['min_count']))
    if best_result['max_count'] <= theory['ideal_max']:
        print(f"✓ 理論的に最適な最大値を達成")
    else:
        print(f"△ 最大値が理論値を超過 (実際: {best_result['max_count']}回, 理論: {theory['ideal_max']}回)")
    
    if best_result['coverage'] == 100:
        print(f"✓ 完全なペアカバレッジを達成")
    elif best_result['coverage'] >= 90:
        print(f"○ 良好なペアカバレッジ ({best_result['coverage']:.1f}%)")
    else:
        print(f"△ ペアカバレッジが低い ({best_result['coverage']:.1f}%)")


def main():
    """メインテスト実行"""
    print("麻雀卓組生成の包括的テスト")
    print("="*80)
    
    # テストケース定義
    test_cases = [
        # 小規模（8-12人）
        (8, 3, False, "小規模・少回戦"),
        (8, 6, False, "小規模・多回戦"),
        (10, 4, False, "小規模・待機あり"),
        (12, 4, False, "小規模・標準"),
        (12, 6, False, "12人6回戦（問題のケース）"),
        (12, 8, False, "小規模・多回戦"),
        
        # 中規模（13-20人）
        (16, 5, False, "中規模・標準"),
        (16, 8, False, "中規模・多回戦"),
        (18, 6, False, "中規模・待機あり"),
        (20, 5, False, "中規模・大人数"),
        (20, 10, False, "中規模・超多回戦"),
        
        # 大規模（21-40人）
        (24, 6, False, "大規模・標準"),
        (28, 5, False, "大規模・標準"),
        (32, 4, False, "大規模・少回戦"),
        (36, 6, False, "大規模・多人数"),
        (40, 3, False, "大規模・最大人数"),
        
        # 5人打ちありのケース
        (9, 4, True, "9人・5人打ちあり"),
        (13, 5, True, "13人・5人打ちあり"),
        (17, 6, True, "17人・5人打ちあり"),
        (21, 4, True, "21人・5人打ちあり"),
        
        # 特殊なケース
        (5, 6, False, "5人（最小待機）"),
        (7, 5, False, "7人（多待機）"),
        (15, 12, False, "15人12回戦（最大回戦）"),
        (25, 3, True, "25人・5人打ちあり（5卓）"),
    ]
    
    # 各テストケースを実行
    all_results = []
    
    for players, rounds, allow_five, description in test_cases:
        print(f"\n\nテスト: {description}")
        print("-" * 50)
        
        try:
            # テスト実行（エラーが出やすいケースは1回のみ）
            runs = 1 if players > 30 or rounds > 10 else 3
            test_data = test_case(players, rounds, allow_five, runs)
            all_results.append(test_data)
            print_test_results(test_data)
        except Exception as e:
            print(f"エラー: {e}")
    
    # サマリー
    print(f"\n\n{'='*80}")
    print("テストサマリー")
    print(f"{'='*80}")
    print(f"\n総テストケース数: {len(test_cases)}")
    print(f"成功: {len(all_results)}")
    print(f"失敗: {len(test_cases) - len(all_results)}")
    
    # 問題のあるケースを特定
    print(f"\n問題のあるケース:")
    problem_cases = []
    for result in all_results:
        best = min(result['results'], key=lambda r: (r['max_count'], -r['min_count']))
        if best['max_count'] > result['theory']['ideal_max'] + 1:
            problem_cases.append(result)
            print(f"- {result['players']}人{result['rounds']}回戦: 最大{best['max_count']}回 (理論: {result['theory']['ideal_max']}回)")
    
    if not problem_cases:
        print("なし（全てのケースで良好な結果）")
    
    # 実行時間の分析
    print(f"\n実行時間分析:")
    time_by_size = defaultdict(list)
    for result in all_results:
        size = result['players']
        avg_time = sum(r['time'] for r in result['results']) / len(result['results'])
        if size <= 12:
            time_by_size['小規模'].append(avg_time)
        elif size <= 20:
            time_by_size['中規模'].append(avg_time)
        else:
            time_by_size['大規模'].append(avg_time)
    
    for size, times in time_by_size.items():
        if times:
            print(f"- {size}: 平均{sum(times)/len(times):.2f}秒")


if __name__ == "__main__":
    main()