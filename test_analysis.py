#!/usr/bin/env python3
"""テスト結果の詳細分析"""

from table_group_web_universal import TableGroupGenerator
from collections import defaultdict
from itertools import combinations
import time


def analyze_specific_cases():
    """特定の問題ケースを詳細分析"""
    
    # 問題のあるケース
    problem_cases = [
        (12, 6, "12人6回戦（ユーザー報告）"),
        (32, 4, "32人4回戦（理論値との乖離大）"),
        (36, 6, "36人6回戦（理論値との乖離大）"),
        (15, 12, "15人12回戦（最大回戦）"),
    ]
    
    print("問題ケースの詳細分析")
    print("="*80)
    
    for players, rounds, description in problem_cases:
        print(f"\n{description}")
        print("-"*60)
        
        # 理論値計算
        total_pairs = players * (players - 1) // 2
        tables_per_round = players // 4
        pairs_per_round = tables_per_round * 6
        total_meetings = pairs_per_round * rounds
        ideal_avg = total_meetings / total_pairs
        ideal_min = int(ideal_avg)
        ideal_max = ideal_min + (1 if ideal_avg % 1 > 0 else 0)
        
        print(f"理論値:")
        print(f"  全ペア数: {total_pairs}")
        print(f"  1ラウンドのペア数: {pairs_per_round}")
        print(f"  総ペア実現数: {total_meetings}")
        print(f"  理想平均: {ideal_avg:.2f}回")
        print(f"  理想的な分布: {ideal_min}回と{ideal_max}回のみ")
        
        # 5回実行して最良・最悪・平均を分析
        results = []
        for i in range(5):
            player_names = [f'P{j+1}' for j in range(players)]
            gen = TableGroupGenerator(player_names, rounds, False)
            game_results = gen.generate()
            formatted = gen.format_results(game_results)
            stats = formatted['statistics']['pair_statistics']
            results.append(stats)
        
        # 結果分析
        print(f"\n実行結果（5回）:")
        min_counts = [r['min_count'] for r in results]
        max_counts = [r['max_count'] for r in results]
        coverages = [r['coverage'] for r in results]
        
        print(f"  最小値: {min(min_counts)}〜{max(min_counts)}回 (平均: {sum(min_counts)/len(min_counts):.1f})")
        print(f"  最大値: {min(max_counts)}〜{max(max_counts)}回 (平均: {sum(max_counts)/len(max_counts):.1f})")
        print(f"  カバレッジ: {min(coverages):.1f}%〜{max(coverages):.1f}% (平均: {sum(coverages)/len(coverages):.1f}%)")
        
        # 最良の結果の分布を表示
        best_result = min(results, key=lambda r: (r['max_count'], -r['min_count']))
        print(f"\n最良の分布:")
        for count in sorted(best_result['distribution'].keys()):
            pairs = best_result['distribution'][count]
            percentage = pairs / total_pairs * 100
            print(f"  {count}回: {pairs}ペア ({percentage:.1f}%)")
        
        # 問題の診断
        print(f"\n診断:")
        if best_result['max_count'] > ideal_max + 1:
            print(f"  ⚠️ 最大値が理論値を大幅に超過（{best_result['max_count']}回 vs 理論{ideal_max}回）")
            if players > 20:
                print(f"  → 大規模ケースでは最適化が不十分")
            if rounds > 10:
                print(f"  → 多回戦では累積誤差が発生")
        
        if best_result['coverage'] < 100 and ideal_avg >= 1.0:
            print(f"  ⚠️ カバレッジが不完全（{best_result['coverage']:.1f}%）")
        
        # 改善の余地
        actual_meetings = sum(count * pairs for count, pairs in best_result['distribution'].items())
        efficiency = actual_meetings / total_meetings * 100
        print(f"\n効率性: {efficiency:.1f}% (実現ペア数/理論最大)")


def analyze_by_category():
    """カテゴリ別の傾向分析"""
    print("\n\nカテゴリ別傾向分析")
    print("="*80)
    
    categories = {
        '小規模（8-12人）': [(8, 4), (10, 5), (12, 6)],
        '中規模（16-20人）': [(16, 5), (18, 6), (20, 8)],
        '大規模（24-40人）': [(24, 6), (32, 4), (40, 3)],
        '少回戦（3-4回）': [(12, 3), (20, 4), (32, 4)],
        '標準回戦（5-6回）': [(12, 6), (16, 5), (24, 6)],
        '多回戦（8-12回）': [(8, 10), (12, 12), (16, 8)],
    }
    
    for category, cases in categories.items():
        print(f"\n{category}")
        print("-"*40)
        
        category_results = []
        for players, rounds in cases:
            # 1回だけ実行
            player_names = [f'P{i+1}' for i in range(players)]
            gen = TableGroupGenerator(player_names, rounds, False)
            game_results = gen.generate()
            formatted = gen.format_results(game_results)
            stats = formatted['statistics']['pair_statistics']
            
            # 理論値
            total_pairs = players * (players - 1) // 2
            pairs_per_round = (players // 4) * 6
            ideal_avg = (pairs_per_round * rounds) / total_pairs
            ideal_max = int(ideal_avg) + (1 if ideal_avg % 1 > 0 else 0)
            
            category_results.append({
                'players': players,
                'rounds': rounds,
                'actual_max': stats['max_count'],
                'ideal_max': ideal_max,
                'coverage': stats['coverage'],
                'excess': stats['max_count'] - ideal_max
            })
        
        # 統計
        avg_excess = sum(r['excess'] for r in category_results) / len(category_results)
        avg_coverage = sum(r['coverage'] for r in category_results) / len(category_results)
        
        print(f"  平均超過回数: {avg_excess:.1f}回")
        print(f"  平均カバレッジ: {avg_coverage:.1f}%")
        
        for r in category_results:
            status = "✓" if r['excess'] <= 0 else f"△ +{r['excess']}"
            print(f"  {r['players']}人{r['rounds']}回: 最大{r['actual_max']}回 (理論{r['ideal_max']}回) {status}")


def suggest_improvements():
    """改善提案"""
    print("\n\n改善提案")
    print("="*80)
    
    print("\n1. 大規模ケース（24人以上）の改善:")
    print("   - 現状: 理論値を2-3回超過することがある")
    print("   - 原因: 探索空間が大きすぎて局所最適に陥る")
    print("   - 提案: より洗練されたメタヒューリスティクスの採用")
    
    print("\n2. 多回戦ケース（8回戦以上）の改善:")
    print("   - 現状: 累積誤差により最大値が増加")
    print("   - 原因: 各ラウンドの最適化が独立的")
    print("   - 提案: 全ラウンドを考慮した大域的最適化")
    
    print("\n3. 12人6回戦の特別対応:")
    print("   - 現状: 最大3回（理論値は2回）")
    print("   - 提案: 事前計算した最適解のデータベース化")
    
    print("\n4. 5人打ち対応の改善:")
    print("   - 現状: 基本的な対応のみ")
    print("   - 提案: 5人卓の配置最適化アルゴリズム")


def main():
    """メイン実行"""
    print("麻雀卓組生成システム - 詳細分析レポート")
    print("="*80)
    print()
    
    # 1. 問題ケースの詳細分析
    analyze_specific_cases()
    
    # 2. カテゴリ別傾向分析
    analyze_by_category()
    
    # 3. 改善提案
    suggest_improvements()
    
    print("\n\n結論:")
    print("="*80)
    print("現在の実装は多くのケースで良好な結果を出していますが、")
    print("以下のケースで改善の余地があります：")
    print("- 大規模（24人以上）")
    print("- 多回戦（8回戦以上）")
    print("- 12人6回戦のような特定の組み合わせ")
    print("\n実用上は十分な品質ですが、理論的最適解への")
    print("さらなる接近は可能です。")


if __name__ == "__main__":
    main()