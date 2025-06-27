#!/usr/bin/env python3
"""12人6回戦の特別扱いなしでテスト"""

from table_group_web_universal import TableGroupGenerator
from collections import defaultdict
from itertools import combinations


def test_without_special_case():
    """特別扱いなしで12人6回戦をテスト"""
    
    player_names = [f"P{i}" for i in range(1, 13)]
    generator = TableGroupGenerator(player_names, 6, allow_five=False)
    
    print("12人6回戦テスト（特別扱いなし）")
    print("="*60)
    
    results = generator.generate()
    formatted = generator.format_results(results)
    
    # 統計を表示
    stats = formatted['statistics']['pair_statistics']
    print(f"\n統計:")
    print(f"  最小同卓回数: {stats['min_count']}回")
    print(f"  最大同卓回数: {stats['max_count']}回")
    print(f"  カバレッジ: {stats['coverage']:.1f}%")
    print(f"  分布: {stats['distribution']}")
    
    # UniversalTableGroupGeneratorの動作を確認
    print(f"\nUniversalTableGroupGeneratorを使用")
    print("階層的最適化（条件1-4）が適用されます")


if __name__ == "__main__":
    test_without_special_case()