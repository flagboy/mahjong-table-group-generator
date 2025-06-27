#!/usr/bin/env python3
"""麻雀卓組生成プログラム（Web版 - Universal実装ラッパー）"""

from typing import List, Dict, Any
from collections import defaultdict
from itertools import combinations
from table_group_universal import UniversalTableGroupGenerator
from table_group_guaranteed import GuaranteedTableGroupGenerator


class TableGroupGenerator:
    def __init__(self, player_names: List[str], rounds: int, allow_five: bool = False):
        """
        Args:
            player_names: 参加者名のリスト
            rounds: 回数
            allow_five: 5人打ちを許可するか
        """
        self.player_names = player_names
        self.players = len(player_names)
        self.rounds = rounds
        self.allow_five = allow_five
        
        # 16人以下の場合はGuaranteedを使用（より確実）
        # それ以上の場合はUniversalを使用
        if self.players <= 16:
            self.generator = GuaranteedTableGroupGenerator(
                players=self.players,
                rounds=self.rounds,
                allow_five=self.allow_five
            )
        else:
            self.generator = UniversalTableGroupGenerator(
                players=self.players,
                rounds=self.rounds,
                allow_five=self.allow_five
            )
        
    def generate(self) -> List[List[List[int]]]:
        """全ラウンドの卓組を生成"""
        # UniversalTableGroupGeneratorは1-indexedのプレイヤーIDを返すので、
        # 0-indexedに変換する必要がある
        results = self.generator.generate()
        
        # 1-indexedから0-indexedに変換
        converted_results = []
        for round_tables in results:
            converted_round = []
            for table in round_tables:
                converted_table = [p - 1 for p in table]
                converted_round.append(converted_table)
            converted_results.append(converted_round)
        
        return converted_results
    
    def format_results(self, results: List[List[List[int]]]) -> Dict[str, Any]:
        """結果を辞書形式にフォーマット"""
        formatted = {
            'players': self.players,
            'rounds': self.rounds,
            'allow_five': self.allow_five,
            'rounds_data': []
        }
        
        for round_num, groups in enumerate(results, 1):
            round_data = {
                'round_num': round_num,
                'tables': [],
                'waiting': []
            }
            
            table_num = 1
            for group in groups:
                if len(group) >= 4:
                    # プレイヤーを元の入力順でソート
                    sorted_group = sorted(group)
                    table_data = {
                        'table_num': table_num,
                        'players': [self.player_names[p] for p in sorted_group],
                        'count': len(group)
                    }
                    round_data['tables'].append(table_data)
                    table_num += 1
                else:
                    # 4人未満は待機（元の入力順でソート）
                    sorted_group = sorted(group)
                    round_data['waiting'] = [self.player_names[p] for p in sorted_group]
            
            formatted['rounds_data'].append(round_data)
        
        # 統計情報を追加
        formatted['statistics'] = self._get_statistics(results)
        
        return formatted
    
    def _get_statistics(self, results: List[List[List[int]]]) -> Dict[str, Any]:
        """統計情報を取得"""
        # ペアカウントを計算
        pair_count = defaultdict(int)
        wait_count = defaultdict(int)
        
        for round_tables in results:
            for table in round_tables:
                if len(table) >= 4:
                    # ペアカウントを更新
                    for p1, p2 in combinations(table, 2):
                        pair = tuple(sorted([p1, p2]))
                        pair_count[pair] += 1
                else:
                    # 待機カウントを更新
                    for player in table:
                        wait_count[player] += 1
        
        stats = {
            'pair_statistics': self._get_pair_statistics(pair_count),
            'wait_statistics': self._get_wait_statistics(wait_count) if wait_count else None
        }
        return stats
    
    def _get_pair_statistics(self, pair_count: Dict[tuple, int]) -> Dict[str, Any]:
        """ペアの統計情報を取得"""
        # 全ペア数を計算
        total_pairs = self.players * (self.players - 1) // 2
        
        # ペアカウントの集計
        count_distribution = defaultdict(int)
        for count in pair_count.values():
            count_distribution[count] += 1
        
        # 0回同卓のペア数を計算
        pairs_with_meetings = len(pair_count)
        pairs_with_zero_meetings = total_pairs - pairs_with_meetings
        if pairs_with_zero_meetings > 0:
            count_distribution[0] = pairs_with_zero_meetings
        
        # 最小・最大同卓回数
        if pair_count:
            all_counts = list(pair_count.values())
            # 0回同卓も含めて考慮
            if pairs_with_zero_meetings > 0:
                all_counts.extend([0] * pairs_with_zero_meetings)
            min_count = min(all_counts)
            max_count = max(all_counts)
        else:
            min_count = 0
            max_count = 0
        
        return {
            'min_count': min_count,
            'max_count': max_count,
            'distribution': dict(count_distribution),
            'coverage': pairs_with_meetings / total_pairs * 100 if total_pairs > 0 else 0
        }
    
    def _get_wait_statistics(self, wait_count: Dict[int, int]) -> List[Dict[str, Any]]:
        """待機回数の統計情報を取得"""
        wait_stats = []
        for i, name in enumerate(self.player_names):
            wait_stats.append({
                'player': name,
                'count': wait_count.get(i, 0)
            })
        return wait_stats