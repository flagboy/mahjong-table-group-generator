#!/usr/bin/env python3
"""麻雀卓組生成プログラム（Web版）"""

import random
from typing import List, Dict, Any
from collections import defaultdict


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
        self.player_indices = list(range(self.players))
        # 各プレイヤーペアが一緒になった回数を記録
        self.pair_count = defaultdict(int)
        # 各プレイヤーの待機回数を記録
        self.wait_count = defaultdict(int)
        
    def _update_pair_counts(self, group: List[int]):
        """グループ内のペアカウントを更新"""
        for i in range(len(group)):
            for j in range(i + 1, len(group)):
                pair = tuple(sorted([group[i], group[j]]))
                self.pair_count[pair] += 1
    
    def _get_pair_score(self, player1: int, player2: int) -> int:
        """2人のプレイヤーが一緒になった回数を返す"""
        pair = tuple(sorted([player1, player2]))
        return self.pair_count.get(pair, 0)
    
    def _calculate_group_score(self, group: List[int]) -> int:
        """グループのスコア（ペアカウントの合計）を計算"""
        score = 0
        for i in range(len(group)):
            for j in range(i + 1, len(group)):
                score += self._get_pair_score(group[i], group[j])
        return score
    
    def _find_best_grouping(self, players: List[int]) -> List[List[int]]:
        """最適なグループ分けを探す"""
        best_groups = None
        best_score = float('inf')
        
        # 待機者が必要な場合（プレイヤー数が4で割り切れない場合）
        waiting_players_count = len(players) % 4
        if waiting_players_count > 0 and not (self.allow_five and len(players) % 4 == 1):
            # 待機回数が最少のプレイヤーを優先的に待機させる
            # 同じ待機回数の場合は、プレイヤーIDで決定的に選択（ランダムではなく）
            # さらに、現在のラウンド番号も考慮して順番に割り当てる
            current_round = sum(self.wait_count.values()) // max(1, waiting_players_count)  # 現在のラウンド番号を推定
            wait_priority = sorted(players, key=lambda p: (self.wait_count[p], (p + current_round) % self.players))
            
            # 必要な人数分の待機者を選択
            waiting_players = wait_priority[:waiting_players_count]
            playing_players = [p for p in players if p not in waiting_players]
            
            # プレイヤーで最適なグループを作成
            for _ in range(100):
                shuffled = playing_players.copy()
                random.shuffle(shuffled)
                groups = self._create_groups(shuffled)
                groups.append(waiting_players)  # 待機者グループを追加
                
                # グループのスコアを計算
                total_score = sum(self._calculate_group_score(g) for g in groups if len(g) >= 4)
                
                if total_score < best_score:
                    best_score = total_score
                    best_groups = groups
            
            return best_groups
        
        # 通常の処理（待機者なし）
        for _ in range(100):
            shuffled = players.copy()
            random.shuffle(shuffled)
            groups = self._create_groups(shuffled)
            
            # 全グループのスコアを計算
            total_score = sum(self._calculate_group_score(g) for g in groups)
            
            if total_score < best_score:
                best_score = total_score
                best_groups = groups
        
        return best_groups
    
    def _create_groups(self, players: List[int]) -> List[List[int]]:
        """プレイヤーリストからグループを作成"""
        groups = []
        remaining = players.copy()
        
        while len(remaining) >= 4:
            if self.allow_five and len(remaining) == 5:
                # 5人打ちありで、残り5人の場合
                groups.append(remaining[:5])
                remaining = remaining[5:]
            else:
                # 通常の4人グループ
                groups.append(remaining[:4])
                remaining = remaining[4:]
        
        # 余りがある場合（3人以下）
        if remaining:
            groups.append(remaining)
        
        return groups
    
    def generate(self) -> List[List[List[int]]]:
        """全ラウンドの卓組を生成"""
        all_rounds = []
        
        for round_num in range(self.rounds):
            # 最適なグループ分けを探す
            groups = self._find_best_grouping(self.player_indices)
            
            # ペアカウントを更新
            for group in groups:
                if len(group) >= 4:  # 4人以上のグループのみカウント
                    self._update_pair_counts(group)
                else:  # 待機グループ
                    for player in group:
                        self.wait_count[player] += 1
            
            all_rounds.append(groups)
        
        return all_rounds
    
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
        formatted['statistics'] = self._get_statistics()
        
        return formatted
    
    def _get_statistics(self) -> Dict[str, Any]:
        """統計情報を取得"""
        stats = {
            'pair_statistics': self._get_pair_statistics(),
            'wait_statistics': self._get_wait_statistics() if self.players == 5 and not self.allow_five else None
        }
        return stats
    
    def _get_pair_statistics(self) -> Dict[str, Any]:
        """ペアの統計情報を取得"""
        # 全ペア数を計算
        total_pairs = self.players * (self.players - 1) // 2
        
        # ペアカウントの集計
        count_distribution = defaultdict(int)
        for count in self.pair_count.values():
            count_distribution[count] += 1
        
        # 0回同卓のペア数を計算
        pairs_with_meetings = len(self.pair_count)
        pairs_with_zero_meetings = total_pairs - pairs_with_meetings
        if pairs_with_zero_meetings > 0:
            count_distribution[0] = pairs_with_zero_meetings
        
        # 最大同卓回数
        max_count = max(self.pair_count.values()) if self.pair_count else 0
        
        return {
            'max_count': max_count,
            'distribution': dict(count_distribution)
        }
    
    def _get_wait_statistics(self) -> List[Dict[str, Any]]:
        """待機回数の統計情報を取得"""
        wait_stats = []
        for i, name in enumerate(self.player_names):
            wait_stats.append({
                'player': name,
                'count': self.wait_count[i]
            })
        return wait_stats