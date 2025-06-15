#!/usr/bin/env python3
"""麻雀卓組生成プログラム"""

import random
from typing import List, Set, Tuple
from collections import defaultdict
import argparse


class TableGroupGenerator:
    def __init__(self, players: int, rounds: int, allow_five: bool = False):
        """
        Args:
            players: 参加人数
            rounds: 回数
            allow_five: 5人打ちを許可するか
        """
        self.players = players
        self.rounds = rounds
        self.allow_five = allow_five
        self.player_ids = list(range(1, players + 1))
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
            wait_priority = sorted(players, key=lambda p: (self.wait_count[p], (p - 1 + current_round) % self.players))
            
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
            groups = self._find_best_grouping(self.player_ids)
            
            # ペアカウントを更新
            for group in groups:
                if len(group) >= 4:  # 4人以上のグループのみカウント
                    self._update_pair_counts(group)
                else:  # 待機グループ
                    for player in group:
                        self.wait_count[player] += 1
            
            all_rounds.append(groups)
        
        return all_rounds
    
    def print_results(self, results: List[List[List[int]]]):
        """結果を見やすく出力"""
        print(f"\n麻雀卓組結果 (参加者: {self.players}人, {self.rounds}回戦)")
        print(f"5人打ち: {'あり' if self.allow_five else 'なし'}")
        print("=" * 50)
        
        for round_num, groups in enumerate(results, 1):
            print(f"\n第{round_num}回戦:")
            for table_num, group in enumerate(groups, 1):
                if len(group) >= 4:
                    players_str = ", ".join(f"P{p}" for p in group)
                    print(f"  卓{table_num}: {players_str} ({len(group)}人)")
                else:
                    # 4人未満は待機
                    players_str = ", ".join(f"P{p}" for p in group)
                    print(f"  待機: {players_str}")
        
        # ペア統計を表示
        self._print_pair_statistics()
        
        # 待機回数統計を表示（5人で5人打ちなしの場合）
        if self.players == 5 and not self.allow_five:
            self._print_wait_statistics()
    
    def _print_pair_statistics(self):
        """ペアの統計情報を表示"""
        print("\n" + "=" * 50)
        print("同卓回数統計:")
        
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
        
        print(f"最大同卓回数: {max_count}回")
        for count in sorted(count_distribution.keys()):
            print(f"  {count}回同卓: {count_distribution[count]}ペア")
    
    def _print_wait_statistics(self):
        """待機回数の統計情報を表示"""
        print("\n" + "=" * 50)
        print("待機回数統計:")
        
        for player in range(1, self.players + 1):
            wait_times = self.wait_count[player]
            print(f"  P{player}: {wait_times}回")


def main():
    parser = argparse.ArgumentParser(description='麻雀卓組生成プログラム')
    parser.add_argument('players', type=int, help='参加人数')
    parser.add_argument('rounds', type=int, help='回数')
    parser.add_argument('--five', action='store_true', help='5人打ちを許可')
    
    args = parser.parse_args()
    
    if args.players < 4:
        print("エラー: 参加人数は4人以上必要です")
        return
    
    if args.rounds < 1:
        print("エラー: 回数は1以上必要です")
        return
    
    generator = TableGroupGenerator(args.players, args.rounds, args.five)
    results = generator.generate()
    generator.print_results(results)


if __name__ == "__main__":
    main()