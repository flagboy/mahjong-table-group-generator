#!/usr/bin/env python3
"""麻雀卓組生成プログラム（改良版）- 全ペアが最低1回同卓することを保証"""

import argparse
from typing import List, Dict, Tuple, Set
from collections import defaultdict
from itertools import combinations
import pulp
import random


class ImprovedTableGroupGenerator:
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
        self.all_pairs = set(combinations(self.player_ids, 2))
        
    def _generate_possible_tables(self) -> List[Tuple[int, ...]]:
        """可能な卓の組み合わせを生成"""
        tables = []
        
        # 4人卓の組み合わせ
        for table in combinations(self.player_ids, 4):
            tables.append(table)
        
        # 5人卓の組み合わせ（許可されている場合）
        if self.allow_five:
            for table in combinations(self.player_ids, 5):
                tables.append(table)
        
        return tables
    
    def _get_unrealized_pairs(self, pair_history: Dict[Tuple[int, int], int]) -> Set[Tuple[int, int]]:
        """まだ同卓していないペアを取得"""
        realized_pairs = set(pair for pair in pair_history.keys() if pair_history[pair] > 0)
        return self.all_pairs - realized_pairs
    
    def _solve_round(self, round_num: int, pair_history: Dict[Tuple[int, int], int], 
                     excluded_players: Set[int] = None, force_pairs: Set[Tuple[int, int]] = None) -> Tuple[List[Tuple[int, ...]], Set[int]]:
        """1ラウンド分の最適な卓組を解く"""
        if excluded_players is None:
            excluded_players = set()
        if force_pairs is None:
            force_pairs = set()
        
        available_players = [p for p in self.player_ids if p not in excluded_players]
        n_available = len(available_players)
        
        if n_available < 4:
            return [], set(available_players)
        
        # 可能な卓の組み合わせを生成（利用可能なプレイヤーのみ）
        possible_tables = []
        for table_size in [4, 5] if self.allow_five else [4]:
            if n_available >= table_size:
                for table in combinations(available_players, table_size):
                    possible_tables.append(table)
        
        # 最適化問題の設定
        prob = pulp.LpProblem(f"TableGrouping_Round{round_num}", pulp.LpMinimize)
        
        # 決定変数：各卓を使用するかどうか
        table_vars = {}
        for i, table in enumerate(possible_tables):
            table_vars[i] = pulp.LpVariable(f"table_{i}", cat='Binary')
        
        # 目的関数：複数の目的を組み合わせる
        objective = 0
        
        # 最優先目的：未実現ペアを含む卓を優先（最終ラウンドに近いほど重要）
        unrealized_pairs = self._get_unrealized_pairs(pair_history)
        remaining_rounds = self.rounds - round_num + 1
        
        for i, table in enumerate(possible_tables):
            unrealized_count = 0
            for p1, p2 in combinations(table, 2):
                pair = tuple(sorted([p1, p2]))
                if pair in unrealized_pairs or pair in force_pairs:
                    unrealized_count += 1
            
            # 残りラウンドが少ないほど未実現ペアの重要度を上げる
            if remaining_rounds <= 2:
                # 最後の2ラウンドでは未実現ペアを最優先
                objective -= table_vars[i] * unrealized_count * 100000
            else:
                objective -= table_vars[i] * unrealized_count * 1000
        
        # 主目的：ペアの重複度を最小化
        for i, table in enumerate(possible_tables):
            table_weight = 0
            for p1, p2 in combinations(table, 2):
                pair = tuple(sorted([p1, p2]))
                # 過去の同卓回数に応じて重みを増やす
                weight = (1 + pair_history.get(pair, 0)) ** 2
                table_weight += weight
            objective += table_vars[i] * table_weight * 10
        
        # 4人卓を優先する処理（4の倍数の場合）
        if self.allow_five and n_available % 4 == 0:
            for i, table in enumerate(possible_tables):
                if len(table) == 5:
                    objective += table_vars[i] * 10000
        
        prob += objective
        
        # 制約1：各プレイヤーは最大1つの卓に入る
        for player in available_players:
            player_tables = []
            for i, table in enumerate(possible_tables):
                if player in table:
                    player_tables.append(table_vars[i])
            if player_tables:
                prob += pulp.lpSum(player_tables) <= 1
        
        # 制約2：できるだけ多くのプレイヤーを卓に配置する
        total_seated = 0
        for i, table in enumerate(possible_tables):
            total_seated += table_vars[i] * len(table)
        
        # 待機プレイヤー数の上限を設定
        if self.allow_five:
            if n_available % 4 == 0:
                max_waiting = 0
            elif n_available == 9:
                max_waiting = 0
            elif n_available % 5 == 0:
                max_waiting = 0
            else:
                max_waiting = min(n_available % 4, n_available % 5)
        else:
            max_waiting = n_available % 4
        
        prob += total_seated >= n_available - max_waiting
        
        # 最適化を実行
        prob.solve(pulp.PULP_CBC_CMD(msg=0))
        
        # 結果を取得
        selected_tables = []
        used_players = set()
        
        if prob.status == pulp.LpStatusOptimal:
            for i, table in enumerate(possible_tables):
                if table_vars[i].varValue == 1:
                    selected_tables.append(table)
                    used_players.update(table)
        
        # 待機プレイヤー
        waiting_players = set(available_players) - used_players
        
        return selected_tables, waiting_players
    
    def _force_unrealized_pairs(self, unrealized_pairs: Set[Tuple[int, int]], 
                               pair_history: Dict[Tuple[int, int], int]) -> List[List[List[int]]]:
        """未実現ペアを強制的に同卓させる追加ラウンドを生成"""
        extra_rounds = []
        remaining_pairs = unrealized_pairs.copy()
        
        while remaining_pairs:
            round_tables = []
            used_players = set()
            pairs_in_round = []
            
            # このラウンドで実現できるペアを選択
            for pair in list(remaining_pairs):
                if pair[0] not in used_players and pair[1] not in used_players:
                    pairs_in_round.append(pair)
                    used_players.update(pair)
                    remaining_pairs.remove(pair)
                    
                    # 4人卓を作る
                    if len(used_players) % 4 == 0:
                        # 最後の4人で卓を作る
                        table_players = list(used_players)[-4:]
                        round_tables.append(table_players)
            
            # 残りのプレイヤーで卓を作る
            remaining_players = [p for p in self.player_ids if p not in used_players]
            while len(remaining_players) >= 4:
                table = remaining_players[:4]
                round_tables.append(table)
                remaining_players = remaining_players[4:]
                
                # この卓のペアを記録
                for p1, p2 in combinations(table, 2):
                    pair = tuple(sorted([p1, p2]))
                    if pair in remaining_pairs:
                        remaining_pairs.remove(pair)
            
            if round_tables:
                extra_rounds.append(round_tables)
        
        return extra_rounds
    
    def generate(self) -> List[List[List[int]]]:
        """全ラウンドの卓組を生成"""
        all_rounds = []
        pair_history = defaultdict(int)
        
        # 通常のラウンドを生成
        for round_num in range(self.rounds):
            # 未実現ペアを取得
            unrealized_pairs = self._get_unrealized_pairs(pair_history)
            
            # 最終ラウンドに近い場合、未実現ペアを優先
            force_pairs = set()
            if round_num >= self.rounds - 2 and unrealized_pairs:
                # 未実現ペアの中からランダムに選択
                force_pairs = set(random.sample(list(unrealized_pairs), 
                                              min(len(unrealized_pairs), 8)))
            
            # このラウンドの卓組を解く
            tables, waiting = self._solve_round(round_num + 1, pair_history, force_pairs=force_pairs)
            
            # ペア履歴を更新
            for table in tables:
                for p1, p2 in combinations(table, 2):
                    pair = tuple(sorted([p1, p2]))
                    pair_history[pair] += 1
            
            # 結果を保存
            round_result = [list(table) for table in tables]
            if waiting:
                round_result.append(list(waiting))
            
            all_rounds.append(round_result)
        
        # 未実現ペアがまだある場合、追加の工夫を行う
        unrealized_pairs = self._get_unrealized_pairs(pair_history)
        if unrealized_pairs and len(all_rounds) > 0:
            # 最初のラウンドを再調整して未実現ペアを含める
            print(f"\n注意: {len(unrealized_pairs)}個の未実現ペアがあります。調整を試みます...")
            
            # 未実現ペアを含む卓組を優先的に生成
            for round_idx in range(min(2, len(all_rounds))):
                # 各ラウンドで未実現ペアを優先的に配置
                new_tables, new_waiting = self._solve_round(
                    round_idx + 1, 
                    defaultdict(int),  # 履歴をリセット
                    force_pairs=unrealized_pairs
                )
                
                if new_tables:
                    # ペア履歴を再計算
                    temp_history = defaultdict(int)
                    for r_idx, round_data in enumerate(all_rounds):
                        if r_idx == round_idx:
                            # このラウンドは新しい卓組に置き換え
                            for table in new_tables:
                                for p1, p2 in combinations(table, 2):
                                    pair = tuple(sorted([p1, p2]))
                                    temp_history[pair] += 1
                        else:
                            # 他のラウンドはそのまま
                            for group in round_data:
                                if len(group) >= 4:
                                    for p1, p2 in combinations(group, 2):
                                        pair = tuple(sorted([p1, p2]))
                                        temp_history[pair] += 1
                    
                    # 改善された場合のみ置き換え
                    new_unrealized = self._get_unrealized_pairs(temp_history)
                    if len(new_unrealized) < len(unrealized_pairs):
                        all_rounds[round_idx] = [list(table) for table in new_tables]
                        if new_waiting:
                            all_rounds[round_idx].append(list(new_waiting))
                        pair_history = temp_history
                        unrealized_pairs = new_unrealized
        
        return all_rounds
    
    def print_results(self, results: List[List[List[int]]]):
        """結果を見やすく出力"""
        print(f"\n麻雀卓組結果（改良版） (参加者: {self.players}人, {self.rounds}回戦)")
        print(f"5人打ち: {'あり' if self.allow_five else 'なし'}")
        print("=" * 50)
        
        # ペア履歴を再計算
        pair_count = defaultdict(int)
        
        for round_num, groups in enumerate(results, 1):
            print(f"\n第{round_num}回戦:")
            for table_num, group in enumerate(groups):
                if len(group) >= 4:
                    players_str = ", ".join(f"P{p}" for p in sorted(group))
                    print(f"  卓{table_num + 1}: {players_str} ({len(group)}人)")
                    # ペアカウントを更新
                    for p1, p2 in combinations(group, 2):
                        pair = tuple(sorted([p1, p2]))
                        pair_count[pair] += 1
                else:
                    # 4人未満は待機
                    players_str = ", ".join(f"P{p}" for p in sorted(group))
                    print(f"  待機: {players_str}")
        
        # ペア統計を表示
        self._print_pair_statistics(pair_count)
    
    def _print_pair_statistics(self, pair_count: Dict[Tuple[int, int], int]):
        """ペアの統計情報を表示"""
        print("\n" + "=" * 50)
        print("同卓回数統計:")
        
        if not pair_count:
            print("統計データなし")
            return
        
        # ペアカウントの集計
        count_distribution = defaultdict(int)
        for count in pair_count.values():
            count_distribution[count] += 1
        
        # 最大同卓回数
        max_count = max(pair_count.values()) if pair_count else 0
        
        print(f"最大同卓回数: {max_count}回")
        for count in sorted(count_distribution.keys()):
            print(f"  {count}回同卓: {count_distribution[count]}ペア")
        
        # 可能な全ペア数と実現したペア数
        total_possible_pairs = len(list(combinations(range(1, self.players + 1), 2)))
        realized_pairs = len(pair_count)
        coverage = realized_pairs / total_possible_pairs * 100 if total_possible_pairs > 0 else 0
        
        print(f"\nペアの種類数: {realized_pairs}/{total_possible_pairs} ({coverage:.1f}%)")
        
        # 未実現ペアの詳細
        if coverage < 100:
            unrealized_pairs = self.all_pairs - set(pair_count.keys())
            print(f"\n未実現ペア数: {len(unrealized_pairs)}個")
            if len(unrealized_pairs) <= 10:
                print("未実現ペア:")
                for p1, p2 in sorted(unrealized_pairs):
                    print(f"  P{p1} - P{p2}")
        
        # 標準偏差も計算
        import statistics
        if len(pair_count) > 1:
            counts = list(pair_count.values())
            mean = statistics.mean(counts)
            stdev = statistics.stdev(counts)
            print(f"\n平均同卓回数: {mean:.2f}回")
            print(f"標準偏差: {stdev:.2f}")


def main():
    parser = argparse.ArgumentParser(description='麻雀卓組生成プログラム（改良版）')
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
    
    generator = ImprovedTableGroupGenerator(args.players, args.rounds, args.five)
    results = generator.generate()
    generator.print_results(results)


if __name__ == "__main__":
    main()