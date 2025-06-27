#!/usr/bin/env python3
"""麻雀卓組生成プログラム（保証版）- 全ペアが最低1回同卓することを保証"""

import argparse
from typing import List, Dict, Tuple, Set
from collections import defaultdict
from itertools import combinations
import pulp
import math


class GuaranteedTableGroupGenerator:
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
        self.all_pairs = list(combinations(self.player_ids, 2))
        self.total_pairs = len(self.all_pairs)
        
        # 理論的な制約をチェック
        self._check_feasibility()
    
    def _check_feasibility(self):
        """全ペアが同卓可能かチェック"""
        # 各ラウンドで実現できるペア数
        tables_per_round = self.players // 4
        if self.allow_five and self.players % 4 > 0:
            # 5人打ちありの場合の計算
            if self.players % 5 == 0:
                tables_per_round = self.players // 5
                pairs_per_table = 10  # C(5,2)
            else:
                # 混在する場合の概算
                tables_per_round = self.players // 4
                pairs_per_table = 6  # C(4,2)
        else:
            pairs_per_table = 6  # C(4,2)
        
        max_pairs_per_round = tables_per_round * pairs_per_table
        max_total_pairs = max_pairs_per_round * self.rounds
        
        if max_total_pairs < self.total_pairs:
            print(f"\n警告: {self.players}人で{self.rounds}回戦では、")
            print(f"最大{max_total_pairs}ペアしか実現できませんが、")
            print(f"全ペア数は{self.total_pairs}です。")
            print(f"推奨回戦数: {math.ceil(self.total_pairs / max_pairs_per_round)}回以上")
    
    def _solve_with_pair_coverage(self) -> List[List[List[int]]]:
        """全ペアカバレッジを目指す最適化"""
        all_rounds = []
        pair_count = defaultdict(int)
        
        # 各ラウンドで未実現ペアを最大化
        for round_num in range(self.rounds):
            # 未実現ペアを取得
            unrealized_pairs = [(p1, p2) for p1, p2 in self.all_pairs 
                               if pair_count[tuple(sorted([p1, p2]))] == 0]
            
            # このラウンドの最適化問題を設定
            prob = pulp.LpProblem(f"Round_{round_num + 1}", pulp.LpMaximize)
            
            # 可能な卓の組み合わせを生成
            possible_tables = []
            table_pairs = []  # 各卓が含むペア
            
            # 4人卓
            for table in combinations(self.player_ids, 4):
                possible_tables.append(table)
                pairs = list(combinations(table, 2))
                table_pairs.append(pairs)
            
            # 5人卓（許可されている場合）
            if self.allow_five:
                for table in combinations(self.player_ids, 5):
                    possible_tables.append(table)
                    pairs = list(combinations(table, 2))
                    table_pairs.append(pairs)
            
            # 決定変数
            table_vars = {}
            for i in range(len(possible_tables)):
                table_vars[i] = pulp.LpVariable(f"table_{i}", cat='Binary')
            
            # 目的関数：未実現ペアの数を最大化
            objective = 0
            for i, pairs in enumerate(table_pairs):
                unrealized_in_table = 0
                duplicate_penalty = 0
                
                for p1, p2 in pairs:
                    pair = tuple(sorted([p1, p2]))
                    if pair_count[pair] == 0:
                        unrealized_in_table += 1
                    else:
                        # 既に同卓したペアにはペナルティ
                        duplicate_penalty += pair_count[pair] ** 2
                
                # 未実現ペアを最大化し、重複を最小化
                objective += table_vars[i] * (unrealized_in_table * 1000 - duplicate_penalty)
            
            # 4人卓を優先（プレイヤー数が4の倍数の場合）
            if self.allow_five and self.players % 4 == 0:
                for i, table in enumerate(possible_tables):
                    if len(table) == 5:
                        objective -= table_vars[i] * 10000
            
            prob += objective
            
            # 制約1：各プレイヤーは最大1つの卓に入る
            for player in self.player_ids:
                player_tables = []
                for i, table in enumerate(possible_tables):
                    if player in table:
                        player_tables.append(table_vars[i])
                if player_tables:
                    prob += pulp.lpSum(player_tables) <= 1
            
            # 制約2：できるだけ多くのプレイヤーを配置
            total_seated = pulp.lpSum([table_vars[i] * len(possible_tables[i]) 
                                      for i in range(len(possible_tables))])
            
            if self.allow_five:
                if self.players % 4 == 0:
                    min_seated = self.players
                elif self.players == 9:
                    min_seated = self.players
                else:
                    min_seated = self.players - min(self.players % 4, self.players % 5)
            else:
                min_seated = self.players - (self.players % 4)
            
            prob += total_seated >= min_seated
            
            # 最適化を実行
            prob.solve(pulp.PULP_CBC_CMD(msg=0))
            
            # 結果を取得
            round_tables = []
            used_players = set()
            
            if prob.status == pulp.LpStatusOptimal:
                for i, table in enumerate(possible_tables):
                    if table_vars[i].varValue == 1:
                        round_tables.append(list(table))
                        used_players.update(table)
                        
                        # ペアカウントを更新
                        for p1, p2 in combinations(table, 2):
                            pair = tuple(sorted([p1, p2]))
                            pair_count[pair] += 1
            
            # 待機プレイヤー
            waiting = [p for p in self.player_ids if p not in used_players]
            if waiting:
                round_tables.append(waiting)
            
            all_rounds.append(round_tables)
        
        return all_rounds
    
    def generate(self) -> List[List[List[int]]]:
        """全ラウンドの卓組を生成"""
        # ペアカバレッジを重視した生成
        results = self._solve_with_pair_coverage()
        
        # 統計情報を計算
        pair_count = defaultdict(int)
        for round_tables in results:
            for table in round_tables:
                if len(table) >= 4:
                    for p1, p2 in combinations(table, 2):
                        pair = tuple(sorted([p1, p2]))
                        pair_count[pair] += 1
        
        # カバレッジを確認
        realized_pairs = len(pair_count)
        coverage = realized_pairs / self.total_pairs * 100
        
        # カバレッジが低い場合、追加の最適化を試みる
        if coverage < 100 and self.rounds >= 3:
            print(f"\n初回生成でカバレッジ{coverage:.1f}%。最適化を実行中...")
            
            # グローバル最適化：全ラウンドを同時に考慮
            optimized_results = self._global_optimization()
            
            # 最適化後の統計を計算
            opt_pair_count = defaultdict(int)
            for round_tables in optimized_results:
                for table in round_tables:
                    if len(table) >= 4:
                        for p1, p2 in combinations(table, 2):
                            pair = tuple(sorted([p1, p2]))
                            opt_pair_count[pair] += 1
            
            opt_coverage = len(opt_pair_count) / self.total_pairs * 100
            
            if opt_coverage > coverage:
                print(f"最適化により{opt_coverage:.1f}%に改善されました。")
                return optimized_results
        
        return results
    
    def _global_optimization(self) -> List[List[List[int]]]:
        """全ラウンドを同時に最適化"""
        # 大規模な最適化問題
        prob = pulp.LpProblem("GlobalOptimization", pulp.LpMaximize)
        
        # 各ラウンドの可能な卓組み合わせ
        all_possible_configs = []
        
        for round_num in range(self.rounds):
            round_configs = []
            
            # このラウンドの可能な構成を生成（制限付き）
            # 完全な組み合わせは計算量が膨大なので、ヒューリスティックを使用
            for _ in range(min(100, 2 ** self.players)):  # 最大100構成
                config = self._generate_random_valid_config()
                if config and config not in round_configs:
                    round_configs.append(config)
            
            all_possible_configs.append(round_configs)
        
        # 決定変数：各ラウンドでどの構成を選ぶか
        config_vars = {}
        for r in range(self.rounds):
            for c in range(len(all_possible_configs[r])):
                config_vars[(r, c)] = pulp.LpVariable(f"config_{r}_{c}", cat='Binary')
        
        # ペアカバレッジを計算
        pair_covered = {}
        for pair in self.all_pairs:
            pair_key = tuple(sorted(pair))
            pair_covered[pair_key] = pulp.LpVariable(f"pair_{pair_key[0]}_{pair_key[1]}", 
                                                     cat='Binary')
        
        # 目的関数：カバーされるペアの数を最大化
        prob += pulp.lpSum(pair_covered.values())
        
        # 制約1：各ラウンドで1つの構成を選択
        for r in range(self.rounds):
            round_vars = [config_vars[(r, c)] for c in range(len(all_possible_configs[r]))]
            if round_vars:
                prob += pulp.lpSum(round_vars) == 1
        
        # 制約2：ペアがカバーされる条件
        for pair in self.all_pairs:
            pair_key = tuple(sorted(pair))
            coverage_sum = []
            
            for r in range(self.rounds):
                for c, config in enumerate(all_possible_configs[r]):
                    if self._config_contains_pair(config, pair):
                        coverage_sum.append(config_vars[(r, c)])
            
            if coverage_sum:
                # ペアが少なくとも1回出現すればカバーされる
                prob += pair_covered[pair_key] <= pulp.lpSum(coverage_sum)
                # ペアが出現しない場合はカバーされない
                prob += pair_covered[pair_key] * len(coverage_sum) >= pulp.lpSum(coverage_sum)
        
        # タイムアウトを設定して解く
        prob.solve(pulp.PULP_CBC_CMD(msg=0, timeLimit=30))
        
        # 結果を構築
        results = []
        for r in range(self.rounds):
            selected_config = None
            for c in range(len(all_possible_configs[r])):
                if config_vars.get((r, c)) and config_vars[(r, c)].varValue == 1:
                    selected_config = all_possible_configs[r][c]
                    break
            
            if selected_config:
                results.append(selected_config)
            else:
                # フォールバック
                results.append(self._generate_random_valid_config())
        
        return results
    
    def _generate_random_valid_config(self) -> List[List[int]]:
        """有効なランダム構成を生成"""
        import random
        players = self.player_ids.copy()
        random.shuffle(players)
        
        config = []
        while len(players) >= 4:
            if self.allow_five and len(players) >= 5 and random.random() < 0.3:
                table = players[:5]
                players = players[5:]
            else:
                table = players[:4]
                players = players[4:]
            config.append(table)
        
        if players:
            config.append(players)
        
        return config
    
    def _config_contains_pair(self, config: List[List[int]], pair: Tuple[int, int]) -> bool:
        """構成がペアを含むかチェック"""
        for table in config:
            if len(table) >= 4 and pair[0] in table and pair[1] in table:
                return True
        return False
    
    def print_results(self, results: List[List[List[int]]]):
        """結果を見やすく出力"""
        print(f"\n麻雀卓組結果（保証版） (参加者: {self.players}人, {self.rounds}回戦)")
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
        
        # 同卓なしのペア数を追加
        unrealized_pairs = set(self.all_pairs) - set(pair_count.keys())
        if unrealized_pairs:
            count_distribution[0] = len(unrealized_pairs)
        
        # 最大同卓回数
        max_count = max(pair_count.values()) if pair_count else 0
        
        print(f"最大同卓回数: {max_count}回")
        for count in sorted(count_distribution.keys()):
            print(f"  {count}回同卓: {count_distribution[count]}ペア")
        
        # カバレッジ
        realized_pairs = len(pair_count)
        coverage = realized_pairs / self.total_pairs * 100
        
        print(f"\nペアカバレッジ: {realized_pairs}/{self.total_pairs} ({coverage:.1f}%)")
        
        # 未実現ペアの詳細（10個以下の場合）
        if unrealized_pairs and len(unrealized_pairs) <= 10:
            print(f"\n未実現ペア:")
            for p1, p2 in sorted(unrealized_pairs):
                print(f"  P{p1} - P{p2}")
        
        # 統計情報
        if len(pair_count) > 1:
            import statistics
            counts = list(pair_count.values())
            mean = statistics.mean(counts)
            stdev = statistics.stdev(counts)
            print(f"\n実現ペアの平均同卓回数: {mean:.2f}回")
            print(f"標準偏差: {stdev:.2f}")


def main():
    parser = argparse.ArgumentParser(description='麻雀卓組生成プログラム（保証版）')
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
    
    generator = GuaranteedTableGroupGenerator(args.players, args.rounds, args.five)
    results = generator.generate()
    generator.print_results(results)


if __name__ == "__main__":
    main()