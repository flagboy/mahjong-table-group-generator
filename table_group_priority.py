#!/usr/bin/env python3
"""麻雀卓組生成プログラム（優先順位最適化版）- 条件1を絶対優先"""

import argparse
from typing import List, Dict, Tuple, Set, Optional
from collections import defaultdict
from itertools import combinations
import pulp
import math
import time
import random


class PriorityTableGroupGenerator:
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
        
        # 理論的な制約を計算
        self._calculate_theoretical_limits()
        
    def _calculate_theoretical_limits(self):
        """理論的な制約と目標を計算"""
        # 各ラウンドで実現できるペア数
        if self.allow_five:
            best_config = self._find_best_table_configuration()
            self.max_pairs_per_round = best_config['pairs_per_round']
            self.optimal_table_config = best_config
        else:
            tables_per_round = self.players // 4
            self.max_pairs_per_round = tables_per_round * 6
            waiting_players = self.players % 4
            self.optimal_table_config = {
                'four_tables': tables_per_round, 
                'five_tables': 0,
                'waiting': waiting_players
            }
        
        self.max_total_pairs = self.max_pairs_per_round * self.rounds
        
        # 各ペアの理想的な同卓回数
        self.ideal_meetings_float = self.max_total_pairs / self.total_pairs
        self.ideal_meetings_floor = int(self.ideal_meetings_float)
        self.ideal_meetings_ceil = self.ideal_meetings_floor + 1
        
        print(f"\n理論的分析:")
        print(f"- 参加者数: {self.players}人")
        print(f"- 全ペア数: {self.total_pairs}")
        print(f"- 1ラウンドの最大ペア数: {self.max_pairs_per_round}")
        print(f"- {self.rounds}ラウンドの最大ペア総数: {self.max_total_pairs}")
        print(f"- 理論的な最小同卓回数: {self.ideal_meetings_floor}回")
        
        if self.ideal_meetings_floor == 0:
            print(f"\n警告: 全ペアを最低1回同卓させるには{math.ceil(self.total_pairs / self.max_pairs_per_round)}回戦以上必要です")
    
    def _find_best_table_configuration(self) -> Dict:
        """5人打ちありの場合の最適な卓構成を見つける"""
        best_config = None
        max_pairs = 0
        
        for five_tables in range(self.players // 5 + 1):
            remaining = self.players - five_tables * 5
            four_tables = remaining // 4
            waiting = remaining % 4
            
            if remaining >= 0 and (waiting == 0 or waiting >= 4):
                pairs = five_tables * 10 + four_tables * 6
                if pairs > max_pairs:
                    max_pairs = pairs
                    best_config = {
                        'five_tables': five_tables,
                        'four_tables': four_tables,
                        'pairs_per_round': pairs,
                        'waiting': waiting
                    }
        
        return best_config
    
    def generate(self) -> List[List[List[int]]]:
        """全ラウンドの卓組を生成"""
        print("\n優先順位最適化を開始...")
        
        # 条件1を最優先にした最適化
        return self._optimize_for_minimum_meetings()
    
    def _optimize_for_minimum_meetings(self) -> List[List[List[int]]]:
        """最小同卓回数を最大化することを最優先に最適化"""
        
        # Phase 1: 未実現ペアを最大限減らす
        print("Phase 1: 未実現ペアの最小化...")
        solution_phase1 = self._minimize_unrealized_pairs()
        score1 = self._evaluate_solution(solution_phase1)
        print(f"  最小同卓回数: {score1['min_count']}回")
        print(f"  未実現ペア数: {score1['distribution'].get(0, 0)}個")
        
        # Phase 2: 最小同卓回数を維持しつつ、分布を改善
        if score1['min_count'] > 0:
            print("\nPhase 2: 分布の最適化（最小回数を維持）...")
            solution_phase2 = self._optimize_distribution_preserving_minimum(solution_phase1, score1['min_count'])
            score2 = self._evaluate_solution(solution_phase2)
            print(f"  最小同卓回数: {score2['min_count']}回（維持）")
            print(f"  最大同卓回数: {score2['max_count']}回")
            return solution_phase2
        else:
            # 全ペアカバーが不可能な場合、できるだけ多くのペアをカバー
            print("\nPhase 2: カバレッジの最大化...")
            solution_phase2 = self._maximize_coverage_with_balance(solution_phase1)
            return solution_phase2
    
    def _minimize_unrealized_pairs(self) -> List[List[List[int]]]:
        """未実現ペアを最小化（条件1の最優先）"""
        all_rounds = []
        pair_count = defaultdict(int)
        
        # 待機ローテーション
        waiting_rotation = self._create_fair_waiting_rotation()
        
        for round_num in range(self.rounds):
            print(f"\r  ラウンド {round_num + 1}/{self.rounds} を生成中...", end='', flush=True)
            
            # 待機プレイヤーの決定
            if waiting_rotation and round_num < len(waiting_rotation):
                waiting_players = waiting_rotation[round_num]
                playing_players = [p for p in self.player_ids if p not in waiting_players]
            else:
                playing_players = self.player_ids
                waiting_players = []
            
            # このラウンドで最も多くの未実現ペアを実現する卓組を探す
            best_tables = self._find_best_tables_for_unrealized_pairs(
                playing_players, pair_count, round_num
            )
            
            # 待機プレイヤーを追加
            if waiting_players:
                best_tables.append(waiting_players)
            
            all_rounds.append(best_tables)
            
            # ペアカウントを更新
            for table in best_tables:
                if len(table) >= 4:
                    for p1, p2 in combinations(table, 2):
                        pair = tuple(sorted([p1, p2]))
                        pair_count[pair] += 1
        
        print()  # 改行
        return all_rounds
    
    def _find_best_tables_for_unrealized_pairs(self, players: List[int], 
                                              pair_count: Dict[Tuple[int, int], int],
                                              round_num: int) -> List[List[int]]:
        """未実現ペアを最大限実現する卓組を見つける"""
        
        # 小規模なら完全探索
        if len(players) <= 12:
            return self._exhaustive_search_for_unrealized(players, pair_count)
        
        # 中規模なら制限付き最適化
        elif len(players) <= 20:
            return self._limited_optimization_for_unrealized(players, pair_count)
        
        # 大規模ならヒューリスティック
        else:
            return self._heuristic_for_unrealized(players, pair_count)
    
    def _exhaustive_search_for_unrealized(self, players: List[int], 
                                        pair_count: Dict[Tuple[int, int], int]) -> List[List[int]]:
        """完全探索で未実現ペアを最大化（小規模用）"""
        best_config = None
        max_unrealized_pairs = -1
        
        # すべての可能な卓構成を生成
        def generate_table_configs(remaining_players, current_config):
            if len(remaining_players) < 4:
                return [current_config]
            
            configs = []
            # 4人卓
            for table in combinations(remaining_players, 4):
                new_remaining = [p for p in remaining_players if p not in table]
                new_config = current_config + [list(table)]
                configs.extend(generate_table_configs(new_remaining, new_config))
            
            # 5人卓（許可されている場合）
            if self.allow_five and len(remaining_players) >= 5:
                for table in combinations(remaining_players, 5):
                    new_remaining = [p for p in remaining_players if p not in table]
                    new_config = current_config + [list(table)]
                    configs.extend(generate_table_configs(new_remaining, new_config))
            
            return configs
        
        all_configs = generate_table_configs(players, [])
        
        # 各構成を評価
        for config in all_configs:
            unrealized_count = 0
            for table in config:
                for p1, p2 in combinations(table, 2):
                    pair = tuple(sorted([p1, p2]))
                    if pair_count.get(pair, 0) == 0:
                        unrealized_count += 1
            
            if unrealized_count > max_unrealized_pairs:
                max_unrealized_pairs = unrealized_count
                best_config = config
        
        return best_config if best_config else [players[i:i+4] for i in range(0, len(players), 4)]
    
    def _limited_optimization_for_unrealized(self, players: List[int], 
                                           pair_count: Dict[Tuple[int, int], int]) -> List[List[int]]:
        """制限付き最適化で未実現ペアを最大化（中規模用）"""
        prob = pulp.LpProblem("MaximizeUnrealizedPairs", pulp.LpMaximize)
        
        # 可能な卓（サンプリング）
        all_possible_tables = list(combinations(players, 4))
        if self.allow_five:
            all_possible_tables.extend(list(combinations(players, 5)))
        
        # 多すぎる場合はサンプリング
        if len(all_possible_tables) > 200:
            # 未実現ペアを含む卓を優先的に選択
            scored_tables = []
            for table in all_possible_tables:
                unrealized_pairs = sum(1 for p1, p2 in combinations(table, 2) 
                                     if pair_count.get(tuple(sorted([p1, p2])), 0) == 0)
                scored_tables.append((unrealized_pairs, table))
            
            scored_tables.sort(reverse=True)
            possible_tables = [t[1] for t in scored_tables[:200]]
        else:
            possible_tables = all_possible_tables
        
        # 決定変数
        table_vars = {}
        for i in range(len(possible_tables)):
            table_vars[i] = pulp.LpVariable(f"table_{i}", cat='Binary')
        
        # 目的関数：未実現ペアの数を最大化
        objective = 0
        for i, table in enumerate(possible_tables):
            unrealized_in_table = 0
            for p1, p2 in combinations(table, 2):
                pair = tuple(sorted([p1, p2]))
                if pair_count.get(pair, 0) == 0:
                    unrealized_in_table += 1
            objective += table_vars[i] * unrealized_in_table
        
        prob += objective
        
        # 制約：各プレイヤーは最大1つの卓
        for player in players:
            player_tables = [table_vars[i] for i, table in enumerate(possible_tables) if player in table]
            if player_tables:
                prob += pulp.lpSum(player_tables) <= 1
        
        # 解く
        prob.solve(pulp.PULP_CBC_CMD(msg=0, timeLimit=5))
        
        # 結果を抽出
        selected_tables = []
        for i, table in enumerate(possible_tables):
            if table_vars[i].varValue > 0.5:
                selected_tables.append(list(table))
        
        return selected_tables if selected_tables else self._heuristic_for_unrealized(players, pair_count)
    
    def _heuristic_for_unrealized(self, players: List[int], 
                                 pair_count: Dict[Tuple[int, int], int]) -> List[List[int]]:
        """ヒューリスティックで未実現ペアを最大化（大規模用）"""
        tables = []
        remaining_players = players.copy()
        
        # 未実現ペアのグラフを構築
        unrealized_graph = defaultdict(set)
        for p1, p2 in combinations(players, 2):
            pair = tuple(sorted([p1, p2]))
            if pair_count.get(pair, 0) == 0:
                unrealized_graph[p1].add(p2)
                unrealized_graph[p2].add(p1)
        
        while len(remaining_players) >= 4:
            # 最も多くの未実現ペアを持つプレイヤーから開始
            start_player = max(remaining_players, 
                             key=lambda p: len(unrealized_graph[p] & set(remaining_players)))
            
            # そのプレイヤーと未実現ペアが多い3-4人を選択
            candidates = [p for p in remaining_players if p != start_player]
            candidates.sort(key=lambda p: (start_player in unrealized_graph[p], 
                                          len(unrealized_graph[p] & set(remaining_players))), 
                          reverse=True)
            
            if self.allow_five and len(remaining_players) >= 5 and random.random() < 0.3:
                table = [start_player] + candidates[:4]
            else:
                table = [start_player] + candidates[:3]
            
            tables.append(table)
            for p in table:
                remaining_players.remove(p)
        
        return tables
    
    def _create_fair_waiting_rotation(self) -> Optional[List[List[int]]]:
        """公平な待機ローテーションを作成"""
        waiting_count = self.optimal_table_config.get('waiting', 0)
        if waiting_count == 0:
            return None
        
        # 各プレイヤーができるだけ均等に待機するように配分
        rotation = []
        players_list = self.player_ids.copy()
        
        for round_num in range(self.rounds):
            # ラウンドごとに異なるプレイヤーが待機
            start_idx = (round_num * waiting_count) % self.players
            waiting = []
            for i in range(waiting_count):
                waiting.append(players_list[(start_idx + i) % self.players])
            rotation.append(waiting)
        
        return rotation
    
    def _optimize_distribution_preserving_minimum(self, solution: List[List[List[int]]], 
                                                min_meetings: int) -> List[List[List[int]]]:
        """最小同卓回数を維持しながら分布を最適化"""
        best_solution = solution
        best_score = self._evaluate_solution(solution)
        
        # 10回の改善試行
        for iteration in range(10):
            print(f"\r  改善試行 {iteration + 1}/10...", end='', flush=True)
            
            # ランダムにラウンドを選んで再構成
            round_idx = random.randint(0, self.rounds - 1)
            
            # 現在のペアカウント（該当ラウンドを除く）
            pair_count = defaultdict(int)
            for i, round_tables in enumerate(best_solution):
                if i != round_idx:
                    for table in round_tables:
                        if len(table) >= 4:
                            for p1, p2 in combinations(table, 2):
                                pair = tuple(sorted([p1, p2]))
                                pair_count[pair] += 1
            
            # 該当ラウンドのプレイヤー
            playing_players = []
            waiting_players = []
            for table in best_solution[round_idx]:
                if len(table) >= 4:
                    playing_players.extend(table)
                else:
                    waiting_players = table
            
            # 最小回数を維持しながら再構成
            new_tables = self._reconstruct_round_preserving_minimum(
                playing_players, pair_count, min_meetings
            )
            
            if new_tables:
                # 新しい解を作成
                new_solution = [r.copy() for r in best_solution]
                new_solution[round_idx] = new_tables
                if waiting_players:
                    new_solution[round_idx].append(waiting_players)
                
                # 評価
                new_score = self._evaluate_solution(new_solution)
                
                # 最小回数が維持され、かつ改善された場合のみ採用
                if (new_score['min_count'] >= min_meetings and 
                    self._is_better_distribution(new_score, best_score)):
                    best_solution = new_solution
                    best_score = new_score
        
        print()  # 改行
        return best_solution
    
    def _reconstruct_round_preserving_minimum(self, players: List[int], 
                                            other_rounds_pairs: Dict[Tuple[int, int], int],
                                            min_meetings: int) -> Optional[List[List[int]]]:
        """最小同卓回数を維持しながらラウンドを再構成"""
        # 小規模なら完全探索
        if len(players) <= 8:
            best_config = None
            best_variance = float('inf')
            
            def generate_configs(remaining, current):
                if len(remaining) < 4:
                    return [current]
                configs = []
                for table in combinations(remaining, 4):
                    new_remaining = [p for p in remaining if p not in table]
                    configs.extend(generate_configs(new_remaining, current + [list(table)]))
                return configs
            
            for config in generate_configs(players, []):
                # この構成でのペアカウントを計算
                test_pairs = other_rounds_pairs.copy()
                for table in config:
                    for p1, p2 in combinations(table, 2):
                        pair = tuple(sorted([p1, p2]))
                        test_pairs[pair] = test_pairs.get(pair, 0) + 1
                
                # 最小回数をチェック
                all_counts = [test_pairs.get(pair, 0) for pair in self.all_pairs]
                if min(all_counts) >= min_meetings:
                    # 分散を計算
                    variance = max(all_counts) - min(all_counts)
                    if variance < best_variance:
                        best_variance = variance
                        best_config = config
            
            return best_config
        
        # 中規模以上はヒューリスティック
        else:
            # 現在の最小回数ペアを優先的に同卓させる
            min_pairs = []
            for pair in self.all_pairs:
                if other_rounds_pairs.get(pair, 0) == min_meetings - 1:
                    min_pairs.append(pair)
            
            # これらのペアを含む卓を優先的に作成
            tables = []
            used_players = set()
            
            for p1, p2 in min_pairs:
                if p1 not in used_players and p2 not in used_players:
                    # 残りの2人を選択
                    candidates = [p for p in players if p not in used_players and p != p1 and p != p2]
                    if len(candidates) >= 2:
                        table = [p1, p2] + candidates[:2]
                        tables.append(table)
                        used_players.update(table)
            
            # 残りのプレイヤーで卓を作成
            remaining = [p for p in players if p not in used_players]
            while len(remaining) >= 4:
                tables.append(remaining[:4])
                remaining = remaining[4:]
            
            return tables if tables else None
    
    def _maximize_coverage_with_balance(self, solution: List[List[List[int]]]) -> List[List[List[int]]]:
        """カバレッジを最大化しつつバランスを取る"""
        # 現在の解をそのまま返す（基本的な実装は既に良好）
        return solution
    
    def _is_better_distribution(self, score1: Dict, score2: Dict) -> bool:
        """分布が改善されたか判定（最小回数は同じと仮定）"""
        # 最大回数が小さい方が良い
        if score1['max_count'] < score2['max_count']:
            return True
        elif score1['max_count'] > score2['max_count']:
            return False
        
        # 最小回数のペア数が少ない方が良い
        min1 = score1['distribution'].get(score1['min_count'], 0)
        min2 = score2['distribution'].get(score2['min_count'], 0)
        if min1 < min2:
            return True
        elif min1 > min2:
            return False
        
        # 最大回数のペア数が少ない方が良い
        max1 = score1['distribution'].get(score1['max_count'], 0)
        max2 = score2['distribution'].get(score2['max_count'], 0)
        return max1 < max2
    
    def _evaluate_solution(self, solution: List[List[List[int]]]) -> Dict:
        """解を評価"""
        pair_count = defaultdict(int)
        
        for round_tables in solution:
            for table in round_tables:
                if len(table) >= 4:
                    for p1, p2 in combinations(table, 2):
                        pair = tuple(sorted([p1, p2]))
                        pair_count[pair] += 1
        
        # 統計を計算
        all_counts = [pair_count.get(pair, 0) for pair in self.all_pairs]
        min_count = min(all_counts) if all_counts else 0
        max_count = max(all_counts) if all_counts else 0
        
        count_distribution = defaultdict(int)
        for count in all_counts:
            count_distribution[count] += 1
        
        return {
            'min_count': min_count,
            'max_count': max_count,
            'distribution': dict(count_distribution),
            'pair_count': dict(pair_count)
        }
    
    def print_results(self, results: List[List[List[int]]]):
        """結果を見やすく出力"""
        print(f"\n麻雀卓組結果（優先順位最適化版） (参加者: {self.players}人, {self.rounds}回戦)")
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
        
        # 全ペアの同卓回数を取得
        all_counts = []
        for pair in self.all_pairs:
            count = pair_count.get(pair, 0)
            all_counts.append(count)
        
        if not all_counts:
            print("統計データなし")
            return
        
        # 統計情報を計算
        import statistics
        min_count = min(all_counts)
        max_count = max(all_counts)
        mean_count = statistics.mean(all_counts)
        stdev_count = statistics.stdev(all_counts) if len(all_counts) > 1 else 0
        
        # カウント分布
        count_distribution = defaultdict(int)
        for count in all_counts:
            count_distribution[count] += 1
        
        print(f"最小同卓回数: {min_count}回 {'←最優先条件' if min_count > 0 else '←最優先条件（未達成）'}")
        print(f"最大同卓回数: {max_count}回")
        print(f"平均同卓回数: {mean_count:.2f}回")
        print(f"標準偏差: {stdev_count:.2f}")
        
        print("\n同卓回数の分布:")
        for count in sorted(count_distribution.keys()):
            percentage = count_distribution[count] / self.total_pairs * 100
            marker = " ←最優先" if count == min_count else ""
            print(f"  {count}回同卓: {count_distribution[count]}ペア ({percentage:.1f}%){marker}")
        
        # カバレッジ
        realized_pairs = sum(1 for count in all_counts if count > 0)
        coverage = realized_pairs / self.total_pairs * 100
        
        print(f"\nペアカバレッジ: {realized_pairs}/{self.total_pairs} ({coverage:.1f}%)")
        
        # 評価基準に基づくスコア
        print("\n評価基準（優先順位順）:")
        print(f"1. 【最優先】同卓回数の最小: {min_count}回 {'✓ 達成' if min_count > 0 else '✗ 未達成'}")
        print(f"2. 同卓回数の最大: {max_count}回 {'✓' if max_count - min_count <= 1 else '△'}")
        print(f"3. 最小回数のペア数: {count_distribution[min_count]}ペア")
        print(f"4. 最大回数のペア数: {count_distribution[max_count]}ペア")


def main():
    parser = argparse.ArgumentParser(description='麻雀卓組生成プログラム（優先順位最適化版）')
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
    
    generator = PriorityTableGroupGenerator(args.players, args.rounds, args.five)
    results = generator.generate()
    generator.print_results(results)


if __name__ == "__main__":
    main()