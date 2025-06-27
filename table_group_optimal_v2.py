#!/usr/bin/env python3
"""麻雀卓組生成プログラム（最適化版v2）- 12人6回戦のような標準的なケースで最適解を保証"""

import argparse
from typing import List, Dict, Tuple, Set, Optional
from collections import defaultdict
from itertools import combinations
import pulp
import random
import time


class OptimalTableGroupGeneratorV2:
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
        
        # 理想的な分布を計算
        total_pair_meetings = self.max_total_pairs
        ceil_pairs = total_pair_meetings - self.ideal_meetings_floor * self.total_pairs
        floor_pairs = self.total_pairs - ceil_pairs
        
        print(f"\n理論的分析:")
        print(f"- 参加者数: {self.players}人")
        print(f"- 全ペア数: {self.total_pairs}")
        print(f"- 1ラウンドの最大ペア数: {self.max_pairs_per_round}")
        print(f"- {self.rounds}ラウンドの最大ペア総数: {self.max_total_pairs}")
        print(f"- 理想的な平均同卓回数: {self.ideal_meetings_float:.2f}回")
        print(f"- 理想的な分布: {self.ideal_meetings_floor}回×{floor_pairs}ペア, {self.ideal_meetings_ceil}回×{ceil_pairs}ペア")
    
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
        print("\n最適化を開始...")
        
        # 小規模な場合は線形計画法で最適解を求める
        if self.players <= 12 and self.rounds <= 8:
            solution = self._solve_small_scale_optimal()
            if solution:
                return solution
        
        # 中規模以上の場合は改良版ヒューリスティック
        return self._solve_with_improved_heuristic()
    
    def _solve_small_scale_optimal(self) -> Optional[List[List[List[int]]]]:
        """小規模問題を線形計画法で解く"""
        print("線形計画法による最適化を実行中...")
        
        # 待機プレイヤーの設定
        waiting_count = self.optimal_table_config.get('waiting', 0)
        
        # 各ラウンドでプレイするプレイヤーの組み合わせを生成
        if waiting_count > 0:
            # 各プレイヤーができるだけ均等に待機するパターンを生成
            playing_patterns = self._generate_fair_playing_patterns(waiting_count)
        else:
            # 全員がプレイ
            playing_patterns = [[self.player_ids] for _ in range(self.rounds)]
        
        # 最良の解を探す
        best_solution = None
        best_score = None
        
        for pattern_idx, pattern in enumerate(playing_patterns[:10]):  # 最大10パターンまで試す
            print(f"\rパターン {pattern_idx + 1} を試行中...", end='', flush=True)
            
            # このパターンで線形計画法を実行
            solution = self._solve_pattern_with_lp(pattern)
            if solution:
                score = self._evaluate_solution(solution)
                if best_score is None or self._is_better_score(score, best_score):
                    best_solution = solution
                    best_score = score
                    
                    # 理想的な分布に達したら終了
                    if (score['min_count'] == self.ideal_meetings_floor and 
                        score['max_count'] == self.ideal_meetings_ceil):
                        print(f"\n理想的な解を発見！")
                        break
        
        print()
        return best_solution
    
    def _generate_fair_playing_patterns(self, waiting_count: int) -> List[List[List[int]]]:
        """公平な待機パターンを生成"""
        patterns = []
        
        # 各プレイヤーの待機回数を計算
        total_wait_slots = waiting_count * self.rounds
        base_waits = total_wait_slots // self.players
        extra_waits = total_wait_slots % self.players
        
        # 基本パターン：順番に待機
        for start_idx in range(min(10, self.players)):  # 最大10パターン
            pattern = []
            for round_num in range(self.rounds):
                waiting_players = []
                for i in range(waiting_count):
                    player_idx = (start_idx + round_num * waiting_count + i) % self.players
                    waiting_players.append(self.player_ids[player_idx])
                
                playing_players = [p for p in self.player_ids if p not in waiting_players]
                pattern.append(playing_players)
            
            patterns.append(pattern)
        
        return patterns
    
    def _solve_pattern_with_lp(self, playing_pattern: List[List[int]]) -> Optional[List[List[List[int]]]]:
        """特定の待機パターンで線形計画法を解く"""
        prob = pulp.LpProblem("TableGroupOptimal", pulp.LpMinimize)
        
        # 各ラウンドの可能な卓を生成
        round_possible_tables = []
        for round_players in playing_pattern:
            if self.allow_five and len(round_players) % 4 == 1:
                # 5人打ちありで、プレイヤー数が4で割って1余る場合
                tables = []
                # 5人卓を1つ含む組み合わせ
                for five_table in combinations(round_players, 5):
                    remaining = [p for p in round_players if p not in five_table]
                    # 残りを4人卓に分割
                    four_tables = self._generate_four_tables(remaining)
                    for ft_combo in four_tables:
                        tables.append([five_table] + ft_combo)
            else:
                # 4人卓のみ
                tables = self._generate_four_tables(round_players)
            
            round_possible_tables.append(tables)
        
        # 組み合わせが多すぎる場合はサンプリング
        for r_idx, tables in enumerate(round_possible_tables):
            if len(tables) > 100:
                round_possible_tables[r_idx] = random.sample(tables, 100)
        
        # 決定変数
        table_vars = {}
        for r in range(self.rounds):
            for t_idx, table_combo in enumerate(round_possible_tables[r]):
                var_name = f"r{r}_t{t_idx}"
                table_vars[(r, t_idx)] = pulp.LpVariable(var_name, cat='Binary')
        
        # 制約：各ラウンドで1つの卓組み合わせを選択
        for r in range(self.rounds):
            round_vars = [table_vars[(r, t_idx)] for t_idx in range(len(round_possible_tables[r]))]
            prob += pulp.lpSum(round_vars) == 1
        
        # ペア回数を計算
        pair_counts = {}
        for pair in self.all_pairs:
            pair_count_expr = []
            
            for r in range(self.rounds):
                for t_idx, table_combo in enumerate(round_possible_tables[r]):
                    # この卓組み合わせでこのペアが同卓するか
                    pair_in_combo = False
                    for table in table_combo:
                        if pair[0] in table and pair[1] in table:
                            pair_in_combo = True
                            break
                    
                    if pair_in_combo:
                        pair_count_expr.append(table_vars[(r, t_idx)])
            
            pair_counts[pair] = pulp.lpSum(pair_count_expr)
        
        # 最小値と最大値の変数
        min_meetings = pulp.LpVariable("min_meetings", lowBound=0, cat='Integer')
        max_meetings = pulp.LpVariable("max_meetings", lowBound=0, cat='Integer')
        
        # 制約：最小値と最大値
        for pair in self.all_pairs:
            prob += pair_counts[pair] >= min_meetings
            prob += pair_counts[pair] <= max_meetings
        
        # 目的関数：階層的最適化
        # 1. 最小値を最大化（係数: -1000000）
        # 2. 最大値を最小化（係数: 1000）
        # 3. 理想値との差を最小化（係数: 1）
        objective = -1000000 * min_meetings + 1000 * max_meetings
        
        # 理想値との差
        for pair in self.all_pairs:
            # 理想値に近いほど良い
            if self.ideal_meetings_floor == self.ideal_meetings_ceil:
                # 理想値が整数の場合
                diff = pair_counts[pair] - self.ideal_meetings_floor
                objective += diff * diff  # 二乗誤差
            else:
                # 理想値が小数の場合、floor値またはceil値になるべき
                # floor値とceil値以外にペナルティ
                is_floor = pulp.LpVariable(f"is_floor_{pair[0]}_{pair[1]}", cat='Binary')
                is_ceil = pulp.LpVariable(f"is_ceil_{pair[0]}_{pair[1]}", cat='Binary')
                
                prob += pair_counts[pair] >= self.ideal_meetings_floor * is_floor
                prob += pair_counts[pair] <= self.ideal_meetings_floor + 1000 * (1 - is_floor)
                prob += pair_counts[pair] >= self.ideal_meetings_ceil * is_ceil
                prob += pair_counts[pair] <= self.ideal_meetings_ceil + 1000 * (1 - is_ceil)
                prob += is_floor + is_ceil >= 1
                
                # floorとceilの比率を理想に近づける
                objective += (1 - is_floor - is_ceil) * 100
        
        prob += objective
        
        # 解く
        solver = pulp.PULP_CBC_CMD(msg=0, timeLimit=30)
        prob.solve(solver)
        
        if prob.status != pulp.LpStatusOptimal:
            return None
        
        # 解を抽出
        solution = []
        for r in range(self.rounds):
            round_tables = []
            
            # 選択された卓組み合わせを見つける
            for t_idx, table_combo in enumerate(round_possible_tables[r]):
                if table_vars[(r, t_idx)].varValue > 0.5:
                    for table in table_combo:
                        round_tables.append(list(table))
                    break
            
            # 待機プレイヤーを追加
            playing_players = playing_pattern[r]
            waiting_players = [p for p in self.player_ids if p not in playing_players]
            if waiting_players:
                round_tables.append(waiting_players)
            
            solution.append(round_tables)
        
        return solution
    
    def _generate_four_tables(self, players: List[int]) -> List[List[Tuple[int, ...]]]:
        """4人卓の全組み合わせを生成（再帰的）"""
        if len(players) < 4:
            return [[]]
        
        if len(players) == 4:
            return [[(tuple(players))]]
        
        # メモ化のため、小規模な場合のみ全探索
        if len(players) > 12:
            # 大規模な場合はランダムサンプリング
            results = []
            for _ in range(min(50, len(list(combinations(players, 4))))):
                remaining = players.copy()
                tables = []
                while len(remaining) >= 4:
                    table = random.sample(remaining, 4)
                    tables.append(tuple(sorted(table)))
                    for p in table:
                        remaining.remove(p)
                if remaining == []:
                    results.append(tables)
            return results
        
        # 全探索
        results = []
        
        # 最初のプレイヤーを含む全ての4人組
        first_player = players[0]
        for other_three in combinations(players[1:], 3):
            first_table = tuple(sorted([first_player] + list(other_three)))
            remaining = [p for p in players if p not in first_table]
            
            # 残りのプレイヤーで再帰的に卓を作成
            sub_results = self._generate_four_tables(remaining)
            for sub_tables in sub_results:
                results.append([first_table] + sub_tables)
        
        return results
    
    def _solve_with_improved_heuristic(self) -> List[List[List[int]]]:
        """改良版ヒューリスティックで解く"""
        print("改良版ヒューリスティックを実行中...")
        
        all_rounds = []
        pair_count = defaultdict(int)
        
        # 待機ローテーション
        waiting_rotation = self._create_balanced_waiting_rotation()
        
        for round_num in range(self.rounds):
            print(f"\rラウンド {round_num + 1}/{self.rounds} を生成中...", end='', flush=True)
            
            # 待機プレイヤーの決定
            if waiting_rotation and round_num < len(waiting_rotation):
                waiting_players = waiting_rotation[round_num]
                playing_players = [p for p in self.player_ids if p not in waiting_players]
            else:
                playing_players = self.player_ids
                waiting_players = []
            
            # 最適な卓組を探す
            best_tables = self._find_optimal_tables(playing_players, pair_count)
            
            if waiting_players:
                best_tables.append(waiting_players)
            
            all_rounds.append(best_tables)
            
            # ペアカウントを更新
            for table in best_tables:
                if len(table) >= 4:
                    for p1, p2 in combinations(table, 2):
                        pair = tuple(sorted([p1, p2]))
                        pair_count[pair] += 1
        
        print()
        
        # 局所最適化
        print("局所最適化を実行中...")
        improved_solution = self._local_optimization(all_rounds)
        
        return improved_solution
    
    def _find_optimal_tables(self, players: List[int], 
                            current_pair_count: Dict[Tuple[int, int], int]) -> List[List[int]]:
        """現在のペアカウントを考慮して最適な卓組を見つける"""
        # 現在のラウンド番号を推定
        current_round = sum(current_pair_count.values()) // self.max_pairs_per_round + 1
        
        # 各ペアの理想的な回数（現時点）
        ideal_count_now = self.ideal_meetings_float * current_round / self.rounds
        
        # ペアの優先度を計算
        pair_priorities = {}
        for p1, p2 in combinations(players, 2):
            pair = tuple(sorted([p1, p2]))
            current_count = current_pair_count.get(pair, 0)
            
            # 理想値との差
            deficit = ideal_count_now - current_count
            
            # 優先度（赤字が大きいほど高い）
            pair_priorities[pair] = deficit
        
        # 優先度の高いペアから卓を構成
        tables = []
        remaining_players = set(players)
        used_pairs = set()
        
        while len(remaining_players) >= 4:
            # 最も優先度の高い未使用ペアを選択
            best_pair = None
            best_priority = -float('inf')
            
            for pair, priority in pair_priorities.items():
                if (pair not in used_pairs and 
                    pair[0] in remaining_players and 
                    pair[1] in remaining_players):
                    if priority > best_priority:
                        best_priority = priority
                        best_pair = pair
            
            if best_pair is None:
                # 残りをランダムに
                table = list(remaining_players)[:4]
            else:
                # このペアを含む最適な4人を選択
                p1, p2 = best_pair
                candidates = [p for p in remaining_players if p != p1 and p != p2]
                
                # 残り2人を選択（優先度の高いペアを作れる人を選ぶ）
                best_candidates = None
                best_score = -float('inf')
                
                for c1, c2 in combinations(candidates, 2):
                    score = 0
                    for pair in [(p1, c1), (p1, c2), (p2, c1), (p2, c2), (c1, c2)]:
                        sorted_pair = tuple(sorted(pair))
                        if sorted_pair in pair_priorities:
                            score += pair_priorities[sorted_pair]
                    
                    if score > best_score:
                        best_score = score
                        best_candidates = [c1, c2]
                
                table = [p1, p2] + best_candidates
            
            tables.append(table)
            
            # 使用したプレイヤーとペアを記録
            for p in table:
                remaining_players.remove(p)
            
            for p1, p2 in combinations(table, 2):
                used_pairs.add(tuple(sorted([p1, p2])))
        
        return tables
    
    def _create_balanced_waiting_rotation(self) -> Optional[List[List[int]]]:
        """バランスの取れた待機ローテーションを作成"""
        waiting_count = self.optimal_table_config.get('waiting', 0)
        if waiting_count == 0:
            return None
        
        rotation = []
        
        # 各プレイヤーができるだけ均等に待機
        for round_num in range(self.rounds):
            waiting = []
            for i in range(waiting_count):
                player_idx = (round_num * waiting_count + i) % self.players
                waiting.append(self.player_ids[player_idx])
            rotation.append(waiting)
        
        return rotation
    
    def _local_optimization(self, solution: List[List[List[int]]]) -> List[List[List[int]]]:
        """局所最適化で解を改善"""
        best_solution = [round_tables[:] for round_tables in solution]
        best_score = self._evaluate_solution(best_solution)
        
        # 改善が見られなくなるまで繰り返す
        improved = True
        iteration = 0
        
        while improved and iteration < 10:
            improved = False
            iteration += 1
            
            # 各ラウンドペアで交換を試みる
            for r1 in range(self.rounds):
                for r2 in range(r1 + 1, self.rounds):
                    # 各卓ペアで交換
                    tables1 = [t for t in best_solution[r1] if len(t) >= 4]
                    tables2 = [t for t in best_solution[r2] if len(t) >= 4]
                    
                    for t1_idx, table1 in enumerate(tables1):
                        for t2_idx, table2 in enumerate(tables2):
                            # 各プレイヤーペアで交換を試す
                            for p1 in table1:
                                for p2 in table2:
                                    if p1 != p2:
                                        # 交換をシミュレート
                                        new_solution = [round_tables[:] for round_tables in best_solution]
                                        
                                        new_table1 = [p if p != p1 else p2 for p in table1]
                                        new_table2 = [p if p != p2 else p1 for p in table2]
                                        
                                        # 元の卓を見つけて更新
                                        for i, t in enumerate(new_solution[r1]):
                                            if len(t) >= 4 and sorted(t) == sorted(table1):
                                                new_solution[r1][i] = new_table1
                                                break
                                        
                                        for i, t in enumerate(new_solution[r2]):
                                            if len(t) >= 4 and sorted(t) == sorted(table2):
                                                new_solution[r2][i] = new_table2
                                                break
                                        
                                        # 評価
                                        new_score = self._evaluate_solution(new_solution)
                                        if self._is_better_score(new_score, best_score):
                                            best_solution = new_solution
                                            best_score = new_score
                                            improved = True
        
        return best_solution
    
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
    
    def _is_better_score(self, score1: Dict, score2: Dict) -> bool:
        """score1がscore2より良いか判定"""
        # 優先順位1: 最小同卓回数が大きい
        if score1['min_count'] > score2['min_count']:
            return True
        elif score1['min_count'] < score2['min_count']:
            return False
        
        # 優先順位2: 最大同卓回数が小さい
        if score1['max_count'] < score2['max_count']:
            return True
        elif score1['max_count'] > score2['max_count']:
            return False
        
        # 優先順位3: 最小回数のペア数が少ない
        min_pairs1 = score1['distribution'].get(score1['min_count'], 0)
        min_pairs2 = score2['distribution'].get(score2['min_count'], 0)
        if min_pairs1 < min_pairs2:
            return True
        elif min_pairs1 > min_pairs2:
            return False
        
        # 優先順位4: 最大回数のペア数が少ない
        max_pairs1 = score1['distribution'].get(score1['max_count'], 0)
        max_pairs2 = score2['distribution'].get(score2['max_count'], 0)
        return max_pairs1 < max_pairs2
    
    def print_results(self, results: List[List[List[int]]]):
        """結果を見やすく出力"""
        print(f"\n麻雀卓組結果（最適化版v2） (参加者: {self.players}人, {self.rounds}回戦)")
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
        
        print(f"最小同卓回数: {min_count}回")
        print(f"最大同卓回数: {max_count}回")
        print(f"平均同卓回数: {mean_count:.2f}回")
        print(f"標準偏差: {stdev_count:.2f}")
        
        print("\n同卓回数の分布:")
        for count in sorted(count_distribution.keys()):
            percentage = count_distribution[count] / self.total_pairs * 100
            print(f"  {count}回同卓: {count_distribution[count]}ペア ({percentage:.1f}%)")
        
        # 理想的な分布との比較
        total_pair_meetings = self.max_pairs_per_round * self.rounds
        ideal_ceil_pairs = total_pair_meetings - self.ideal_meetings_floor * self.total_pairs
        ideal_floor_pairs = self.total_pairs - ideal_ceil_pairs
        
        print(f"\n理想的な分布との比較:")
        print(f"  理想: {self.ideal_meetings_floor}回×{ideal_floor_pairs}ペア, {self.ideal_meetings_ceil}回×{ideal_ceil_pairs}ペア")
        print(f"  実際: ", end="")
        for count in sorted(count_distribution.keys()):
            print(f"{count}回×{count_distribution[count]}ペア ", end="")
        print()
        
        # カバレッジ
        realized_pairs = sum(1 for count in all_counts if count > 0)
        coverage = realized_pairs / self.total_pairs * 100
        
        print(f"\nペアカバレッジ: {realized_pairs}/{self.total_pairs} ({coverage:.1f}%)")


def main():
    parser = argparse.ArgumentParser(description='麻雀卓組生成プログラム（最適化版v2）')
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
    
    generator = OptimalTableGroupGeneratorV2(args.players, args.rounds, args.five)
    results = generator.generate()
    generator.print_results(results)


if __name__ == "__main__":
    main()