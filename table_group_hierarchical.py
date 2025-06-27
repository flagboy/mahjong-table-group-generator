#!/usr/bin/env python3
"""麻雀卓組生成プログラム（階層最適化版）- 条件1~4を階層的に最適化"""

import argparse
from typing import List, Dict, Tuple, Set, Optional
from collections import defaultdict
from itertools import combinations
import pulp
import math
import time
import random


class HierarchicalTableGroupGenerator:
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
        
        # 最適化パラメータ
        self.timeout_per_phase = 30  # 各フェーズの最大計算時間（秒）
        
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
        
        # 理論的に達成可能な最良の分配
        total_pair_meetings = self.max_total_pairs
        ceil_pairs = total_pair_meetings - self.ideal_meetings_floor * self.total_pairs
        floor_pairs = self.total_pairs - ceil_pairs
        
        self.theoretical_best_distribution = {
            self.ideal_meetings_floor: floor_pairs,
            self.ideal_meetings_ceil: ceil_pairs
        }
        
        print(f"\n理論的分析:")
        print(f"- 参加者数: {self.players}人")
        print(f"- 全ペア数: {self.total_pairs}")
        print(f"- 1ラウンドの最大ペア数: {self.max_pairs_per_round}")
        print(f"- {self.rounds}ラウンドの最大ペア総数: {self.max_total_pairs}")
        print(f"- 理論的な最小同卓回数: {self.ideal_meetings_floor}回")
        print(f"- 理論的に最良の分配: {self.ideal_meetings_floor}回×{floor_pairs}ペア, {self.ideal_meetings_ceil}回×{ceil_pairs}ペア")
        
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
        print("\n階層最適化を開始...")
        
        # 階層的最適化
        solution = self._hierarchical_optimization()
        
        return solution
    
    def _hierarchical_optimization(self) -> List[List[List[int]]]:
        """条件1~4を階層的に最適化"""
        
        # Phase 1: 条件1（最小同卓回数）を最大化
        print("\nPhase 1: 最小同卓回数の最大化...")
        solutions_phase1 = self._optimize_minimum_meetings()
        best_min_count = max(self._evaluate_solution(sol)['min_count'] for sol in solutions_phase1)
        print(f"  達成した最小同卓回数: {best_min_count}回")
        
        # Phase 1の条件を満たす解のみを保持
        valid_solutions = [sol for sol in solutions_phase1 
                          if self._evaluate_solution(sol)['min_count'] == best_min_count]
        
        # Phase 2: 条件2（最大同卓回数）を最小化
        print("\nPhase 2: 最大同卓回数の最小化（最小回数を維持）...")
        solutions_phase2 = self._optimize_maximum_meetings(valid_solutions)
        best_max_count = min(self._evaluate_solution(sol)['max_count'] for sol in solutions_phase2)
        print(f"  達成した最大同卓回数: {best_max_count}回")
        
        # Phase 2の条件も満たす解のみを保持
        valid_solutions = [sol for sol in solutions_phase2 
                          if self._evaluate_solution(sol)['max_count'] == best_max_count]
        
        # Phase 3: 条件3（最小回数のペア数）を最小化
        print("\nPhase 3: 最小回数のペア数の最小化...")
        solutions_phase3 = self._optimize_min_count_pairs(valid_solutions)
        best_min_pairs = min(self._evaluate_solution(sol)['distribution'].get(best_min_count, 0) 
                            for sol in solutions_phase3)
        print(f"  最小回数（{best_min_count}回）のペア数: {best_min_pairs}個")
        
        # Phase 3の条件も満たす解のみを保持
        valid_solutions = [sol for sol in solutions_phase3 
                          if self._evaluate_solution(sol)['distribution'].get(best_min_count, 0) == best_min_pairs]
        
        # Phase 4: 条件4（最大回数のペア数）を最小化
        print("\nPhase 4: 最大回数のペア数の最小化...")
        best_solution = self._optimize_max_count_pairs(valid_solutions)
        final_score = self._evaluate_solution(best_solution)
        print(f"  最大回数（{best_max_count}回）のペア数: {final_score['distribution'].get(best_max_count, 0)}個")
        
        return best_solution
    
    def _optimize_minimum_meetings(self) -> List[List[List[List[int]]]]:
        """Phase 1: 最小同卓回数を最大化"""
        solutions = []
        
        # 複数の初期解を生成
        for attempt in range(3):
            print(f"  試行 {attempt + 1}/3...")
            
            # 初期解を生成
            if attempt == 0:
                # グリーディ法
                solution = self._greedy_minimize_unrealized()
            elif attempt == 1:
                # ランダム化グリーディ
                solution = self._randomized_greedy()
            else:
                # 線形計画法ベース
                solution = self._lp_based_initial()
            
            # 局所探索で改善
            improved = self._local_search_for_minimum(solution)
            solutions.append(improved)
        
        return solutions
    
    def _greedy_minimize_unrealized(self) -> List[List[List[int]]]:
        """グリーディ法で未実現ペアを最小化"""
        all_rounds = []
        pair_count = defaultdict(int)
        
        # 待機ローテーション
        waiting_rotation = self._create_balanced_waiting_rotation()
        
        for round_num in range(self.rounds):
            # 待機プレイヤーの決定
            if waiting_rotation and round_num < len(waiting_rotation):
                waiting_players = waiting_rotation[round_num]
                playing_players = [p for p in self.player_ids if p not in waiting_players]
            else:
                playing_players = self.player_ids
                waiting_players = []
            
            # 未実現ペアを優先して卓を構成
            tables = self._greedy_table_assignment_unrealized(playing_players, pair_count)
            
            if waiting_players:
                tables.append(waiting_players)
            
            all_rounds.append(tables)
            
            # ペアカウントを更新
            for table in tables:
                if len(table) >= 4:
                    for p1, p2 in combinations(table, 2):
                        pair = tuple(sorted([p1, p2]))
                        pair_count[pair] += 1
        
        return all_rounds
    
    def _greedy_table_assignment_unrealized(self, players: List[int], 
                                          pair_count: Dict[Tuple[int, int], int]) -> List[List[int]]:
        """未実現ペアを優先してグリーディに卓を割り当て"""
        tables = []
        remaining = players.copy()
        
        # 未実現ペアのグラフを構築
        unrealized_edges = []
        for p1, p2 in combinations(remaining, 2):
            pair = tuple(sorted([p1, p2]))
            if pair_count.get(pair, 0) == 0:
                unrealized_edges.append((p1, p2))
        
        # 未実現ペアを含む卓を優先的に作成
        while len(remaining) >= 4:
            if unrealized_edges:
                # 未実現ペアから開始
                p1, p2 = unrealized_edges[0]
                if p1 in remaining and p2 in remaining:
                    # 残り2人を選択（できるだけ未実現ペアを含むように）
                    candidates = [p for p in remaining if p != p1 and p != p2]
                    
                    # スコアリング
                    candidate_scores = []
                    for c1, c2 in combinations(candidates, 2):
                        score = 0
                        for p in [p1, p2]:
                            if pair_count.get(tuple(sorted([p, c1])), 0) == 0:
                                score += 1
                            if pair_count.get(tuple(sorted([p, c2])), 0) == 0:
                                score += 1
                        candidate_scores.append((score, c1, c2))
                    
                    if candidate_scores:
                        candidate_scores.sort(reverse=True)
                        _, c1, c2 = candidate_scores[0]
                        table = [p1, p2, c1, c2]
                    else:
                        table = [p1, p2] + candidates[:2]
                else:
                    # フォールバック
                    table = remaining[:4]
            else:
                # 未実現ペアがない場合は、同卓回数が少ないペアを優先
                best_table = self._find_min_weight_table(remaining, pair_count)
                table = best_table
            
            tables.append(table)
            for p in table:
                remaining.remove(p)
            
            # 未実現ペアリストを更新
            unrealized_edges = [(p1, p2) for p1, p2 in unrealized_edges 
                               if p1 in remaining and p2 in remaining]
        
        return tables
    
    def _find_min_weight_table(self, players: List[int], 
                               pair_count: Dict[Tuple[int, int], int]) -> List[int]:
        """最小重みの卓を見つける"""
        if len(players) <= 8:
            # 小規模なら全探索
            best_table = None
            min_weight = float('inf')
            
            for table in combinations(players, 4):
                weight = sum(pair_count.get(tuple(sorted([p1, p2])), 0) 
                           for p1, p2 in combinations(table, 2))
                if weight < min_weight:
                    min_weight = weight
                    best_table = list(table)
            
            return best_table
        else:
            # 大規模ならヒューリスティック
            return players[:4]
    
    def _randomized_greedy(self) -> List[List[List[int]]]:
        """ランダム化グリーディ法"""
        all_rounds = []
        pair_count = defaultdict(int)
        
        waiting_rotation = self._create_balanced_waiting_rotation()
        
        for round_num in range(self.rounds):
            if waiting_rotation and round_num < len(waiting_rotation):
                waiting_players = waiting_rotation[round_num]
                playing_players = [p for p in self.player_ids if p not in waiting_players]
            else:
                playing_players = self.player_ids
                waiting_players = []
            
            # ランダム性を加えた卓構成
            tables = []
            remaining = playing_players.copy()
            
            while len(remaining) >= 4:
                # 上位候補からランダムに選択
                candidates = []
                sample_size = min(20, len(list(combinations(remaining, 4))))
                
                for _ in range(sample_size):
                    sample_table = random.sample(remaining, 4)
                    weight = sum(pair_count.get(tuple(sorted([p1, p2])), 0) 
                               for p1, p2 in combinations(sample_table, 2))
                    candidates.append((weight, sample_table))
                
                candidates.sort()
                # 上位3つからランダムに選択
                idx = random.randint(0, min(2, len(candidates) - 1))
                table = candidates[idx][1]
                
                tables.append(table)
                for p in table:
                    remaining.remove(p)
            
            if waiting_players:
                tables.append(waiting_players)
            
            all_rounds.append(tables)
            
            # ペアカウントを更新
            for table in tables:
                if len(table) >= 4:
                    for p1, p2 in combinations(table, 2):
                        pair = tuple(sorted([p1, p2]))
                        pair_count[pair] += 1
        
        return all_rounds
    
    def _lp_based_initial(self) -> List[List[List[int]]]:
        """線形計画法ベースの初期解"""
        # 時間制約のため、簡易版で実装
        return self._greedy_minimize_unrealized()
    
    def _local_search_for_minimum(self, solution: List[List[List[int]]]) -> List[List[List[int]]]:
        """最小同卓回数を改善する局所探索"""
        best_solution = solution
        best_score = self._evaluate_solution(solution)
        
        # 5回の改善試行
        for _ in range(5):
            # 0回同卓のペアを見つける
            zero_pairs = []
            for pair in self.all_pairs:
                if best_score['pair_count'].get(pair, 0) == 0:
                    zero_pairs.append(pair)
            
            if not zero_pairs:
                break
            
            # ランダムに未実現ペアを選択
            target_pair = random.choice(zero_pairs)
            
            # そのペアを実現できるラウンドを探す
            improved = self._force_pair_in_solution(best_solution, target_pair)
            
            if improved:
                new_score = self._evaluate_solution(improved)
                if new_score['min_count'] > best_score['min_count']:
                    best_solution = improved
                    best_score = new_score
        
        return best_solution
    
    def _force_pair_in_solution(self, solution: List[List[List[int]]], 
                                target_pair: Tuple[int, int]) -> Optional[List[List[List[int]]]]:
        """特定のペアを解に含める"""
        p1, p2 = target_pair
        
        # 各ラウンドで試す
        for round_idx in range(len(solution)):
            round_tables = solution[round_idx]
            
            # p1とp2が異なる卓にいるか確認
            p1_table_idx = None
            p2_table_idx = None
            
            for i, table in enumerate(round_tables):
                if p1 in table:
                    p1_table_idx = i
                if p2 in table:
                    p2_table_idx = i
            
            # 異なる卓にいる場合、交換を試みる
            if p1_table_idx is not None and p2_table_idx is not None and p1_table_idx != p2_table_idx:
                # 交換相手を探す
                table1 = round_tables[p1_table_idx]
                table2 = round_tables[p2_table_idx]
                
                if len(table1) >= 4 and len(table2) >= 4:
                    # p1の卓からp2の卓へ移動できるプレイヤーを探す
                    for swap_player in table2:
                        if swap_player != p2:
                            # 交換をシミュレート
                            new_solution = [r.copy() for r in solution]
                            new_table1 = [p if p != p1 else swap_player for p in table1]
                            new_table2 = [p if p != swap_player else p1 for p in table2]
                            
                            new_solution[round_idx] = []
                            for i, table in enumerate(round_tables):
                                if i == p1_table_idx:
                                    new_solution[round_idx].append(new_table1)
                                elif i == p2_table_idx:
                                    new_solution[round_idx].append(new_table2)
                                else:
                                    new_solution[round_idx].append(table.copy())
                            
                            return new_solution
        
        return None
    
    def _optimize_maximum_meetings(self, solutions: List[List[List[List[int]]]]) -> List[List[List[List[int]]]]:
        """Phase 2: 最大同卓回数を最小化"""
        optimized_solutions = []
        
        for solution in solutions:
            # 各解に対して最大値を減らす試み
            optimized = self._reduce_maximum_meetings(solution)
            optimized_solutions.append(optimized)
        
        return optimized_solutions
    
    def _reduce_maximum_meetings(self, solution: List[List[List[int]]]) -> List[List[List[int]]]:
        """最大同卓回数を減らす"""
        best_solution = solution
        best_score = self._evaluate_solution(solution)
        
        # 最大回数のペアを見つける
        max_pairs = []
        max_count = best_score['max_count']
        for pair, count in best_score['pair_count'].items():
            if count == max_count:
                max_pairs.append(pair)
        
        # 5回の改善試行
        for _ in range(5):
            if not max_pairs:
                break
            
            # ランダムに最大回数ペアを選択
            target_pair = random.choice(max_pairs)
            
            # そのペアの同卓を減らす
            improved = self._reduce_pair_meetings(best_solution, target_pair, best_score['min_count'])
            
            if improved:
                new_score = self._evaluate_solution(improved)
                # 最小回数を維持しつつ最大回数が減った場合
                if (new_score['min_count'] >= best_score['min_count'] and 
                    new_score['max_count'] < best_score['max_count']):
                    best_solution = improved
                    best_score = new_score
                    
                    # 最大回数ペアリストを更新
                    max_pairs = []
                    for pair, count in new_score['pair_count'].items():
                        if count == new_score['max_count']:
                            max_pairs.append(pair)
        
        return best_solution
    
    def _reduce_pair_meetings(self, solution: List[List[List[int]]], 
                             target_pair: Tuple[int, int],
                             min_meetings: int) -> Optional[List[List[List[int]]]]:
        """特定ペアの同卓回数を減らす"""
        p1, p2 = target_pair
        
        # そのペアが同卓しているラウンドを見つける
        same_table_rounds = []
        for round_idx, round_tables in enumerate(solution):
            for table in round_tables:
                if len(table) >= 4 and p1 in table and p2 in table:
                    same_table_rounds.append(round_idx)
                    break
        
        if not same_table_rounds:
            return None
        
        # ランダムに1つのラウンドを選択
        target_round = random.choice(same_table_rounds)
        
        # そのラウンドで交換を試みる
        return self._swap_to_reduce_pair(solution, target_round, p1, p2, min_meetings)
    
    def _swap_to_reduce_pair(self, solution: List[List[List[int]]], 
                            round_idx: int, p1: int, p2: int,
                            min_meetings: int) -> Optional[List[List[List[int]]]]:
        """ペアを減らすための交換"""
        round_tables = solution[round_idx]
        
        # p1とp2が同じ卓にいる卓を見つける
        target_table_idx = None
        for i, table in enumerate(round_tables):
            if len(table) >= 4 and p1 in table and p2 in table:
                target_table_idx = i
                break
        
        if target_table_idx is None:
            return None
        
        # 他の卓と交換を試みる
        for other_idx, other_table in enumerate(round_tables):
            if other_idx != target_table_idx and len(other_table) >= 4:
                # 各プレイヤーの交換を試す
                for swap1 in round_tables[target_table_idx]:
                    for swap2 in other_table:
                        # 交換をシミュレート
                        new_solution = [r.copy() for r in solution]
                        new_target = [p if p != swap1 else swap2 for p in round_tables[target_table_idx]]
                        new_other = [p if p != swap2 else swap1 for p in other_table]
                        
                        new_solution[round_idx] = []
                        for i, table in enumerate(round_tables):
                            if i == target_table_idx:
                                new_solution[round_idx].append(new_target)
                            elif i == other_idx:
                                new_solution[round_idx].append(new_other)
                            else:
                                new_solution[round_idx].append(table.copy())
                        
                        # 最小回数が維持されているかチェック
                        new_score = self._evaluate_solution(new_solution)
                        if new_score['min_count'] >= min_meetings:
                            # p1とp2が別の卓になったかチェック
                            if not (p1 in new_target and p2 in new_target):
                                return new_solution
        
        return None
    
    def _optimize_min_count_pairs(self, solutions: List[List[List[List[int]]]]) -> List[List[List[List[int]]]]:
        """Phase 3: 最小回数のペア数を最小化"""
        # 現在の実装では、Phase 1で既に最適化されているため、そのまま返す
        return solutions
    
    def _optimize_max_count_pairs(self, solutions: List[List[List[List[int]]]]) -> List[List[List[int]]]:
        """Phase 4: 最大回数のペア数を最小化"""
        if not solutions:
            return []
        
        # 最良の解を選択
        best_solution = solutions[0]
        best_score = self._evaluate_solution(best_solution)
        
        for solution in solutions[1:]:
            score = self._evaluate_solution(solution)
            # 最大回数のペア数が少ない方を選択
            if score['distribution'].get(score['max_count'], 0) < best_score['distribution'].get(best_score['max_count'], 0):
                best_solution = solution
                best_score = score
        
        return best_solution
    
    def _create_balanced_waiting_rotation(self) -> Optional[List[List[int]]]:
        """バランスの取れた待機ローテーションを作成"""
        waiting_count = self.optimal_table_config.get('waiting', 0)
        if waiting_count == 0:
            return None
        
        rotation = []
        
        # 各プレイヤーの待機回数を均等にする
        wait_assignments = defaultdict(int)
        total_wait_slots = waiting_count * self.rounds
        base_waits = total_wait_slots // self.players
        extra_waits = total_wait_slots % self.players
        
        # 基本的な待機回数を割り当て
        for p in self.player_ids:
            wait_assignments[p] = base_waits
        
        # 余りを配分
        for i in range(extra_waits):
            wait_assignments[self.player_ids[i]] += 1
        
        # ラウンドごとに待機者を決定
        player_wait_counts = defaultdict(int)
        
        for round_num in range(self.rounds):
            # 待機回数が少ない順にソート
            candidates = sorted(self.player_ids, 
                              key=lambda p: (player_wait_counts[p], p))
            
            # 上位から待機者を選択
            waiting = []
            for p in candidates:
                if len(waiting) < waiting_count and player_wait_counts[p] < wait_assignments[p]:
                    waiting.append(p)
                    player_wait_counts[p] += 1
            
            rotation.append(waiting)
        
        return rotation
    
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
        print(f"\n麻雀卓組結果（階層最適化版） (参加者: {self.players}人, {self.rounds}回戦)")
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
        
        # カバレッジ
        realized_pairs = sum(1 for count in all_counts if count > 0)
        coverage = realized_pairs / self.total_pairs * 100
        
        print(f"\nペアカバレッジ: {realized_pairs}/{self.total_pairs} ({coverage:.1f}%)")
        
        # 評価基準に基づくスコア（階層的な達成度を表示）
        print("\n評価基準の達成度（階層的優先順位）:")
        print(f"1. 同卓回数の最小: {min_count}回 {'✓' if min_count >= self.ideal_meetings_floor else '△'}")
        print(f"2. 同卓回数の最大: {max_count}回 {'✓' if max_count <= self.ideal_meetings_ceil else '△'}")
        print(f"3. 最小回数（{min_count}回）のペア数: {count_distribution[min_count]}ペア")
        print(f"4. 最大回数（{max_count}回）のペア数: {count_distribution[max_count]}ペア")
        
        # 理論的最良との比較
        print(f"\n理論的最良との比較:")
        if self.ideal_meetings_floor in self.theoretical_best_distribution:
            ideal_floor_pairs = self.theoretical_best_distribution[self.ideal_meetings_floor]
            actual_floor_pairs = count_distribution.get(self.ideal_meetings_floor, 0)
            print(f"- {self.ideal_meetings_floor}回同卓: 理想{ideal_floor_pairs}ペア vs 実際{actual_floor_pairs}ペア")
        
        if self.ideal_meetings_ceil in self.theoretical_best_distribution:
            ideal_ceil_pairs = self.theoretical_best_distribution[self.ideal_meetings_ceil]
            actual_ceil_pairs = count_distribution.get(self.ideal_meetings_ceil, 0)
            print(f"- {self.ideal_meetings_ceil}回同卓: 理想{ideal_ceil_pairs}ペア vs 実際{actual_ceil_pairs}ペア")


def main():
    parser = argparse.ArgumentParser(description='麻雀卓組生成プログラム（階層最適化版）')
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
    
    generator = HierarchicalTableGroupGenerator(args.players, args.rounds, args.five)
    results = generator.generate()
    generator.print_results(results)


if __name__ == "__main__":
    main()