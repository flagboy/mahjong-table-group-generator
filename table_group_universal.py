#!/usr/bin/env python3
"""麻雀卓組生成プログラム（汎用最適化版）- 8~40人、3~12回戦で条件1~4を階層的に最適化"""

import argparse
from typing import List, Dict, Tuple, Set, Optional
from collections import defaultdict
from itertools import combinations
import pulp
import math
import time
import random


class UniversalTableGroupGenerator:
    def __init__(self, players: int, rounds: int, allow_five: bool = False):
        """
        Args:
            players: 参加人数（8~40人）
            rounds: 回数（3~12回戦）
            allow_five: 5人打ちを許可するか
        """
        if not (8 <= players <= 40):
            raise ValueError("参加人数は8人から40人の間で指定してください")
        if not (3 <= rounds <= 12):
            raise ValueError("回戦数は3から12の間で指定してください")
        
        self.players = players
        self.rounds = rounds
        self.allow_five = allow_five
        self.player_ids = list(range(1, players + 1))
        self.all_pairs = list(combinations(self.player_ids, 2))
        self.total_pairs = len(self.all_pairs)
        
        # 最適化戦略の決定
        self._determine_optimization_strategy()
        
        # 理論的な制約を計算
        self._calculate_theoretical_limits()
        
    def _determine_optimization_strategy(self):
        """問題サイズに応じた最適化戦略を決定"""
        # 問題の複雑度を評価
        self.problem_size = self.players * self.rounds
        
        if self.players <= 12 and self.rounds <= 6:
            self.strategy = "exact"  # 厳密解を目指す
            self.iterations = 10
            self.timeout = 60
        elif self.players <= 20 and self.rounds <= 8:
            self.strategy = "intensive"  # 集中的な最適化
            self.iterations = 5
            self.timeout = 30
        elif self.players <= 30:
            self.strategy = "balanced"  # バランス重視
            self.iterations = 3
            self.timeout = 20
        else:
            self.strategy = "fast"  # 高速処理優先
            self.iterations = 2
            self.timeout = 10
        
        print(f"最適化戦略: {self.strategy} (問題サイズ: {self.players}人×{self.rounds}回戦)")
        
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
        
        self.theoretical_best = {
            'min_meetings': self.ideal_meetings_floor,
            'max_meetings': self.ideal_meetings_ceil,
            'floor_pairs': floor_pairs,
            'ceil_pairs': ceil_pairs
        }
        
        print(f"\n理論的分析:")
        print(f"- 参加者数: {self.players}人")
        print(f"- 全ペア数: {self.total_pairs}")
        print(f"- 1ラウンドの最大ペア数: {self.max_pairs_per_round}")
        print(f"- {self.rounds}ラウンドの最大ペア総数: {self.max_total_pairs}")
        print(f"- 理論的な最小同卓回数: {self.ideal_meetings_floor}回")
        
        if self.ideal_meetings_floor == 0:
            self.min_rounds_for_full_coverage = math.ceil(self.total_pairs / self.max_pairs_per_round)
            print(f"\n注意: 全ペアを最低1回同卓させるには{self.min_rounds_for_full_coverage}回戦以上必要です")
        
    def _find_best_table_configuration(self) -> Dict:
        """5人打ちありの場合の最適な卓構成を見つける"""
        best_config = None
        max_pairs = 0
        
        for five_tables in range(self.players // 5 + 1):
            remaining = self.players - five_tables * 5
            four_tables = remaining // 4
            waiting = remaining % 4
            
            if remaining >= 0 and (waiting == 0 or (waiting >= 4 and five_tables > 0)):
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
        start_time = time.time()
        
        # 複数の手法で解を生成し、最良のものを選択
        candidates = []
        
        # 手法1: 未実現ペア優先法
        print("手法1: 未実現ペア優先法...")
        solution1 = self._generate_unrealized_first()
        score1 = self._evaluate_hierarchical(solution1)
        candidates.append((score1, solution1, "未実現ペア優先"))
        
        # 手法2: バランス重視法
        if self.strategy in ["exact", "intensive", "balanced"]:
            print("手法2: バランス重視法...")
            solution2 = self._generate_balanced()
            score2 = self._evaluate_hierarchical(solution2)
            candidates.append((score2, solution2, "バランス重視"))
        
        # 手法3: 線形計画法
        if self.strategy in ["exact", "intensive"] and self.players <= 20:
            print("手法3: 線形計画法...")
            solution3 = self._generate_lp_based()
            if solution3:
                score3 = self._evaluate_hierarchical(solution3)
                candidates.append((score3, solution3, "線形計画"))
        
        # 最良の解を選択
        candidates.sort(key=lambda x: x[0], reverse=True)
        best_score, best_solution, best_method = candidates[0]
        
        print(f"\n採用手法: {best_method}")
        print(f"条件達成度スコア: {best_score}")
        
        # 追加の最適化
        if self.strategy in ["exact", "intensive"]:
            print("\n追加最適化を実行中...")
            best_solution = self._post_optimization(best_solution)
        
        elapsed_time = time.time() - start_time
        print(f"\n最適化完了（所要時間: {elapsed_time:.2f}秒）")
        
        return best_solution
    
    def _generate_unrealized_first(self) -> List[List[List[int]]]:
        """未実現ペアを優先する手法"""
        all_rounds = []
        pair_count = defaultdict(int)
        
        # 待機ローテーションの作成
        waiting_rotation = self._create_optimized_waiting_rotation()
        
        for round_num in range(self.rounds):
            # 待機プレイヤーの決定
            if waiting_rotation and round_num < len(waiting_rotation):
                waiting_players = waiting_rotation[round_num]
                playing_players = [p for p in self.player_ids if p not in waiting_players]
            else:
                playing_players = self.player_ids
                waiting_players = []
            
            # このラウンドの卓組を最適化
            if self.players <= 16:
                # 小規模：より精密な最適化
                tables = self._optimize_round_small(playing_players, pair_count)
            else:
                # 大規模：高速ヒューリスティック
                tables = self._optimize_round_large(playing_players, pair_count)
            
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
    
    def _optimize_round_small(self, players: List[int], 
                             pair_count: Dict[Tuple[int, int], int]) -> List[List[int]]:
        """小規模問題用の精密な最適化"""
        best_tables = None
        best_score = float('-inf')
        
        # 複数の候補を生成
        for attempt in range(min(5, 2 ** len(players))):
            if attempt == 0:
                # グリーディ法
                tables = self._greedy_assignment(players, pair_count)
            else:
                # ランダム性を加えた探索
                tables = self._randomized_assignment(players, pair_count)
            
            # スコアを計算
            score = self._score_table_assignment(tables, pair_count)
            
            if score > best_score:
                best_score = score
                best_tables = tables
        
        return best_tables
    
    def _greedy_assignment(self, players: List[int], 
                          pair_count: Dict[Tuple[int, int], int]) -> List[List[int]]:
        """グリーディな卓割り当て"""
        tables = []
        remaining = players.copy()
        
        while len(remaining) >= 4:
            # 未実現ペアまたは最小回数ペアを優先
            table = self._select_best_table(remaining, pair_count)
            tables.append(table)
            for p in table:
                remaining.remove(p)
        
        return tables
    
    def _select_best_table(self, players: List[int], 
                          pair_count: Dict[Tuple[int, int], int]) -> List[int]:
        """最適な卓を選択"""
        if len(players) <= 8:
            # 全探索
            best_table = None
            best_score = float('-inf')
            
            for table in combinations(players, 4):
                score = 0
                for p1, p2 in combinations(table, 2):
                    pair = tuple(sorted([p1, p2]))
                    count = pair_count.get(pair, 0)
                    if count == 0:
                        score += 1000  # 未実現ペアを最優先
                    else:
                        score -= count ** 2  # 既存ペアにペナルティ
                
                if score > best_score:
                    best_score = score
                    best_table = list(table)
            
            return best_table
        else:
            # ヒューリスティック
            # 最も同卓回数が少ないプレイヤーから開始
            player_scores = {}
            for p in players:
                score = sum(pair_count.get(tuple(sorted([p, other])), 0) 
                           for other in players if other != p)
                player_scores[p] = score
            
            sorted_players = sorted(players, key=lambda p: player_scores[p])
            start_player = sorted_players[0]
            
            # そのプレイヤーと最も同卓していない3人を選択
            candidates = [p for p in players if p != start_player]
            candidates.sort(key=lambda p: pair_count.get(tuple(sorted([start_player, p])), 0))
            
            return [start_player] + candidates[:3]
    
    def _randomized_assignment(self, players: List[int], 
                              pair_count: Dict[Tuple[int, int], int]) -> List[List[int]]:
        """ランダム性を加えた卓割り当て"""
        tables = []
        remaining = players.copy()
        
        while len(remaining) >= 4:
            # 上位候補からランダムに選択
            candidates = []
            
            # サンプル数を制限
            sample_size = min(10, math.comb(len(remaining), 4))
            
            for _ in range(sample_size):
                table = random.sample(remaining, 4)
                score = self._score_single_table(table, pair_count)
                candidates.append((score, table))
            
            # 上位からランダムに選択
            candidates.sort(reverse=True)
            top_k = min(3, len(candidates))
            _, selected_table = candidates[random.randint(0, top_k - 1)]
            
            tables.append(selected_table)
            for p in selected_table:
                remaining.remove(p)
        
        return tables
    
    def _score_single_table(self, table: List[int], 
                           pair_count: Dict[Tuple[int, int], int]) -> float:
        """単一の卓のスコアを計算"""
        score = 0
        
        for p1, p2 in combinations(table, 2):
            pair = tuple(sorted([p1, p2]))
            count = pair_count.get(pair, 0)
            
            if count == 0:
                score += 1000  # 未実現ペア
            elif count < self.ideal_meetings_floor:
                score += 100 / (count + 1)  # 少ない回数
            else:
                score -= count ** 2  # 多い回数にペナルティ
        
        return score
    
    def _score_table_assignment(self, tables: List[List[int]], 
                               pair_count: Dict[Tuple[int, int], int]) -> float:
        """卓割り当て全体のスコアを計算"""
        total_score = 0
        
        for table in tables:
            if len(table) >= 4:
                total_score += self._score_single_table(table, pair_count)
        
        return total_score
    
    def _optimize_round_large(self, players: List[int], 
                             pair_count: Dict[Tuple[int, int], int]) -> List[List[int]]:
        """大規模問題用の高速最適化"""
        # クラスタリングベースのアプローチ
        tables = []
        remaining = players.copy()
        
        # プレイヤーを同卓回数でグループ化
        player_groups = self._cluster_players_by_meetings(remaining, pair_count)
        
        while len(remaining) >= 4:
            # 異なるグループから選択
            table = self._select_from_clusters(player_groups, remaining)
            
            if not table:
                # フォールバック
                table = remaining[:4]
            
            tables.append(table)
            for p in table:
                remaining.remove(p)
                # グループから削除
                for group in player_groups.values():
                    if p in group:
                        group.remove(p)
        
        return tables
    
    def _cluster_players_by_meetings(self, players: List[int], 
                                    pair_count: Dict[Tuple[int, int], int]) -> Dict[int, List[int]]:
        """プレイヤーを同卓回数でクラスタリング"""
        player_scores = {}
        
        for p in players:
            total_meetings = sum(pair_count.get(tuple(sorted([p, other])), 0) 
                               for other in players if other != p)
            player_scores[p] = total_meetings
        
        # 4つのグループに分割
        sorted_players = sorted(players, key=lambda p: player_scores[p])
        group_size = len(players) // 4
        
        groups = {}
        for i in range(4):
            start = i * group_size
            end = start + group_size if i < 3 else len(players)
            groups[i] = sorted_players[start:end]
        
        return groups
    
    def _select_from_clusters(self, groups: Dict[int, List[int]], 
                             remaining: List[int]) -> Optional[List[int]]:
        """各クラスタから1人ずつ選択"""
        table = []
        
        for group_id in range(4):
            candidates = [p for p in groups.get(group_id, []) if p in remaining]
            if candidates:
                selected = random.choice(candidates)
                table.append(selected)
        
        return table if len(table) == 4 else None
    
    def _generate_balanced(self) -> List[List[List[int]]]:
        """バランス重視の手法"""
        # 理想的な分配に近づけることを目指す
        all_rounds = []
        pair_count = defaultdict(int)
        
        waiting_rotation = self._create_optimized_waiting_rotation()
        
        for round_num in range(self.rounds):
            # 進捗率
            progress = (round_num + 1) / self.rounds
            target_meetings = self.ideal_meetings_float * progress
            
            # 待機プレイヤー
            if waiting_rotation and round_num < len(waiting_rotation):
                waiting_players = waiting_rotation[round_num]
                playing_players = [p for p in self.player_ids if p not in waiting_players]
            else:
                playing_players = self.player_ids
                waiting_players = []
            
            # バランスを考慮した卓組
            tables = self._balanced_assignment(playing_players, pair_count, target_meetings)
            
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
    
    def _balanced_assignment(self, players: List[int], 
                            pair_count: Dict[Tuple[int, int], int],
                            target_meetings: float) -> List[List[int]]:
        """バランスを考慮した卓割り当て"""
        tables = []
        remaining = players.copy()
        
        while len(remaining) >= 4:
            # 目標との差が大きいペアを優先
            best_table = None
            best_deviation = float('inf')
            
            # 候補を生成
            candidates = self._generate_table_candidates(remaining)
            
            for table in candidates:
                deviation = 0
                for p1, p2 in combinations(table, 2):
                    pair = tuple(sorted([p1, p2]))
                    current = pair_count.get(pair, 0)
                    deviation += abs(current + 1 - target_meetings)
                
                if deviation < best_deviation:
                    best_deviation = deviation
                    best_table = table
            
            if best_table:
                tables.append(best_table)
                for p in best_table:
                    remaining.remove(p)
            else:
                # フォールバック
                tables.append(remaining[:4])
                remaining = remaining[4:]
        
        return tables
    
    def _generate_table_candidates(self, players: List[int]) -> List[List[int]]:
        """卓の候補を生成"""
        if len(players) <= 12:
            # 全組み合わせ
            return [list(table) for table in combinations(players, 4)][:50]
        else:
            # ランダムサンプリング
            candidates = []
            for _ in range(20):
                candidates.append(random.sample(players, 4))
            return candidates
    
    def _generate_lp_based(self) -> Optional[List[List[List[int]]]]:
        """線形計画法による生成"""
        try:
            # タイムアウトを設定
            timeout = min(self.timeout, 60)
            
            # 小規模な問題のみ
            if self.players > 20:
                return None
            
            # 全ラウンドを同時に最適化（簡易版）
            return self._lp_optimization_simplified(timeout)
            
        except Exception as e:
            print(f"線形計画法でエラー: {e}")
            return None
    
    def _lp_optimization_simplified(self, timeout: int) -> List[List[List[int]]]:
        """簡易版の線形計画法最適化"""
        # 各ラウンドを独立に最適化
        all_rounds = []
        pair_count = defaultdict(int)
        
        waiting_rotation = self._create_optimized_waiting_rotation()
        
        for round_num in range(self.rounds):
            if waiting_rotation and round_num < len(waiting_rotation):
                waiting_players = waiting_rotation[round_num]
                playing_players = [p for p in self.player_ids if p not in waiting_players]
            else:
                playing_players = self.player_ids
                waiting_players = []
            
            # LPで1ラウンドを最適化
            tables = self._lp_single_round(playing_players, pair_count, timeout // self.rounds)
            
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
    
    def _lp_single_round(self, players: List[int], 
                        pair_count: Dict[Tuple[int, int], int],
                        timeout: int) -> List[List[int]]:
        """単一ラウンドのLP最適化"""
        # 可能な卓
        all_tables = list(combinations(players, 4))
        if self.allow_five and len(players) >= 5:
            all_tables.extend(list(combinations(players, 5)))
        
        # サンプリング
        if len(all_tables) > 100:
            # スコアでソート
            scored_tables = []
            for table in all_tables:
                score = self._score_single_table(list(table), pair_count)
                scored_tables.append((score, table))
            scored_tables.sort(reverse=True)
            possible_tables = [t[1] for t in scored_tables[:100]]
        else:
            possible_tables = all_tables
        
        # LP問題
        prob = pulp.LpProblem("RoundOptimization", pulp.LpMaximize)
        
        # 決定変数
        table_vars = {}
        for i, table in enumerate(possible_tables):
            table_vars[i] = pulp.LpVariable(f"table_{i}", cat='Binary')
        
        # 目的関数
        objective = 0
        for i, table in enumerate(possible_tables):
            score = self._score_single_table(list(table), pair_count)
            objective += table_vars[i] * score
        
        prob += objective
        
        # 制約：各プレイヤーは最大1つの卓
        for player in players:
            player_tables = []
            for i, table in enumerate(possible_tables):
                if player in table:
                    player_tables.append(table_vars[i])
            if player_tables:
                prob += pulp.lpSum(player_tables) <= 1
        
        # 解く
        prob.solve(pulp.PULP_CBC_CMD(msg=0, timeLimit=timeout))
        
        # 結果を抽出
        selected_tables = []
        for i, table in enumerate(possible_tables):
            if table_vars[i].varValue > 0.5:
                selected_tables.append(list(table))
        
        return selected_tables if selected_tables else self._greedy_assignment(players, pair_count)
    
    def _create_optimized_waiting_rotation(self) -> Optional[List[List[int]]]:
        """最適化された待機ローテーション"""
        waiting_count = self.optimal_table_config.get('waiting', 0)
        if waiting_count == 0:
            return None
        
        # 各プレイヤーの待機回数を均等化
        rotation = []
        wait_counts = defaultdict(int)
        
        for round_num in range(self.rounds):
            # 待機回数が少ない順に選択
            candidates = sorted(self.player_ids, key=lambda p: (wait_counts[p], p))
            waiting = candidates[:waiting_count]
            
            rotation.append(waiting)
            for p in waiting:
                wait_counts[p] += 1
        
        return rotation
    
    def _post_optimization(self, solution: List[List[List[int]]]) -> List[List[List[int]]]:
        """解の事後最適化"""
        best_solution = solution
        best_score = self._evaluate_hierarchical(solution)
        
        # 改善試行
        for iteration in range(self.iterations):
            # 条件1の改善を試みる
            improved = self._improve_minimum_meetings(best_solution)
            score = self._evaluate_hierarchical(improved)
            
            if score > best_score:
                best_solution = improved
                best_score = score
            
            # 条件2の改善を試みる
            improved = self._improve_maximum_meetings(best_solution)
            score = self._evaluate_hierarchical(improved)
            
            if score > best_score:
                best_solution = improved
                best_score = score
        
        return best_solution
    
    def _improve_minimum_meetings(self, solution: List[List[List[int]]]) -> List[List[List[int]]]:
        """最小同卓回数を改善"""
        # 0回同卓のペアを見つける
        pair_count = self._count_all_pairs(solution)
        zero_pairs = [pair for pair in self.all_pairs if pair_count.get(pair, 0) == 0]
        
        if not zero_pairs:
            return solution
        
        # ランダムに選択して改善を試みる
        target_pair = random.choice(zero_pairs)
        
        # 各ラウンドで挿入を試みる
        for round_idx in range(len(solution)):
            improved = self._try_insert_pair(solution, round_idx, target_pair)
            if improved:
                return improved
        
        return solution
    
    def _try_insert_pair(self, solution: List[List[List[int]]], 
                        round_idx: int, pair: Tuple[int, int]) -> Optional[List[List[List[int]]]]:
        """特定のペアを挿入"""
        p1, p2 = pair
        round_tables = solution[round_idx]
        
        # p1とp2の位置を確認
        p1_table_idx = None
        p2_table_idx = None
        
        for i, table in enumerate(round_tables):
            if p1 in table:
                p1_table_idx = i
            if p2 in table:
                p2_table_idx = i
        
        # 異なる卓にいる場合、交換を試みる
        if (p1_table_idx is not None and p2_table_idx is not None and 
            p1_table_idx != p2_table_idx and 
            len(round_tables[p1_table_idx]) >= 4 and 
            len(round_tables[p2_table_idx]) >= 4):
            
            # 交換候補を探す
            for swap_player in round_tables[p2_table_idx]:
                if swap_player != p2:
                    # 交換
                    new_solution = [r.copy() for r in solution]
                    new_solution[round_idx] = []
                    
                    for i, table in enumerate(round_tables):
                        if i == p1_table_idx:
                            new_table = [p if p != p1 else swap_player for p in table]
                            new_solution[round_idx].append(new_table)
                        elif i == p2_table_idx:
                            new_table = [p if p != swap_player else p1 for p in table]
                            new_solution[round_idx].append(new_table)
                        else:
                            new_solution[round_idx].append(table.copy())
                    
                    return new_solution
        
        return None
    
    def _improve_maximum_meetings(self, solution: List[List[List[int]]]) -> List[List[List[int]]]:
        """最大同卓回数を改善"""
        pair_count = self._count_all_pairs(solution)
        max_count = max(pair_count.values()) if pair_count else 0
        
        # 最大回数のペアを見つける
        max_pairs = [pair for pair, count in pair_count.items() if count == max_count]
        
        if not max_pairs or max_count <= self.ideal_meetings_ceil:
            return solution
        
        # ランダムに選択
        target_pair = random.choice(max_pairs)
        
        # 同卓を減らす
        return self._reduce_pair_frequency(solution, target_pair)
    
    def _reduce_pair_frequency(self, solution: List[List[List[int]]], 
                              pair: Tuple[int, int]) -> List[List[List[int]]]:
        """特定ペアの頻度を減らす"""
        p1, p2 = pair
        
        # 同卓しているラウンドを見つける
        together_rounds = []
        for round_idx, round_tables in enumerate(solution):
            for table in round_tables:
                if len(table) >= 4 and p1 in table and p2 in table:
                    together_rounds.append(round_idx)
                    break
        
        if len(together_rounds) <= 1:
            return solution
        
        # ランダムに1つ選んで分離を試みる
        target_round = random.choice(together_rounds)
        
        # 交換で分離
        new_solution = [r.copy() for r in solution]
        round_tables = new_solution[target_round]
        
        for i, table in enumerate(round_tables):
            if len(table) >= 4 and p1 in table and p2 in table:
                # 他の卓と交換
                for j, other_table in enumerate(round_tables):
                    if i != j and len(other_table) >= 4:
                        # p1を他の卓に移動
                        for swap_player in other_table:
                            new_table1 = [p if p != p1 else swap_player for p in table]
                            new_table2 = [p if p != swap_player else p1 for p in other_table]
                            
                            new_solution[target_round] = []
                            for k, t in enumerate(round_tables):
                                if k == i:
                                    new_solution[target_round].append(new_table1)
                                elif k == j:
                                    new_solution[target_round].append(new_table2)
                                else:
                                    new_solution[target_round].append(t.copy())
                            
                            return new_solution
        
        return solution
    
    def _count_all_pairs(self, solution: List[List[List[int]]]) -> Dict[Tuple[int, int], int]:
        """全ペアの同卓回数をカウント"""
        pair_count = defaultdict(int)
        
        for round_tables in solution:
            for table in round_tables:
                if len(table) >= 4:
                    for p1, p2 in combinations(table, 2):
                        pair = tuple(sorted([p1, p2]))
                        pair_count[pair] += 1
        
        return pair_count
    
    def _evaluate_hierarchical(self, solution: List[List[List[int]]]) -> float:
        """階層的な評価（条件1~4の優先順位を考慮）"""
        pair_count = self._count_all_pairs(solution)
        
        # 統計を計算
        all_counts = [pair_count.get(pair, 0) for pair in self.all_pairs]
        min_count = min(all_counts) if all_counts else 0
        max_count = max(all_counts) if all_counts else 0
        
        count_distribution = defaultdict(int)
        for count in all_counts:
            count_distribution[count] += 1
        
        # 階層的スコア（桁を大きく分けて優先順位を明確化）
        score = 0
        
        # 条件1: 最小同卓回数（最重要：10^9のオーダー）
        score += min_count * 1e9
        
        # 条件2: 最大同卓回数（重要：10^6のオーダー）
        score -= max_count * 1e6
        
        # 条件3: 最小回数のペア数（やや重要：10^3のオーダー）
        score -= count_distribution[min_count] * 1e3
        
        # 条件4: 最大回数のペア数（重要度低：10^0のオーダー）
        score -= count_distribution[max_count]
        
        return score
    
    def print_results(self, results: List[List[List[int]]]):
        """結果を見やすく出力"""
        print(f"\n麻雀卓組結果（汎用最適化版） (参加者: {self.players}人, {self.rounds}回戦)")
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
        
        # 評価基準の達成度
        print("\n条件達成度（階層的優先順位）:")
        
        # 条件1の評価
        if self.ideal_meetings_floor > 0:
            achievement1 = "✓ 完璧" if min_count >= self.ideal_meetings_floor else "△ 部分的"
        else:
            achievement1 = "✓ 最良" if min_count == 0 else "✗ 未達成"
        print(f"1. 【最重要】同卓回数の最小: {min_count}回 {achievement1}")
        
        # 条件2の評価
        if max_count <= self.ideal_meetings_ceil:
            achievement2 = "✓ 理想的"
        elif max_count <= self.ideal_meetings_ceil + 1:
            achievement2 = "△ 許容範囲"
        else:
            achievement2 = "✗ 要改善"
        print(f"2. 【重要】同卓回数の最大: {max_count}回 {achievement2}")
        
        # 条件3の評価
        min_pairs_ratio = count_distribution[min_count] / self.total_pairs
        if min_pairs_ratio <= 0.3:
            achievement3 = "✓ 良好"
        elif min_pairs_ratio <= 0.5:
            achievement3 = "△ 普通"
        else:
            achievement3 = "✗ 偏り大"
        print(f"3. 【やや重要】最小回数のペア数: {count_distribution[min_count]}ペア ({min_pairs_ratio*100:.1f}%) {achievement3}")
        
        # 条件4の評価
        max_pairs_ratio = count_distribution[max_count] / self.total_pairs
        if max_pairs_ratio <= 0.1:
            achievement4 = "✓ 良好"
        elif max_pairs_ratio <= 0.2:
            achievement4 = "△ 普通"
        else:
            achievement4 = "✗ 偏り大"
        print(f"4. 【優先度低】最大回数のペア数: {count_distribution[max_count]}ペア ({max_pairs_ratio*100:.1f}%) {achievement4}")
        
        # 理論値との比較
        print(f"\n理論値との比較:")
        print(f"- 理論的な最小同卓回数: {self.ideal_meetings_floor}回")
        print(f"- 理論的な最大同卓回数: {self.ideal_meetings_ceil}回")
        print(f"- 実際の平均同卓回数: {mean_count:.2f}回（理論値: {self.ideal_meetings_float:.2f}回）")


def main():
    parser = argparse.ArgumentParser(description='麻雀卓組生成プログラム（汎用最適化版）')
    parser.add_argument('players', type=int, help='参加人数（8~40人）')
    parser.add_argument('rounds', type=int, help='回数（3~12回戦）')
    parser.add_argument('--five', action='store_true', help='5人打ちを許可')
    
    args = parser.parse_args()
    
    try:
        generator = UniversalTableGroupGenerator(args.players, args.rounds, args.five)
        results = generator.generate()
        generator.print_results(results)
    except ValueError as e:
        print(f"エラー: {e}")
        return


if __name__ == "__main__":
    main()