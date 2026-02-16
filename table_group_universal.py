#!/usr/bin/env python3
"""麻雀卓組生成プログラム（汎用最適化版）- 8~40人、3~12回戦でSA+LNS+複数初期解による大域的最適化"""

import argparse
from typing import List, Dict, Tuple, Set, Optional
from collections import defaultdict
from itertools import combinations
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

        # 理論的な制約を計算
        self._calculate_theoretical_limits()

        # SA用の時間予算を設定
        self._set_time_budget()

    def _calculate_theoretical_limits(self):
        """理論的な制約と目標を計算"""
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

        self.ideal_meetings_float = self.max_total_pairs / self.total_pairs
        self.ideal_meetings_floor = int(self.ideal_meetings_float)
        # floorとceilが同じ場合（割り切れる場合）はceil = floor
        if self.max_total_pairs == self.ideal_meetings_floor * self.total_pairs:
            self.ideal_meetings_ceil = self.ideal_meetings_floor
        else:
            self.ideal_meetings_ceil = self.ideal_meetings_floor + 1

        self.theoretical_best = {
            'min_meetings': self.ideal_meetings_floor,
            'max_meetings': self.ideal_meetings_ceil,
        }

    def _set_time_budget(self):
        """問題サイズに応じたSA時間予算を設定"""
        if self.players <= 12:
            self.sa_time_limit = 2.0
        elif self.players <= 20:
            self.sa_time_limit = 3.0
        elif self.players <= 32:
            self.sa_time_limit = 4.0
        else:
            self.sa_time_limit = 5.0

    def _find_best_table_configuration(self) -> Dict:
        """5人打ちありの場合の最適な卓構成を見つける"""
        best_config = None
        max_pairs = 0

        for five_tables in range(self.players // 5 + 1):
            remaining = self.players - five_tables * 5
            four_tables = remaining // 4
            waiting = remaining % 4

            if remaining >= 0 and (five_tables + four_tables) > 0:
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

    # =========================================================================
    # メインの生成ロジック
    # =========================================================================

    def generate(self) -> List[List[List[int]]]:
        """全ラウンドの卓組を生成"""
        floor_val = self.ideal_meetings_floor
        ceil_val = self.ideal_meetings_ceil

        # 複数の初期解を生成
        candidates = []
        for gen_func in [self._generate_unrealized_first,
                         self._generate_balanced,
                         self._generate_random,
                         self._generate_random,
                         self._generate_random]:
            sol = gen_func()
            cost = self._compute_cost_from_solution(sol)
            candidates.append((cost, sol))
            if cost == 0:
                return sol

        # 各初期解にLNSを適用（高速）
        lns_time = 0.5
        for i, (cost, sol) in enumerate(candidates):
            if cost == 0:
                continue
            sa_tables = []
            sa_waiting = []
            for round_tables in sol:
                game = [t[:] for t in round_tables if len(t) >= 4]
                wait = next((t[:] for t in round_tables if len(t) < 4), None)
                sa_tables.append(game)
                sa_waiting.append(wait)
            sa_tables, new_cost = self._large_neighborhood_search(
                sa_tables, sa_waiting, floor_val, ceil_val, lns_time
            )
            candidates[i] = (new_cost, self._reconstruct_solution(sa_tables, sa_waiting))
            if new_cost == 0:
                return candidates[i][1]

        # 最良のLNS結果にSAを適用
        candidates.sort(key=lambda x: x[0])
        best_cost, best_solution = candidates[0]

        if best_cost > 0:
            best_solution = self._simulated_annealing(
                best_solution, self.sa_time_limit
            )
            best_cost = self._compute_cost_from_solution(best_solution)

        # SA後にもう一度LNS
        if best_cost > 0:
            sa_tables = []
            sa_waiting = []
            for round_tables in best_solution:
                game = [t[:] for t in round_tables if len(t) >= 4]
                wait = next((t[:] for t in round_tables if len(t) < 4), None)
                sa_tables.append(game)
                sa_waiting.append(wait)
            sa_tables, best_cost = self._large_neighborhood_search(
                sa_tables, sa_waiting, floor_val, ceil_val, self.sa_time_limit * 0.5
            )
            best_solution = self._reconstruct_solution(sa_tables, sa_waiting)

        return best_solution

    # =========================================================================
    # 初期解生成
    # =========================================================================

    def _generate_unrealized_first(self) -> List[List[List[int]]]:
        """未実現ペアを優先する手法"""
        all_rounds = []
        pair_count = defaultdict(int)
        waiting_rotation = self._create_optimized_waiting_rotation()

        for round_num in range(self.rounds):
            if waiting_rotation and round_num < len(waiting_rotation):
                waiting_players = waiting_rotation[round_num]
                playing_players = [p for p in self.player_ids if p not in waiting_players]
            else:
                playing_players = list(self.player_ids)
                waiting_players = []

            # 小規模ならパーティション全列挙、それ以外はグリーディ
            if len(playing_players) <= 12:
                tables = self._best_partition(playing_players, pair_count)
            else:
                tables = self._multi_trial_assignment(playing_players, pair_count, trials=30)

            if waiting_players:
                tables.append(waiting_players)

            all_rounds.append(tables)

            for table in tables:
                if len(table) >= 4:
                    for p1, p2 in combinations(table, 2):
                        pair_count[tuple(sorted([p1, p2]))] += 1

        return all_rounds

    def _generate_balanced(self) -> List[List[List[int]]]:
        """バランス重視の手法"""
        all_rounds = []
        pair_count = defaultdict(int)
        waiting_rotation = self._create_optimized_waiting_rotation()

        for round_num in range(self.rounds):
            progress = (round_num + 1) / self.rounds
            target_meetings = self.ideal_meetings_float * progress

            if waiting_rotation and round_num < len(waiting_rotation):
                waiting_players = waiting_rotation[round_num]
                playing_players = [p for p in self.player_ids if p not in waiting_players]
            else:
                playing_players = list(self.player_ids)
                waiting_players = []

            if len(playing_players) <= 12:
                tables = self._best_partition_balanced(playing_players, pair_count, target_meetings)
            else:
                tables = self._balanced_assignment(playing_players, pair_count, target_meetings)

            if waiting_players:
                tables.append(waiting_players)

            all_rounds.append(tables)

            for table in tables:
                if len(table) >= 4:
                    for p1, p2 in combinations(table, 2):
                        pair_count[tuple(sorted([p1, p2]))] += 1

        return all_rounds

    def _generate_random(self) -> List[List[List[int]]]:
        """ランダムな初期解を生成"""
        all_rounds = []
        waiting_rotation = self._create_optimized_waiting_rotation()

        for round_num in range(self.rounds):
            if waiting_rotation and round_num < len(waiting_rotation):
                waiting_players = waiting_rotation[round_num]
                playing_players = [p for p in self.player_ids if p not in waiting_players]
            else:
                playing_players = list(self.player_ids)
                waiting_players = []

            shuffled = playing_players[:]
            random.shuffle(shuffled)
            tables = []
            if self.allow_five:
                num_five = self.optimal_table_config.get('five_tables', 0)
                for _ in range(num_five):
                    if len(shuffled) >= 5:
                        tables.append(shuffled[:5])
                        shuffled = shuffled[5:]
            while len(shuffled) >= 4:
                tables.append(shuffled[:4])
                shuffled = shuffled[4:]

            if waiting_players:
                tables.append(waiting_players)

            all_rounds.append(tables)

        return all_rounds

    # =========================================================================
    # パーティション列挙（12人以下用）
    # =========================================================================

    def _enumerate_partitions(self, players: List[int]) -> List[List[List[int]]]:
        """プレイヤーリストを4人卓に分割する全パーティションを列挙"""
        result = []
        self._partition_helper(players, [], result)
        return result

    def _partition_helper(self, remaining: List[int], current: List[List[int]],
                          result: List[List[List[int]]]):
        """パーティション列挙の再帰ヘルパー"""
        if len(remaining) == 0:
            result.append([table[:] for table in current])
            return
        if len(remaining) < 4:
            return

        # 対称性排除: 最初の要素を固定
        first = remaining[0]
        rest = remaining[1:]

        for combo in combinations(rest, 3):
            table = [first] + list(combo)
            new_remaining = [p for p in rest if p not in combo]
            current.append(table)
            self._partition_helper(new_remaining, current, result)
            current.pop()

    def _best_partition(self, players: List[int],
                        pair_count: Dict[Tuple[int, int], int]) -> List[List[int]]:
        """全パーティションから未実現ペア最優先で最良を選択"""
        partitions = self._enumerate_partitions(players)
        if not partitions:
            return self._greedy_assignment(players, pair_count)

        best_partition = None
        best_score = float('-inf')

        for partition in partitions:
            score = 0
            for table in partition:
                for p1, p2 in combinations(table, 2):
                    pair = tuple(sorted([p1, p2]))
                    count = pair_count.get(pair, 0)
                    if count == 0:
                        score += 1000
                    else:
                        score -= count * count

            if score > best_score:
                best_score = score
                best_partition = partition

        return best_partition

    def _best_partition_balanced(self, players: List[int],
                                 pair_count: Dict[Tuple[int, int], int],
                                 target_meetings: float) -> List[List[int]]:
        """全パーティションからバランス重視で最良を選択"""
        partitions = self._enumerate_partitions(players)
        if not partitions:
            return self._balanced_assignment(players, pair_count, target_meetings)

        best_partition = None
        best_deviation = float('inf')

        for partition in partitions:
            deviation = 0
            for table in partition:
                for p1, p2 in combinations(table, 2):
                    pair = tuple(sorted([p1, p2]))
                    current = pair_count.get(pair, 0)
                    deviation += abs(current + 1 - target_meetings)

            if deviation < best_deviation:
                best_deviation = deviation
                best_partition = partition

        return best_partition

    # =========================================================================
    # グリーディ・ランダム割り当て（16人以上用）
    # =========================================================================

    def _multi_trial_assignment(self, players: List[int],
                                pair_count: Dict[Tuple[int, int], int],
                                trials: int = 30) -> List[List[int]]:
        """複数回試行して最良の割り当てを選択"""
        best_tables = None
        best_score = float('-inf')

        for attempt in range(trials):
            if attempt == 0:
                tables = self._greedy_assignment(players, pair_count)
            else:
                tables = self._randomized_assignment(players, pair_count)

            score = self._score_table_assignment(tables, pair_count)
            if score > best_score:
                best_score = score
                best_tables = tables

        return best_tables

    def _greedy_assignment(self, players: List[int],
                          pair_count: Dict[Tuple[int, int], int]) -> List[List[int]]:
        """グリーディな卓割り当て"""
        tables = []
        remaining = players[:]

        if self.allow_five and len(remaining) % 4 != 0:
            num_five_tables = len(remaining) % 4
            for _ in range(num_five_tables):
                if len(remaining) >= 5:
                    table = self._select_best_table_size(remaining, pair_count, 5)
                    tables.append(table)
                    for p in table:
                        remaining.remove(p)

        while len(remaining) >= 4:
            table = self._select_best_table(remaining, pair_count)
            tables.append(table)
            for p in table:
                remaining.remove(p)

        return tables

    def _select_best_table_size(self, players: List[int],
                               pair_count: Dict[Tuple[int, int], int],
                               size: int) -> List[int]:
        """指定サイズの最適な卓を選択"""
        if len(players) <= 10:
            best_table = None
            best_score = float('-inf')
            for table in combinations(players, size):
                score = 0
                for p1, p2 in combinations(table, 2):
                    pair = tuple(sorted([p1, p2]))
                    count = pair_count.get(pair, 0)
                    if count == 0:
                        score += 1000
                    else:
                        score -= count * count
                if score > best_score:
                    best_score = score
                    best_table = list(table)
            return best_table
        else:
            return random.sample(players, size)

    def _select_best_table(self, players: List[int],
                          pair_count: Dict[Tuple[int, int], int]) -> List[int]:
        """最適な4人卓を選択"""
        if len(players) <= 8:
            best_table = None
            best_score = float('-inf')
            for table in combinations(players, 4):
                score = 0
                for p1, p2 in combinations(table, 2):
                    pair = tuple(sorted([p1, p2]))
                    count = pair_count.get(pair, 0)
                    if count == 0:
                        score += 1000
                    else:
                        score -= count * count
                if score > best_score:
                    best_score = score
                    best_table = list(table)
            return best_table
        else:
            player_scores = {}
            for p in players:
                score = sum(pair_count.get(tuple(sorted([p, other])), 0)
                           for other in players if other != p)
                player_scores[p] = score
            sorted_players = sorted(players, key=lambda p: player_scores[p])
            start_player = sorted_players[0]
            candidates = [p for p in players if p != start_player]
            candidates.sort(key=lambda p: pair_count.get(tuple(sorted([start_player, p])), 0))
            return [start_player] + candidates[:3]

    def _randomized_assignment(self, players: List[int],
                              pair_count: Dict[Tuple[int, int], int]) -> List[List[int]]:
        """ランダム性を加えた卓割り当て"""
        tables = []
        remaining = players[:]

        # 5人卓を先に処理
        if self.allow_five:
            num_five = self.optimal_table_config.get('five_tables', 0)
            for _ in range(num_five):
                if len(remaining) >= 5:
                    table = random.sample(remaining, 5)
                    tables.append(table)
                    for p in table:
                        remaining.remove(p)

        while len(remaining) >= 4:
            candidates = []
            sample_size = min(10, math.comb(len(remaining), 4))
            for _ in range(sample_size):
                table = random.sample(remaining, 4)
                score = self._score_single_table(table, pair_count)
                candidates.append((score, table))
            candidates.sort(reverse=True)
            top_k = min(3, len(candidates))
            _, selected_table = candidates[random.randint(0, top_k - 1)]
            tables.append(selected_table)
            for p in selected_table:
                remaining.remove(p)

        return tables

    def _balanced_assignment(self, players: List[int],
                            pair_count: Dict[Tuple[int, int], int],
                            target_meetings: float) -> List[List[int]]:
        """バランスを考慮した卓割り当て"""
        tables = []
        remaining = players[:]

        # 5人卓を先に処理
        if self.allow_five:
            num_five = self.optimal_table_config.get('five_tables', 0)
            for _ in range(num_five):
                if len(remaining) >= 5:
                    best_table = None
                    best_deviation = float('inf')
                    if len(remaining) <= 12:
                        candidates = [list(t) for t in combinations(remaining, 5)]
                    else:
                        candidates = [random.sample(remaining, 5) for _ in range(30)]
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

        while len(remaining) >= 4:
            best_table = None
            best_deviation = float('inf')

            # 候補を生成
            if len(remaining) <= 12:
                candidates = [list(table) for table in combinations(remaining, 4)]
            else:
                candidates = [random.sample(remaining, 4) for _ in range(30)]

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
                tables.append(remaining[:4])
                remaining = remaining[4:]

        return tables

    def _score_single_table(self, table: List[int],
                           pair_count: Dict[Tuple[int, int], int]) -> float:
        """単一卓のスコア"""
        score = 0
        for p1, p2 in combinations(table, 2):
            pair = tuple(sorted([p1, p2]))
            count = pair_count.get(pair, 0)
            if count == 0:
                score += 1000
            elif count < self.ideal_meetings_floor:
                score += 100 / (count + 1)
            else:
                score -= count * count
        return score

    def _score_table_assignment(self, tables: List[List[int]],
                               pair_count: Dict[Tuple[int, int], int]) -> float:
        """卓割り当て全体のスコア"""
        return sum(self._score_single_table(t, pair_count)
                   for t in tables if len(t) >= 4)

    def _create_optimized_waiting_rotation(self) -> Optional[List[List[int]]]:
        """最適化された待機ローテーション"""
        waiting_count = self.optimal_table_config.get('waiting', 0)
        if waiting_count == 0:
            return None

        rotation = []
        wait_counts = defaultdict(int)

        for round_num in range(self.rounds):
            candidates = sorted(self.player_ids, key=lambda p: (wait_counts[p], p))
            waiting = candidates[:waiting_count]
            rotation.append(waiting)
            for p in waiting:
                wait_counts[p] += 1

        return rotation

    # =========================================================================
    # 焼きなまし法（Simulated Annealing）
    # =========================================================================

    @staticmethod
    def _penalty(count: int, floor_val: int, ceil_val: int) -> int:
        """1ペアのペナルティ"""
        if count < floor_val:
            return (floor_val - count) ** 2
        elif count > ceil_val:
            return (count - ceil_val) ** 2
        return 0

    def _compute_cost_from_solution(self, solution: List[List[List[int]]]) -> int:
        """解からコストを直接計算"""
        pair_count = defaultdict(int)
        for round_tables in solution:
            for table in round_tables:
                if len(table) >= 4:
                    for p1, p2 in combinations(table, 2):
                        pair_count[tuple(sorted([p1, p2]))] += 1

        floor_val = self.ideal_meetings_floor
        ceil_val = self.ideal_meetings_ceil
        cost = 0
        for pair in self.all_pairs:
            c = pair_count.get(pair, 0)
            cost += self._penalty(c, floor_val, ceil_val)
        return cost

    def _simulated_annealing(self, solution: List[List[List[int]]],
                             time_limit_sec: float) -> List[List[List[int]]]:
        """焼きなまし法による大域的最適化"""
        floor_val = self.ideal_meetings_floor
        ceil_val = self.ideal_meetings_ceil
        _penalty = self._penalty

        # SA用データ構造を初期化
        n = self.players
        pair_matrix = [[0] * (n + 1) for _ in range(n + 1)]

        sa_round_tables = []
        sa_waiting = []

        for round_tables in solution:
            game_tables = []
            wait = None
            for table in round_tables:
                if len(table) >= 4:
                    game_tables.append(list(table))
                    for p1, p2 in combinations(table, 2):
                        pair_matrix[p1][p2] += 1
                        pair_matrix[p2][p1] += 1
                else:
                    wait = list(table)
            sa_round_tables.append(game_tables)
            sa_waiting.append(wait)

        # 初期コスト計算
        current_cost = 0
        for i in range(1, n + 1):
            for j in range(i + 1, n + 1):
                current_cost += _penalty(pair_matrix[i][j], floor_val, ceil_val)

        best_cost = current_cost
        best_tables = [[table[:] for table in rt] for rt in sa_round_tables]

        if best_cost == 0:
            return self._reconstruct_solution(best_tables, sa_waiting)

        num_rounds = len(sa_round_tables)

        # 温度パラメータ
        T_start = 5.0
        T_end = 0.005
        log_ratio = math.log(T_end / T_start)

        start_time = time.time()
        iteration = 0
        last_improve_iter = 0

        # ローカル変数化で高速化
        _randint = random.randint
        _random = random.random
        _exp = math.exp
        _time = time.time
        check_interval = 500

        while True:
            if iteration % check_interval == 0:
                elapsed = _time() - start_time
                if elapsed >= time_limit_sec:
                    break
                progress = elapsed / time_limit_sec
                T = T_start * _exp(log_ratio * progress)

                # スタグネーション検出: 長時間改善なしなら最良解に戻る
                if iteration - last_improve_iter > 50000 and current_cost > best_cost:
                    # 最良解に復元
                    pair_matrix = [[0] * (n + 1) for _ in range(n + 1)]
                    for r in range(num_rounds):
                        sa_round_tables[r] = [table[:] for table in best_tables[r]]
                        for table in sa_round_tables[r]:
                            for p1, p2 in combinations(table, 2):
                                pair_matrix[p1][p2] += 1
                                pair_matrix[p2][p1] += 1
                    current_cost = best_cost
                    last_improve_iter = iteration

            round_r = _randint(0, num_rounds - 1)
            tables_r = sa_round_tables[round_r]

            if len(tables_r) < 2:
                iteration += 1
                continue

            ta_idx = _randint(0, len(tables_r) - 1)
            tb_idx = _randint(0, len(tables_r) - 2)
            if tb_idx >= ta_idx:
                tb_idx += 1

            table_a = tables_r[ta_idx]
            table_b = tables_r[tb_idx]

            pi_idx = _randint(0, len(table_a) - 1)
            pj_idx = _randint(0, len(table_b) - 1)
            player_i = table_a[pi_idx]
            player_j = table_b[pj_idx]

            # 差分コスト計算
            delta = 0
            pm_i = pair_matrix[player_i]
            pm_j = pair_matrix[player_j]

            for k in range(len(table_a)):
                if k == pi_idx:
                    continue
                m = table_a[k]
                c = pm_i[m]
                delta += _penalty(c - 1, floor_val, ceil_val) - _penalty(c, floor_val, ceil_val)
                c = pm_j[m]
                delta += _penalty(c + 1, floor_val, ceil_val) - _penalty(c, floor_val, ceil_val)

            for k in range(len(table_b)):
                if k == pj_idx:
                    continue
                m = table_b[k]
                c = pm_j[m]
                delta += _penalty(c - 1, floor_val, ceil_val) - _penalty(c, floor_val, ceil_val)
                c = pm_i[m]
                delta += _penalty(c + 1, floor_val, ceil_val) - _penalty(c, floor_val, ceil_val)

            # 受容判定
            if delta <= 0 or (T > 0 and _random() < _exp(-delta / T)):
                table_a[pi_idx] = player_j
                table_b[pj_idx] = player_i

                for k in range(len(table_a)):
                    if k == pi_idx:
                        continue
                    m = table_a[k]
                    pair_matrix[player_i][m] -= 1
                    pair_matrix[m][player_i] -= 1
                    pair_matrix[player_j][m] += 1
                    pair_matrix[m][player_j] += 1

                for k in range(len(table_b)):
                    if k == pj_idx:
                        continue
                    m = table_b[k]
                    pair_matrix[player_j][m] -= 1
                    pair_matrix[m][player_j] -= 1
                    pair_matrix[player_i][m] += 1
                    pair_matrix[m][player_i] += 1

                current_cost += delta

                if current_cost < best_cost:
                    best_cost = current_cost
                    best_tables = [[table[:] for table in rt] for rt in sa_round_tables]
                    last_improve_iter = iteration

                    if best_cost == 0:
                        break

            iteration += 1

        return self._reconstruct_solution(best_tables, sa_waiting)

    def _large_neighborhood_search(self, tables: List[List[List[int]]],
                                    waiting: List[Optional[List[int]]],
                                    floor_val: int, ceil_val: int,
                                    time_limit: float) -> Tuple[List[List[List[int]]], int]:
        """ラウンド再構成による大域的改善"""
        _penalty = self._penalty
        n = self.players
        num_rounds = len(tables)

        # pair_matrixを構築
        pair_matrix = [[0] * (n + 1) for _ in range(n + 1)]
        for rt in tables:
            for table in rt:
                for p1, p2 in combinations(table, 2):
                    pair_matrix[p1][p2] += 1
                    pair_matrix[p2][p1] += 1

        current_cost = 0
        for i in range(1, n + 1):
            for j in range(i + 1, n + 1):
                current_cost += _penalty(pair_matrix[i][j], floor_val, ceil_val)

        start_time = time.time()
        improved = True

        while improved and current_cost > 0:
            improved = False
            if time.time() - start_time > time_limit:
                break

            for r in range(num_rounds):
                if time.time() - start_time > time_limit:
                    break

                playing_players = []
                for table in tables[r]:
                    playing_players.extend(table)

                # このラウンドの貢献をpair_matrixから除去
                for table in tables[r]:
                    for p1, p2 in combinations(table, 2):
                        pair_matrix[p1][p2] -= 1
                        pair_matrix[p2][p1] -= 1

                # パーティション候補の生成
                num_five = self.optimal_table_config.get('five_tables', 0) if self.allow_five else 0
                if len(playing_players) <= 12:
                    partitions = self._enumerate_partitions(playing_players)
                elif len(playing_players) <= 20:
                    partitions = []
                    for _ in range(1000):
                        shuffled = playing_players[:]
                        random.shuffle(shuffled)
                        part = []
                        for _ in range(num_five):
                            if len(shuffled) >= 5:
                                part.append(shuffled[:5])
                                shuffled = shuffled[5:]
                        while len(shuffled) >= 4:
                            part.append(shuffled[:4])
                            shuffled = shuffled[4:]
                        partitions.append(part)
                else:
                    # 復元して次のラウンドへ
                    for table in tables[r]:
                        for p1, p2 in combinations(table, 2):
                            pair_matrix[p1][p2] += 1
                            pair_matrix[p2][p1] += 1
                    continue

                # 現在のパーティションのコスト（ベースライン）
                cur_part_cost = 0
                for table in tables[r]:
                    for p1, p2 in combinations(table, 2):
                        cur_part_cost += _penalty(
                            pair_matrix[p1][p2] + 1, floor_val, ceil_val
                        )

                # 差分コスト計算で最良パーティションを選択
                best_partition = tables[r]
                best_part_cost = cur_part_cost

                for partition in partitions:
                    part_cost = 0
                    for table in partition:
                        for p1, p2 in combinations(table, 2):
                            part_cost += _penalty(
                                pair_matrix[p1][p2] + 1, floor_val, ceil_val
                            )
                        if part_cost >= best_part_cost:
                            break  # 早期打ち切り
                    if part_cost < best_part_cost:
                        best_part_cost = part_cost
                        best_partition = [table[:] for table in partition]

                # 最良パーティションを適用
                tables[r] = best_partition
                for table in best_partition:
                    for p1, p2 in combinations(table, 2):
                        pair_matrix[p1][p2] += 1
                        pair_matrix[p2][p1] += 1

                delta = best_part_cost - cur_part_cost
                if delta < 0:
                    current_cost += delta
                    improved = True
                    if current_cost == 0:
                        break

        return tables, current_cost

    def _reconstruct_solution(self, sa_tables: List[List[List[int]]],
                              sa_waiting: List[Optional[List[int]]]) -> List[List[List[int]]]:
        """SA用データ構造から元の解形式に復元"""
        solution = []
        for r in range(len(sa_tables)):
            round_data = [table[:] for table in sa_tables[r]]
            if sa_waiting[r]:
                round_data.append(sa_waiting[r][:])
            solution.append(round_data)
        return solution

    # =========================================================================
    # ユーティリティ
    # =========================================================================

    def _count_all_pairs(self, solution: List[List[List[int]]]) -> Dict[Tuple[int, int], int]:
        """全ペアの同卓回数をカウント"""
        pair_count = defaultdict(int)
        for round_tables in solution:
            for table in round_tables:
                if len(table) >= 4:
                    for p1, p2 in combinations(table, 2):
                        pair_count[tuple(sorted([p1, p2]))] += 1
        return pair_count

    def _evaluate_hierarchical(self, solution: List[List[List[int]]]) -> float:
        """階層的な評価（条件1~4の優先順位を考慮）"""
        pair_count = self._count_all_pairs(solution)
        all_counts = [pair_count.get(pair, 0) for pair in self.all_pairs]
        min_count = min(all_counts) if all_counts else 0
        max_count = max(all_counts) if all_counts else 0

        count_distribution = defaultdict(int)
        for count in all_counts:
            count_distribution[count] += 1

        score = 0
        score += min_count * 1e9
        score -= max_count * 1e6
        score -= count_distribution[min_count] * 1e3
        score -= count_distribution[max_count]
        return score

    def print_results(self, results: List[List[List[int]]]):
        """結果を見やすく出力"""
        print(f"\n麻雀卓組結果（汎用最適化版） (参加者: {self.players}人, {self.rounds}回戦)")
        print(f"5人打ち: {'あり' if self.allow_five else 'なし'}")
        print("=" * 50)

        pair_count = defaultdict(int)

        for round_num, groups in enumerate(results, 1):
            print(f"\n第{round_num}回戦:")
            for table_num, group in enumerate(groups):
                if len(group) >= 4:
                    players_str = ", ".join(f"P{p}" for p in sorted(group))
                    print(f"  卓{table_num + 1}: {players_str} ({len(group)}人)")
                    for p1, p2 in combinations(group, 2):
                        pair = tuple(sorted([p1, p2]))
                        pair_count[pair] += 1
                else:
                    players_str = ", ".join(f"P{p}" for p in sorted(group))
                    print(f"  待機: {players_str}")

        self._print_pair_statistics(pair_count)

    def _print_pair_statistics(self, pair_count: Dict[Tuple[int, int], int]):
        """ペアの統計情報を表示"""
        print("\n" + "=" * 50)
        print("同卓回数統計:")

        all_counts = []
        for pair in self.all_pairs:
            count = pair_count.get(pair, 0)
            all_counts.append(count)

        if not all_counts:
            print("統計データなし")
            return

        import statistics
        min_count = min(all_counts)
        max_count = max(all_counts)
        mean_count = statistics.mean(all_counts)
        stdev_count = statistics.stdev(all_counts) if len(all_counts) > 1 else 0

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

        realized_pairs = sum(1 for count in all_counts if count > 0)
        coverage = realized_pairs / self.total_pairs * 100

        print(f"\nペアカバレッジ: {realized_pairs}/{self.total_pairs} ({coverage:.1f}%)")

        print("\n条件達成度（階層的優先順位）:")

        if self.ideal_meetings_floor > 0:
            achievement1 = "✓ 完璧" if min_count >= self.ideal_meetings_floor else "△ 部分的"
        else:
            achievement1 = "✓ 最良" if min_count == 0 else "✗ 未達成"
        print(f"1. 【最重要】同卓回数の最小: {min_count}回 {achievement1}")

        if max_count <= self.ideal_meetings_ceil:
            achievement2 = "✓ 理想的"
        elif max_count <= self.ideal_meetings_ceil + 1:
            achievement2 = "△ 許容範囲"
        else:
            achievement2 = "✗ 要改善"
        print(f"2. 【重要】同卓回数の最大: {max_count}回 {achievement2}")

        min_pairs_ratio = count_distribution[min_count] / self.total_pairs
        if min_pairs_ratio <= 0.3:
            achievement3 = "✓ 良好"
        elif min_pairs_ratio <= 0.5:
            achievement3 = "△ 普通"
        else:
            achievement3 = "✗ 偏り大"
        print(f"3. 【やや重要】最小回数のペア数: {count_distribution[min_count]}ペア ({min_pairs_ratio*100:.1f}%) {achievement3}")

        max_pairs_ratio = count_distribution[max_count] / self.total_pairs
        if max_pairs_ratio <= 0.1:
            achievement4 = "✓ 良好"
        elif max_pairs_ratio <= 0.2:
            achievement4 = "△ 普通"
        else:
            achievement4 = "✗ 偏り大"
        print(f"4. 【優先度低】最大回数のペア数: {count_distribution[max_count]}ペア ({max_pairs_ratio*100:.1f}%) {achievement4}")

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
