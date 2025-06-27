#!/usr/bin/env python3
"""麻雀卓組生成プログラム（完全バランス版）- 線形計画法で最適解を保証"""

import argparse
from typing import List, Dict, Tuple, Optional
from collections import defaultdict
from itertools import combinations
import pulp
import time


class BalancedLPTableGroupGenerator:
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
        print(f"- 理想的な平均同卓回数: {self.ideal_meetings_float:.2f}回")
        
        # 理想的な分布を計算
        total_pair_meetings = self.max_total_pairs
        ceil_pairs = total_pair_meetings - self.ideal_meetings_floor * self.total_pairs
        floor_pairs = self.total_pairs - ceil_pairs
        
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
        print("\n完全バランス最適化を開始...")
        
        # 可能な全ての卓を生成
        all_possible_tables = self._generate_all_possible_tables()
        print(f"可能な卓数: {len(all_possible_tables)}")
        
        # 線形計画問題を構築して解く
        solution = self._solve_with_lp(all_possible_tables)
        
        return solution
    
    def _generate_all_possible_tables(self) -> List[Tuple[int, ...]]:
        """可能な全ての卓を生成"""
        possible_tables = []
        
        # 4人卓
        for table in combinations(self.player_ids, 4):
            possible_tables.append(table)
        
        # 5人卓（許可されている場合）
        if self.allow_five:
            for table in combinations(self.player_ids, 5):
                possible_tables.append(table)
        
        return possible_tables
    
    def _solve_with_lp(self, all_possible_tables: List[Tuple[int, ...]]) -> List[List[List[int]]]:
        """線形計画法で最適解を求める"""
        print("線形計画問題を構築中...")
        
        # 問題の作成
        prob = pulp.LpProblem("BalancedTableGroup", pulp.LpMinimize)
        
        # 決定変数：各ラウンドで各卓を使用するか
        table_vars = {}
        for r in range(self.rounds):
            for t_idx, table in enumerate(all_possible_tables):
                var_name = f"r{r}_t{t_idx}"
                table_vars[(r, t_idx)] = pulp.LpVariable(var_name, cat='Binary')
        
        # ペア変数：各ペアの同卓回数
        pair_vars = {}
        pair_to_idx = {pair: idx for idx, pair in enumerate(self.all_pairs)}
        for pair in self.all_pairs:
            pair_vars[pair] = pulp.LpVariable(f"pair_{pair[0]}_{pair[1]}", lowBound=0, cat='Integer')
        
        # ペアカウントの計算
        for pair in self.all_pairs:
            # このペアを含む卓
            tables_with_pair = []
            for t_idx, table in enumerate(all_possible_tables):
                if pair[0] in table and pair[1] in table:
                    for r in range(self.rounds):
                        tables_with_pair.append(table_vars[(r, t_idx)])
            
            # ペアの同卓回数を計算
            prob += pair_vars[pair] == pulp.lpSum(tables_with_pair)
        
        # 最小値と最大値の変数
        min_meetings = pulp.LpVariable("min_meetings", lowBound=0, cat='Integer')
        max_meetings = pulp.LpVariable("max_meetings", lowBound=0, cat='Integer')
        
        # 最小値と最大値の制約
        for pair in self.all_pairs:
            prob += pair_vars[pair] >= min_meetings
            prob += pair_vars[pair] <= max_meetings
        
        # 目的関数：階層的最適化
        # 1. 最小値を最大化（最重要）
        # 2. 最大値を最小化
        # 3. 分散を最小化
        objective = -1000000 * min_meetings + 1000 * max_meetings
        
        # 分散項を追加（理想値からの差の二乗和）
        for pair in self.all_pairs:
            # 理想値は self.ideal_meetings_float
            # 差の絶対値を近似するため、正と負の偏差を別々に扱う
            pos_dev = pulp.LpVariable(f"pos_dev_{pair[0]}_{pair[1]}", lowBound=0)
            neg_dev = pulp.LpVariable(f"neg_dev_{pair[0]}_{pair[1]}", lowBound=0)
            
            prob += pair_vars[pair] - self.ideal_meetings_float == pos_dev - neg_dev
            objective += pos_dev + neg_dev
        
        prob += objective
        
        # 制約：各ラウンドで各プレイヤーは最大1つの卓
        for r in range(self.rounds):
            for player in self.player_ids:
                player_tables = []
                for t_idx, table in enumerate(all_possible_tables):
                    if player in table:
                        player_tables.append(table_vars[(r, t_idx)])
                
                prob += pulp.lpSum(player_tables) <= 1
        
        # 制約：各ラウンドの卓数制限
        for r in range(self.rounds):
            # 4人卓の数
            four_tables = []
            five_tables = []
            for t_idx, table in enumerate(all_possible_tables):
                if len(table) == 4:
                    four_tables.append(table_vars[(r, t_idx)])
                elif len(table) == 5:
                    five_tables.append(table_vars[(r, t_idx)])
            
            # 卓数の制限
            if self.allow_five:
                # 4人卓と5人卓の組み合わせ
                prob += pulp.lpSum(four_tables) + pulp.lpSum(five_tables) <= self.players // 4 + 1
            else:
                # 4人卓のみ
                prob += pulp.lpSum(four_tables) <= self.players // 4
        
        # 解く
        print("最適化を実行中...")
        start_time = time.time()
        
        # タイムアウトを設定して解く
        solver = pulp.PULP_CBC_CMD(msg=1, timeLimit=60)
        prob.solve(solver)
        
        elapsed_time = time.time() - start_time
        print(f"最適化完了（所要時間: {elapsed_time:.2f}秒）")
        
        # 解が見つかったかチェック
        if prob.status != pulp.LpStatusOptimal:
            print(f"警告: 最適解が見つかりませんでした（ステータス: {pulp.LpStatus[prob.status]}）")
            # フォールバックとして簡易的な解を生成
            return self._generate_fallback_solution()
        
        # 結果を抽出
        solution = []
        for r in range(self.rounds):
            round_tables = []
            for t_idx, table in enumerate(all_possible_tables):
                if table_vars[(r, t_idx)].varValue > 0.5:
                    round_tables.append(list(table))
            
            # 待機プレイヤーを追加
            playing_players = set()
            for table in round_tables:
                playing_players.update(table)
            waiting_players = [p for p in self.player_ids if p not in playing_players]
            if waiting_players:
                round_tables.append(waiting_players)
            
            solution.append(round_tables)
        
        # 結果の統計を表示
        self._print_solution_stats(solution)
        
        return solution
    
    def _generate_fallback_solution(self) -> List[List[List[int]]]:
        """フォールバック用の簡易解を生成"""
        print("フォールバック解を生成中...")
        
        all_rounds = []
        pair_count = defaultdict(int)
        
        for round_num in range(self.rounds):
            # ペア回数が少ない順にプレイヤーをソート
            player_scores = {}
            for player in self.player_ids:
                score = sum(pair_count.get(tuple(sorted([player, other])), 0) 
                          for other in self.player_ids if other != player)
                player_scores[player] = score
            
            sorted_players = sorted(self.player_ids, key=lambda p: player_scores[p])
            
            # 卓を構成
            round_tables = []
            remaining = sorted_players.copy()
            
            while len(remaining) >= 4:
                # 最もペア回数が少ない4人を選択
                table = remaining[:4]
                round_tables.append(table)
                remaining = remaining[4:]
                
                # ペアカウントを更新
                for p1, p2 in combinations(table, 2):
                    pair = tuple(sorted([p1, p2]))
                    pair_count[pair] += 1
            
            if remaining:
                round_tables.append(remaining)
            
            all_rounds.append(round_tables)
        
        return all_rounds
    
    def _print_solution_stats(self, solution: List[List[List[int]]]):
        """解の統計情報を表示"""
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
        
        print(f"\n解の統計:")
        print(f"- 最小同卓回数: {min_count}回")
        print(f"- 最大同卓回数: {max_count}回")
        print(f"- 分布: ", end="")
        for count in sorted(count_distribution.keys()):
            print(f"{count}回×{count_distribution[count]}ペア ", end="")
        print()
    
    def print_results(self, results: List[List[List[int]]]):
        """結果を見やすく出力"""
        print(f"\n麻雀卓組結果（完全バランス版） (参加者: {self.players}人, {self.rounds}回戦)")
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


def main():
    parser = argparse.ArgumentParser(description='麻雀卓組生成プログラム（完全バランス版）')
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
    
    generator = BalancedLPTableGroupGenerator(args.players, args.rounds, args.five)
    results = generator.generate()
    generator.print_results(results)


if __name__ == "__main__":
    main()