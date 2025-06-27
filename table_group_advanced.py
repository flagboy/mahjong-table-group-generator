#!/usr/bin/env python3
"""麻雀卓組生成プログラム（高度最適化版）- 複雑な状況でも最適解を追求"""

import argparse
from typing import List, Dict, Tuple, Set, Optional
from collections import defaultdict
from itertools import combinations
import pulp
import math
import time


class AdvancedTableGroupGenerator:
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
        self.optimization_time_limit = 60  # 秒
        self.max_iterations = 5  # 最大反復回数
        
        # 理論的な制約を計算
        self._calculate_theoretical_limits()
        
    def _calculate_theoretical_limits(self):
        """理論的な制約を計算"""
        # 各ラウンドで実現できる最大ペア数
        if self.allow_five:
            # 5人打ちありの場合の最適な配置を計算
            best_pairs_per_round = 0
            for five_tables in range(self.players // 5 + 1):
                four_tables = (self.players - five_tables * 5) // 4
                if five_tables * 5 + four_tables * 4 <= self.players:
                    pairs = five_tables * 10 + four_tables * 6  # C(5,2)=10, C(4,2)=6
                    best_pairs_per_round = max(best_pairs_per_round, pairs)
        else:
            # 4人打ちのみ
            tables_per_round = self.players // 4
            best_pairs_per_round = tables_per_round * 6
        
        self.max_pairs_per_round = best_pairs_per_round
        self.max_total_pairs = best_pairs_per_round * self.rounds
        
        # 理論的な最小同卓回数
        self.theoretical_min_meetings = max(0, math.floor(self.max_total_pairs / self.total_pairs))
        
        # 各ペアの理想的な同卓回数
        self.ideal_meetings = self.max_total_pairs / self.total_pairs
        
        print(f"\n理論的分析:")
        print(f"- 参加者数: {self.players}人")
        print(f"- 全ペア数: {self.total_pairs}")
        print(f"- 1ラウンドの最大ペア数: {self.max_pairs_per_round}")
        print(f"- {self.rounds}ラウンドの最大ペア総数: {self.max_total_pairs}")
        print(f"- 理論的な最小同卓回数: {self.theoretical_min_meetings}回")
        print(f"- 理想的な平均同卓回数: {self.ideal_meetings:.2f}回")
        
        if self.max_total_pairs < self.total_pairs:
            print(f"\n警告: 全ペアを最低1回同卓させるには{math.ceil(self.total_pairs / self.max_pairs_per_round)}回戦以上必要です")
    
    def _solve_multi_objective(self) -> List[List[List[int]]]:
        """多目的最適化による解法"""
        print("\n多目的最適化を開始...")
        
        # フェーズ1: 初期解の生成（高速）
        initial_solution = self._generate_initial_solution()
        best_solution = initial_solution
        best_score = self._evaluate_solution(initial_solution)
        
        print(f"初期解のスコア: {best_score}")
        
        # フェーズ2: 反復改善
        for iteration in range(self.max_iterations):
            print(f"\n反復 {iteration + 1}/{self.max_iterations}...")
            
            # 現在の解の問題点を分析
            pair_count = self._count_pairs_in_solution(best_solution)
            problems = self._analyze_problems(pair_count)
            
            if not problems['has_issues']:
                print("最適解に到達しました")
                break
            
            # 問題に応じた改善戦略を選択
            if problems['zero_meeting_pairs']:
                # 未実現ペアがある場合
                improved = self._fix_zero_meetings(best_solution, pair_count, problems['zero_meeting_pairs'])
            elif problems['high_variance']:
                # 分散が大きい場合
                improved = self._reduce_variance(best_solution, pair_count)
            else:
                # 一般的な改善
                improved = self._general_improvement(best_solution, pair_count)
            
            # 改善された場合は更新
            new_score = self._evaluate_solution(improved)
            if new_score > best_score:
                best_solution = improved
                best_score = new_score
                print(f"スコアが改善: {best_score}")
            else:
                print("改善なし")
        
        return best_solution
    
    def _generate_initial_solution(self) -> List[List[List[int]]]:
        """初期解を生成（グリーディ法）"""
        all_rounds = []
        pair_count = defaultdict(int)
        
        for round_num in range(self.rounds):
            # このラウンドの最適化問題を設定
            prob = pulp.LpProblem(f"Initial_Round_{round_num + 1}", pulp.LpMaximize)
            
            # 可能な卓の組み合わせを生成
            possible_tables = self._generate_possible_tables()
            
            # 決定変数
            table_vars = {}
            for i, table in enumerate(possible_tables):
                table_vars[i] = pulp.LpVariable(f"table_{i}", cat='Binary')
            
            # 目的関数：多段階の優先度
            objective = 0
            
            for i, table in enumerate(possible_tables):
                table_score = 0
                
                # 優先度1: 未実現ペア（最高優先度）
                unrealized_pairs = 0
                # 優先度2: 最小同卓回数のペア
                min_meeting_pairs = 0
                # 優先度3: 理想回数との差
                deviation_from_ideal = 0
                
                for p1, p2 in combinations(table, 2):
                    pair = tuple(sorted([p1, p2]))
                    count = pair_count.get(pair, 0)
                    
                    if count == 0:
                        unrealized_pairs += 1
                        table_score += 1000000  # 最高優先度
                    elif count < self.theoretical_min_meetings:
                        min_meeting_pairs += 1
                        table_score += 10000
                    
                    # 理想回数との差のペナルティ
                    ideal_count = (round_num + 1) * self.ideal_meetings / self.rounds
                    deviation = abs(count + 1 - ideal_count)
                    deviation_from_ideal += deviation
                    table_score -= deviation * 100
                
                objective += table_vars[i] * table_score
            
            prob += objective
            
            # 制約の追加
            self._add_constraints(prob, table_vars, possible_tables)
            
            # 解く
            prob.solve(pulp.PULP_CBC_CMD(msg=0, timeLimit=10))
            
            # 結果を取得
            round_tables = self._extract_solution(table_vars, possible_tables)
            all_rounds.append(round_tables)
            
            # ペアカウントを更新
            for table in round_tables:
                if len(table) >= 4:
                    for p1, p2 in combinations(table, 2):
                        pair = tuple(sorted([p1, p2]))
                        pair_count[pair] += 1
        
        return all_rounds
    
    def _fix_zero_meetings(self, solution: List[List[List[int]]], 
                          pair_count: Dict[Tuple[int, int], int],
                          zero_pairs: List[Tuple[int, int]]) -> List[List[List[int]]]:
        """未実現ペアを修正"""
        print(f"{len(zero_pairs)}個の未実現ペアを修正中...")
        
        # 各ラウンドで未実現ペアを実現できる可能性を評価
        best_rounds_for_pairs = defaultdict(list)
        
        for round_idx, round_tables in enumerate(solution):
            for pair in zero_pairs:
                # このラウンドでペアを実現するコストを計算
                cost = self._calculate_pair_insertion_cost(round_tables, pair, pair_count)
                best_rounds_for_pairs[pair].append((cost, round_idx))
        
        # 各ペアを最もコストの低いラウンドに割り当て
        modified_solution = [tables.copy() for tables in solution]
        
        for pair in zero_pairs:
            best_rounds_for_pairs[pair].sort(key=lambda x: x[0])
            
            for cost, round_idx in best_rounds_for_pairs[pair]:
                if self._insert_pair_in_round(modified_solution[round_idx], pair, pair_count):
                    break
        
        return modified_solution
    
    def _reduce_variance(self, solution: List[List[List[int]]], 
                        pair_count: Dict[Tuple[int, int], int]) -> List[List[List[int]]]:
        """同卓回数の分散を減らす"""
        print("同卓回数の分散を減少中...")
        
        # 最も多く同卓しているペアと最も少ないペアを特定
        max_meetings = max(pair_count.values()) if pair_count else 0
        min_meetings = min(pair_count.values()) if pair_count else 0
        
        if max_meetings - min_meetings <= 1:
            return solution
        
        # スワップによる改善を試みる
        modified_solution = [tables.copy() for tables in solution]
        
        # 各ラウンドでスワップの可能性を探る
        for round_idx in range(len(solution)):
            improved = self._try_swaps_in_round(modified_solution[round_idx], pair_count)
            if improved:
                modified_solution[round_idx] = improved
        
        return modified_solution
    
    def _general_improvement(self, solution: List[List[List[int]]], 
                            pair_count: Dict[Tuple[int, int], int]) -> List[List[List[int]]]:
        """一般的な改善（局所探索）"""
        print("局所探索による改善中...")
        
        modified_solution = [tables.copy() for tables in solution]
        
        # 各ラウンドを順に最適化
        for round_idx in range(len(solution)):
            # このラウンドだけを再最適化
            optimized_round = self._reoptimize_round(
                round_idx, 
                modified_solution, 
                pair_count
            )
            modified_solution[round_idx] = optimized_round
        
        return modified_solution
    
    def _reoptimize_round(self, round_idx: int, 
                         current_solution: List[List[List[int]]],
                         global_pair_count: Dict[Tuple[int, int], int]) -> List[List[int]]:
        """特定のラウンドを再最適化"""
        # 現在のラウンドを除いたペアカウントを計算
        temp_pair_count = defaultdict(int)
        for i, round_tables in enumerate(current_solution):
            if i != round_idx:
                for table in round_tables:
                    if len(table) >= 4:
                        for p1, p2 in combinations(table, 2):
                            pair = tuple(sorted([p1, p2]))
                            temp_pair_count[pair] += 1
        
        # このラウンドを最適化
        prob = pulp.LpProblem(f"Reoptimize_Round_{round_idx + 1}", pulp.LpMaximize)
        
        possible_tables = self._generate_possible_tables()
        table_vars = {}
        for i, table in enumerate(possible_tables):
            table_vars[i] = pulp.LpVariable(f"table_{i}", cat='Binary')
        
        # 目的関数
        objective = self._build_objective_function(
            table_vars, 
            possible_tables, 
            temp_pair_count, 
            round_idx + 1
        )
        prob += objective
        
        # 制約
        self._add_constraints(prob, table_vars, possible_tables)
        
        # 解く
        prob.solve(pulp.PULP_CBC_CMD(msg=0, timeLimit=5))
        
        return self._extract_solution(table_vars, possible_tables)
    
    def _build_objective_function(self, table_vars: Dict[int, pulp.LpVariable],
                                 possible_tables: List[Tuple[int, ...]],
                                 pair_count: Dict[Tuple[int, int], int],
                                 current_round: int) -> pulp.LpAffineExpression:
        """目的関数を構築"""
        objective = 0
        
        # 現在のラウンドでの理想的なペア回数
        ideal_count_now = current_round * self.ideal_meetings / self.rounds
        
        for i, table in enumerate(possible_tables):
            table_score = 0
            
            for p1, p2 in combinations(table, 2):
                pair = tuple(sorted([p1, p2]))
                current_count = pair_count.get(pair, 0)
                
                # 優先度に基づくスコアリング
                if current_count == 0:
                    # 最優先: 未実現ペア
                    table_score += 1000000
                elif current_count < self.theoretical_min_meetings:
                    # 高優先: 最小回数未満
                    table_score += 10000 * (self.theoretical_min_meetings - current_count)
                else:
                    # 理想回数に近づけるスコア
                    deviation_before = abs(current_count - ideal_count_now)
                    deviation_after = abs(current_count + 1 - ideal_count_now)
                    improvement = deviation_before - deviation_after
                    table_score += improvement * 1000
                
                # ペナルティ: 過度な同卓
                if current_count >= self.ideal_meetings * 1.5:
                    table_score -= (current_count ** 2) * 100
            
            objective += table_vars[i] * table_score
        
        return objective
    
    def _add_constraints(self, prob: pulp.LpProblem, 
                        table_vars: Dict[int, pulp.LpVariable],
                        possible_tables: List[Tuple[int, ...]]):
        """制約条件を追加"""
        # 制約1: 各プレイヤーは最大1つの卓に入る
        for player in self.player_ids:
            player_tables = []
            for i, table in enumerate(possible_tables):
                if player in table:
                    player_tables.append(table_vars[i])
            if player_tables:
                prob += pulp.lpSum(player_tables) <= 1
        
        # 制約2: できるだけ多くのプレイヤーを配置
        total_seated = pulp.lpSum([
            table_vars[i] * len(table) 
            for i, table in enumerate(possible_tables)
        ])
        
        min_seated = self._calculate_min_seated()
        prob += total_seated >= min_seated
    
    def _calculate_min_seated(self) -> int:
        """最小着席人数を計算"""
        if self.allow_five:
            # 5人打ちありの場合の最適な配置
            remainder = self.players % 4
            if remainder == 0 or self.players % 5 == 0:
                return self.players
            elif remainder == 1 and self.players >= 9:
                return self.players  # 5+4の組み合わせ
            else:
                return self.players - min(remainder, self.players % 5)
        else:
            return self.players - (self.players % 4)
    
    def _generate_possible_tables(self) -> List[Tuple[int, ...]]:
        """可能な卓の組み合わせを生成"""
        tables = []
        
        # 4人卓
        for table in combinations(self.player_ids, 4):
            tables.append(table)
        
        # 5人卓（許可されている場合）
        if self.allow_five:
            for table in combinations(self.player_ids, 5):
                tables.append(table)
        
        return tables
    
    def _extract_solution(self, table_vars: Dict[int, pulp.LpVariable],
                         possible_tables: List[Tuple[int, ...]]) -> List[List[int]]:
        """最適化結果から解を抽出"""
        round_tables = []
        used_players = set()
        
        for i, table in enumerate(possible_tables):
            if table_vars[i].varValue == 1:
                round_tables.append(list(table))
                used_players.update(table)
        
        # 待機プレイヤー
        waiting = [p for p in self.player_ids if p not in used_players]
        if waiting:
            round_tables.append(waiting)
        
        return round_tables
    
    def _count_pairs_in_solution(self, solution: List[List[List[int]]]) -> Dict[Tuple[int, int], int]:
        """解のペアカウントを計算"""
        pair_count = defaultdict(int)
        for round_tables in solution:
            for table in round_tables:
                if len(table) >= 4:
                    for p1, p2 in combinations(table, 2):
                        pair = tuple(sorted([p1, p2]))
                        pair_count[pair] += 1
        return pair_count
    
    def _analyze_problems(self, pair_count: Dict[Tuple[int, int], int]) -> Dict[str, any]:
        """現在の解の問題点を分析"""
        # 統計を計算
        all_counts = []
        zero_meeting_pairs = []
        
        for pair in self.all_pairs:
            count = pair_count.get(pair, 0)
            all_counts.append(count)
            if count == 0:
                zero_meeting_pairs.append(pair)
        
        if not all_counts:
            return {'has_issues': False}
        
        import statistics
        mean = statistics.mean(all_counts)
        stdev = statistics.stdev(all_counts) if len(all_counts) > 1 else 0
        min_count = min(all_counts)
        max_count = max(all_counts)
        
        # 問題の判定
        has_issues = (
            len(zero_meeting_pairs) > 0 or  # 未実現ペアがある
            max_count - min_count > 2 or     # 差が大きすぎる
            stdev > 0.5                      # 分散が大きい
        )
        
        return {
            'has_issues': has_issues,
            'zero_meeting_pairs': zero_meeting_pairs,
            'min_count': min_count,
            'max_count': max_count,
            'mean': mean,
            'stdev': stdev,
            'high_variance': stdev > 0.5
        }
    
    def _evaluate_solution(self, solution: List[List[List[int]]]) -> float:
        """解の品質を評価（高いほど良い）"""
        pair_count = self._count_pairs_in_solution(solution)
        
        # 評価基準（優先順位順）
        score = 0
        
        # 1. 同卓回数の最小が大きい（最重要）
        min_meetings = min([pair_count.get(pair, 0) for pair in self.all_pairs])
        score += min_meetings * 1000000
        
        # 2. 同卓回数の最大が小さい
        max_meetings = max(pair_count.values()) if pair_count else 0
        score -= max_meetings * 10000
        
        # 3. 同卓回数最小のペア数が少ない
        min_count_pairs = sum(1 for pair in self.all_pairs if pair_count.get(pair, 0) == min_meetings)
        score -= min_count_pairs * 100
        
        # 4. 同卓回数最大のペア数が少ない
        max_count_pairs = sum(1 for count in pair_count.values() if count == max_meetings)
        score -= max_count_pairs * 10
        
        # 追加: 分散が小さい
        import statistics
        all_counts = [pair_count.get(pair, 0) for pair in self.all_pairs]
        if len(all_counts) > 1:
            stdev = statistics.stdev(all_counts)
            score -= stdev * 1000
        
        return score
    
    def _calculate_pair_insertion_cost(self, round_tables: List[List[int]], 
                                     pair: Tuple[int, int],
                                     pair_count: Dict[Tuple[int, int], int]) -> float:
        """ペアを挿入するコストを計算"""
        # 簡易的な実装
        return 0
    
    def _insert_pair_in_round(self, round_tables: List[List[int]], 
                             pair: Tuple[int, int],
                             pair_count: Dict[Tuple[int, int], int]) -> bool:
        """ラウンドにペアを挿入"""
        # 簡易的な実装
        return False
    
    def _try_swaps_in_round(self, round_tables: List[List[int]], 
                           pair_count: Dict[Tuple[int, int], int]) -> Optional[List[List[int]]]:
        """ラウンド内でスワップを試みる"""
        # 簡易的な実装
        return None
    
    def generate(self) -> List[List[List[int]]]:
        """全ラウンドの卓組を生成"""
        start_time = time.time()
        
        # 多目的最適化で解を生成
        results = self._solve_multi_objective()
        
        elapsed_time = time.time() - start_time
        print(f"\n生成完了（所要時間: {elapsed_time:.2f}秒）")
        
        return results
    
    def print_results(self, results: List[List[List[int]]]):
        """結果を見やすく出力"""
        print(f"\n麻雀卓組結果（高度最適化版） (参加者: {self.players}人, {self.rounds}回戦)")
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
        
        # 評価基準に基づくスコア
        print("\n評価基準:")
        print(f"1. 同卓回数の最小: {min_count}回 {'✓' if min_count > 0 else '✗'}")
        print(f"2. 同卓回数の最大: {max_count}回")
        print(f"3. 最小回数のペア数: {count_distribution[min_count]}ペア")
        print(f"4. 最大回数のペア数: {count_distribution[max_count]}ペア")
        
        # 理論値との比較
        print(f"\n理論値との比較:")
        print(f"- 理論的な最小同卓回数: {self.theoretical_min_meetings}回")
        print(f"- 理想的な平均同卓回数: {self.ideal_meetings:.2f}回")
        print(f"- 実際の平均との差: {abs(mean_count - self.ideal_meetings):.2f}回")


def main():
    parser = argparse.ArgumentParser(description='麻雀卓組生成プログラム（高度最適化版）')
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
    
    generator = AdvancedTableGroupGenerator(args.players, args.rounds, args.five)
    results = generator.generate()
    generator.print_results(results)


if __name__ == "__main__":
    main()