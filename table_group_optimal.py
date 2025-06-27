#!/usr/bin/env python3
"""麻雀卓組生成プログラム（最適実装版）- 全規模で安定動作し条件を最大限満たす"""

import argparse
from typing import List, Dict, Tuple, Set, Optional
from collections import defaultdict
from itertools import combinations
import pulp
import math
import time
import random


class OptimalTableGroupGenerator:
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
        
        # タイムアウト設定
        self.timeout_per_round = 10  # 各ラウンドの最大計算時間（秒）
        
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
        
        # 理想的な分配
        total_pair_meetings = self.max_total_pairs
        ceil_pairs = total_pair_meetings - self.ideal_meetings_floor * self.total_pairs
        floor_pairs = self.total_pairs - ceil_pairs
        
        self.target_distribution = {
            self.ideal_meetings_floor: floor_pairs,
            self.ideal_meetings_ceil: ceil_pairs
        }
        
        print(f"\n理論的分析:")
        print(f"- 参加者数: {self.players}人")
        print(f"- 全ペア数: {self.total_pairs}")
        print(f"- 1ラウンドの最大ペア数: {self.max_pairs_per_round}")
        print(f"- {self.rounds}ラウンドの最大ペア総数: {self.max_total_pairs}")
        print(f"- 理想的な同卓回数: {self.ideal_meetings_float:.2f}回")
        
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
        print("\n最適化を開始...")
        
        # 階層的アプローチ：まず高速なヒューリスティックで初期解を生成
        initial_solution = self._generate_heuristic_solution()
        
        # 初期解の評価
        initial_score = self._evaluate_solution(initial_solution)
        print(f"初期解の評価: {self._format_evaluation(initial_score)}")
        
        # 改善フェーズ
        improved_solution = self._improve_solution(initial_solution, initial_score)
        
        return improved_solution
    
    def _generate_heuristic_solution(self) -> List[List[List[int]]]:
        """ヒューリスティックによる初期解生成"""
        all_rounds = []
        pair_count = defaultdict(int)
        
        # 待機ローテーション用
        if hasattr(self.optimal_table_config, 'waiting') and self.optimal_table_config['waiting'] > 0:
            waiting_rotation = self._create_waiting_rotation()
        else:
            waiting_rotation = None
        
        for round_num in range(self.rounds):
            # このラウンドの卓組を生成
            if waiting_rotation:
                waiting_players = waiting_rotation[round_num % len(waiting_rotation)]
                playing_players = [p for p in self.player_ids if p not in waiting_players]
            else:
                playing_players = self.player_ids
                waiting_players = []
            
            # グリーディアルゴリズムで卓を構成
            round_tables = self._greedy_table_assignment(playing_players, pair_count)
            
            # 待機プレイヤーを追加
            if waiting_players:
                round_tables.append(waiting_players)
            
            all_rounds.append(round_tables)
            
            # ペアカウントを更新
            for table in round_tables:
                if len(table) >= 4:
                    for p1, p2 in combinations(table, 2):
                        pair = tuple(sorted([p1, p2]))
                        pair_count[pair] += 1
        
        return all_rounds
    
    def _create_waiting_rotation(self) -> List[List[int]]:
        """待機プレイヤーのローテーションを作成"""
        waiting_count = self.optimal_table_config.get('waiting', 0)
        if waiting_count == 0:
            return []
        
        # 各プレイヤーが均等に待機するようにローテーション
        rotations = []
        for i in range(self.players):
            waiting = []
            for j in range(waiting_count):
                waiting.append(((i + j) % self.players) + 1)
            rotations.append(waiting)
        
        return rotations
    
    def _greedy_table_assignment(self, players: List[int], 
                                pair_count: Dict[Tuple[int, int], int]) -> List[List[int]]:
        """グリーディアルゴリズムで卓を割り当て"""
        tables = []
        remaining_players = players.copy()
        
        while len(remaining_players) >= 4:
            # 最も同卓回数が少ないペアを優先して卓を作る
            if len(tables) == 0:
                # 最初の卓はランダムに選択
                table = self._select_initial_table(remaining_players, pair_count)
            else:
                # 次の卓は最小ペアを優先
                table = self._select_best_table(remaining_players, pair_count)
            
            tables.append(table)
            for p in table:
                remaining_players.remove(p)
        
        return tables
    
    def _select_initial_table(self, players: List[int], 
                            pair_count: Dict[Tuple[int, int], int]) -> List[int]:
        """最初の卓を選択（多様性のため）"""
        # ペア回数が最小のプレイヤーから開始
        min_pair_sum = float('inf')
        best_player = players[0]
        
        for p in players:
            pair_sum = sum(pair_count.get(tuple(sorted([p, other])), 0) 
                          for other in players if other != p)
            if pair_sum < min_pair_sum:
                min_pair_sum = pair_sum
                best_player = p
        
        # そのプレイヤーと最も同卓していない3人を選択
        candidates = [p for p in players if p != best_player]
        candidates.sort(key=lambda x: pair_count.get(tuple(sorted([best_player, x])), 0))
        
        return [best_player] + candidates[:3]
    
    def _select_best_table(self, players: List[int], 
                          pair_count: Dict[Tuple[int, int], int]) -> List[int]:
        """最適な卓を選択"""
        best_table = None
        best_score = float('inf')
        
        # 限定的な探索（計算時間を抑える）
        max_attempts = min(100, len(list(combinations(players, 4))))
        attempts = 0
        
        for table in combinations(players, 4):
            if attempts >= max_attempts:
                break
            attempts += 1
            
            # この卓のスコアを計算（ペア回数の合計）
            score = 0
            for p1, p2 in combinations(table, 2):
                pair = tuple(sorted([p1, p2]))
                score += pair_count.get(pair, 0) ** 2  # 2乗で重みづけ
            
            if score < best_score:
                best_score = score
                best_table = list(table)
        
        return best_table if best_table else list(players[:4])
    
    def _improve_solution(self, solution: List[List[List[int]]], 
                         initial_score: Dict) -> List[List[List[int]]]:
        """解を改善"""
        print("\n解の改善を開始...")
        
        best_solution = solution
        best_score = initial_score
        
        # 改善手法1: ラウンド単位の再最適化
        for round_idx in range(self.rounds):
            print(f"\rラウンド {round_idx + 1}/{self.rounds} を最適化中...", end='', flush=True)
            
            improved = self._optimize_single_round(best_solution, round_idx)
            improved_score = self._evaluate_solution(improved)
            
            if self._is_better_score(improved_score, best_score):
                best_solution = improved
                best_score = improved_score
        
        print()
        
        # 改善手法2: ペア交換による局所探索
        print("ペア交換による最適化中...")
        for _ in range(min(10, self.rounds)):
            improved = self._swap_optimization(best_solution)
            improved_score = self._evaluate_solution(improved)
            
            if self._is_better_score(improved_score, best_score):
                best_solution = improved
                best_score = improved_score
            else:
                break
        
        print(f"\n最終評価: {self._format_evaluation(best_score)}")
        
        return best_solution
    
    def _optimize_single_round(self, solution: List[List[List[int]]], 
                              round_idx: int) -> List[List[List[int]]]:
        """単一ラウンドを最適化"""
        # 現在の解をコピー
        new_solution = [round_tables.copy() for round_tables in solution]
        
        # 他のラウンドのペアカウントを計算
        pair_count = defaultdict(int)
        for i, round_tables in enumerate(new_solution):
            if i != round_idx:
                for table in round_tables:
                    if len(table) >= 4:
                        for p1, p2 in combinations(table, 2):
                            pair = tuple(sorted([p1, p2]))
                            pair_count[pair] += 1
        
        # このラウンドを再最適化
        round_tables = new_solution[round_idx]
        playing_players = []
        waiting_players = []
        
        for table in round_tables:
            if len(table) >= 4:
                playing_players.extend(table)
            else:
                waiting_players = table
        
        # 線形計画法で最適化（タイムアウト付き）
        optimized_tables = self._optimize_with_lp(playing_players, pair_count, round_idx)
        
        if optimized_tables:
            new_solution[round_idx] = optimized_tables
            if waiting_players:
                new_solution[round_idx].append(waiting_players)
        
        return new_solution
    
    def _optimize_with_lp(self, players: List[int], 
                         existing_pairs: Dict[Tuple[int, int], int],
                         round_num: int) -> Optional[List[List[int]]]:
        """線形計画法による最適化（タイムアウト付き）"""
        if len(players) > 20:
            # 大規模すぎる場合はスキップ
            return None
        
        prob = pulp.LpProblem(f"Round_{round_num}", pulp.LpMinimize)
        
        # 可能な卓
        possible_tables = list(combinations(players, 4))
        if self.allow_five and len(players) >= 5:
            possible_tables.extend(list(combinations(players, 5)))
        
        # サンプリング（計算時間短縮）
        if len(possible_tables) > 500:
            possible_tables = random.sample(possible_tables, 500)
        
        # 決定変数
        table_vars = {}
        for i, table in enumerate(possible_tables):
            table_vars[i] = pulp.LpVariable(f"table_{i}", cat='Binary')
        
        # 目的関数
        objective = 0
        for i, table in enumerate(possible_tables):
            table_score = 0
            for p1, p2 in combinations(table, 2):
                pair = tuple(sorted([p1, p2]))
                count = existing_pairs.get(pair, 0)
                
                # 優先度に基づくスコア
                if count == 0 and self.ideal_meetings_floor > 0:
                    table_score -= 10000  # 未実現ペアを優先
                else:
                    # 理想値との差
                    ideal = self.ideal_meetings_float * (round_num + 1) / self.rounds
                    table_score += abs(count + 1 - ideal) * 100
            
            objective += table_vars[i] * table_score
        
        prob += objective
        
        # 制約：各プレイヤーは1つの卓のみ
        for player in players:
            player_tables = [table_vars[i] for i, table in enumerate(possible_tables) if player in table]
            if player_tables:
                prob += pulp.lpSum(player_tables) == 1
        
        # 解く（タイムアウト付き）
        try:
            prob.solve(pulp.PULP_CBC_CMD(msg=0, timeLimit=self.timeout_per_round))
            
            if prob.status == pulp.LpStatusOptimal:
                result_tables = []
                for i, table in enumerate(possible_tables):
                    if table_vars[i].varValue > 0.5:
                        result_tables.append(list(table))
                return result_tables
        except:
            pass
        
        return None
    
    def _swap_optimization(self, solution: List[List[List[int]]]) -> List[List[List[int]]]:
        """ペア交換による最適化"""
        new_solution = [round_tables.copy() for round_tables in solution]
        
        # ランダムに2つのラウンドを選択
        if self.rounds < 2:
            return new_solution
        
        round1, round2 = random.sample(range(self.rounds), 2)
        
        # 各ラウンドからランダムに卓を選択
        tables1 = [t for t in new_solution[round1] if len(t) >= 4]
        tables2 = [t for t in new_solution[round2] if len(t) >= 4]
        
        if not tables1 or not tables2:
            return new_solution
        
        table1_idx = random.randint(0, len(tables1) - 1)
        table2_idx = random.randint(0, len(tables2) - 1)
        
        # プレイヤーを交換してみる
        player1 = random.choice(tables1[table1_idx])
        player2 = random.choice(tables2[table2_idx])
        
        # 交換可能かチェック
        if player1 not in tables2[table2_idx] and player2 not in tables1[table1_idx]:
            # 交換
            new_table1 = [p if p != player1 else player2 for p in tables1[table1_idx]]
            new_table2 = [p if p != player2 else player1 for p in tables2[table2_idx]]
            
            # 更新
            new_solution[round1][table1_idx] = new_table1
            new_solution[round2][table2_idx] = new_table2
        
        return new_solution
    
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
    
    def _format_evaluation(self, score: Dict) -> str:
        """評価を文字列化"""
        return (f"最小{score['min_count']}回, "
                f"最大{score['max_count']}回, "
                f"最小ペア{score['distribution'].get(score['min_count'], 0)}個, "
                f"最大ペア{score['distribution'].get(score['max_count'], 0)}個")
    
    def print_results(self, results: List[List[List[int]]]):
        """結果を見やすく出力"""
        print(f"\n麻雀卓組結果（最適実装版） (参加者: {self.players}人, {self.rounds}回戦)")
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
        print(f"2. 同卓回数の最大: {max_count}回 {'✓' if max_count - min_count <= 1 else '△'}")
        print(f"3. 最小回数のペア数: {count_distribution[min_count]}ペア")
        print(f"4. 最大回数のペア数: {count_distribution[max_count]}ペア")
        
        # 理論値との比較
        print(f"\n理論値との比較:")
        print(f"- 理想的な平均同卓回数: {self.ideal_meetings_float:.2f}回")
        print(f"- 実際の平均との差: {abs(mean_count - self.ideal_meetings_float):.2f}回")


def main():
    parser = argparse.ArgumentParser(description='麻雀卓組生成プログラム（最適実装版）')
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
    
    generator = OptimalTableGroupGenerator(args.players, args.rounds, args.five)
    results = generator.generate()
    generator.print_results(results)


if __name__ == "__main__":
    main()