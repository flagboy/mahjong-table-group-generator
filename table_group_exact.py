#!/usr/bin/env python3
"""麻雀卓組生成プログラム（完全最適化版）- 列挙探索と制約プログラミング"""

import argparse
from typing import List, Dict, Tuple, Set, Optional
from collections import defaultdict
from itertools import combinations, product
import pulp
import math
import time
from functools import lru_cache


class ExactTableGroupGenerator:
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
        
        # 最適化方法の選択
        self.use_enumeration = self._should_use_enumeration()
        
        # 理論的な制約を計算
        self._calculate_theoretical_limits()
        
    def _should_use_enumeration(self) -> bool:
        """列挙探索を使用すべきか判定"""
        # 小規模問題の場合は列挙探索
        if self.players <= 8 and self.rounds <= 4:
            return True
        
        # 待機なしの完全な問題も列挙可能
        if self.players == 8 and self.rounds <= 6:
            return True
        
        return False
    
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
            self.optimal_table_config = {'four_tables': tables_per_round, 'five_tables': 0}
        
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
        print(f"\n最適化方法: {'完全列挙探索' if self.use_enumeration else '制約プログラミング'}")
    
    def _find_best_table_configuration(self) -> Dict:
        """5人打ちありの場合の最適な卓構成を見つける"""
        best_config = None
        max_pairs = 0
        
        for five_tables in range(self.players // 5 + 1):
            remaining = self.players - five_tables * 5
            if remaining >= 0 and remaining % 4 == 0:
                four_tables = remaining // 4
                pairs = five_tables * 10 + four_tables * 6
                if pairs > max_pairs:
                    max_pairs = pairs
                    best_config = {
                        'five_tables': five_tables,
                        'four_tables': four_tables,
                        'pairs_per_round': pairs
                    }
        
        return best_config
    
    def generate(self) -> List[List[List[int]]]:
        """全ラウンドの卓組を生成"""
        if self.use_enumeration:
            print("\n完全列挙探索を開始...")
            return self._solve_by_enumeration()
        else:
            print("\n制約プログラミングによる最適化を開始...")
            return self._solve_by_constraint_programming()
    
    def _solve_by_enumeration(self) -> List[List[List[int]]]:
        """完全列挙探索による解法"""
        start_time = time.time()
        
        # すべての可能な卓構成を生成
        all_round_configs = self._enumerate_all_round_configurations()
        print(f"1ラウンドあたり{len(all_round_configs)}通りの構成")
        
        # 実現可能な組み合わせ数を計算
        total_combinations = len(all_round_configs) ** self.rounds
        print(f"全体で{total_combinations}通りの組み合わせ")
        
        if total_combinations > 1000000:
            print("組み合わせが多すぎるため、制約プログラミングに切り替えます")
            self.use_enumeration = False
            return self._solve_by_constraint_programming()
        
        # すべての組み合わせを評価
        best_solution = None
        best_score = float('-inf')
        
        # プログレスバー用
        evaluated = 0
        print_interval = max(1, total_combinations // 100)
        
        # すべての組み合わせを生成して評価
        for combination in product(all_round_configs, repeat=self.rounds):
            evaluated += 1
            
            if evaluated % print_interval == 0:
                progress = evaluated / total_combinations * 100
                print(f"\r進捗: {progress:.1f}%", end='', flush=True)
            
            # この組み合わせを評価
            score = self._evaluate_combination(combination)
            
            if score > best_score:
                best_score = score
                best_solution = combination
        
        print(f"\n完了！ (所要時間: {time.time() - start_time:.2f}秒)")
        print(f"最良スコア: {best_score}")
        
        # 最良解を返す
        return list(best_solution)
    
    def _enumerate_all_round_configurations(self) -> List[List[List[int]]]:
        """1ラウンドの全ての可能な構成を列挙"""
        configurations = []
        
        if self.players % 4 == 0 and not self.allow_five:
            # 待機なしの場合（4の倍数）
            configurations = self._enumerate_perfect_configurations()
        else:
            # 待機ありの場合
            configurations = self._enumerate_with_waiting()
        
        return configurations
    
    def _enumerate_perfect_configurations(self) -> List[List[List[int]]]:
        """待機なしの完全な構成を列挙"""
        configs = []
        n_tables = self.players // 4
        
        # 再帰的に卓を構成
        def generate_tables(remaining_players, current_tables):
            if len(current_tables) == n_tables:
                configs.append(current_tables[:])
                return
            
            if len(remaining_players) < 4:
                return
            
            # 最初のプレイヤーを含む4人の組み合わせを試す
            first_player = remaining_players[0]
            for other_three in combinations(remaining_players[1:], 3):
                table = [first_player] + list(other_three)
                new_remaining = [p for p in remaining_players if p not in table]
                generate_tables(new_remaining, current_tables + [table])
        
        generate_tables(self.player_ids, [])
        
        # 重複を除去（卓の順序は関係ない）
        unique_configs = []
        seen = set()
        
        for config in configs:
            # 構成を正規化
            normalized = tuple(sorted(tuple(sorted(table)) for table in config))
            if normalized not in seen:
                seen.add(normalized)
                unique_configs.append(config)
        
        return unique_configs
    
    def _enumerate_with_waiting(self) -> List[List[List[int]]]:
        """待機ありの構成を列挙"""
        configs = []
        waiting_count = self.players % 4
        
        # 待機するプレイヤーの組み合わせ
        for waiting_players in combinations(self.player_ids, waiting_count):
            playing_players = [p for p in self.player_ids if p not in waiting_players]
            
            # プレイするプレイヤーで卓を構成
            table_configs = self._enumerate_perfect_configurations_for_players(playing_players)
            
            # 待機プレイヤーを追加
            for config in table_configs:
                full_config = config + [list(waiting_players)]
                configs.append(full_config)
        
        return configs
    
    def _enumerate_perfect_configurations_for_players(self, players: List[int]) -> List[List[List[int]]]:
        """特定のプレイヤーセットに対する完全な構成を列挙"""
        if len(players) % 4 != 0:
            return []
        
        configs = []
        n_tables = len(players) // 4
        
        def generate_tables(remaining_players, current_tables):
            if len(current_tables) == n_tables:
                configs.append(current_tables[:])
                return
            
            if len(remaining_players) < 4:
                return
            
            first_player = remaining_players[0]
            for other_three in combinations(remaining_players[1:], 3):
                table = [first_player] + list(other_three)
                new_remaining = [p for p in remaining_players if p not in table]
                generate_tables(new_remaining, current_tables + [table])
        
        generate_tables(players, [])
        return configs
    
    def _evaluate_combination(self, combination: Tuple[List[List[int]], ...]) -> float:
        """組み合わせを評価"""
        # ペアカウントを計算
        pair_count = defaultdict(int)
        for round_config in combination:
            for table in round_config:
                if len(table) >= 4:
                    for p1, p2 in combinations(table, 2):
                        pair = tuple(sorted([p1, p2]))
                        pair_count[pair] += 1
        
        # 評価基準に基づくスコア計算
        all_counts = [pair_count.get(pair, 0) for pair in self.all_pairs]
        
        min_count = min(all_counts)
        max_count = max(all_counts)
        
        # カウント分布
        count_distribution = defaultdict(int)
        for count in all_counts:
            count_distribution[count] += 1
        
        # スコア計算（優先順位順）
        score = 0
        
        # 1. 同卓回数の最小が大きい（最重要）
        score += min_count * 10000000
        
        # 2. 同卓回数の最大が小さい
        score -= max_count * 100000
        
        # 3. 最小回数のペア数が少ない
        score -= count_distribution[min_count] * 1000
        
        # 4. 最大回数のペア数が少ない
        score -= count_distribution[max_count] * 10
        
        # 追加: 理想分配に近い
        for count, ideal_pairs in self.target_distribution.items():
            actual_pairs = count_distribution.get(count, 0)
            score -= abs(ideal_pairs - actual_pairs) * 100
        
        return score
    
    def _solve_by_constraint_programming(self) -> List[List[List[int]]]:
        """制約プログラミングによる完全最適化"""
        print("制約プログラミングモデルを構築中...")
        
        # 問題の設定
        prob = pulp.LpProblem("ExactTableGrouping", pulp.LpMaximize)
        
        # すべての可能な卓
        all_tables = []
        table_to_pairs = {}
        
        # 4人卓
        for table in combinations(self.player_ids, 4):
            all_tables.append(table)
            table_to_pairs[table] = list(combinations(table, 2))
        
        # 5人卓（許可されている場合）
        if self.allow_five:
            for table in combinations(self.player_ids, 5):
                all_tables.append(table)
                table_to_pairs[table] = list(combinations(table, 2))
        
        print(f"可能な卓数: {len(all_tables)}")
        
        # 決定変数：各ラウンドで各卓を使用するか
        table_vars = {}
        for r in range(self.rounds):
            for i, table in enumerate(all_tables):
                table_vars[(r, i)] = pulp.LpVariable(f"table_r{r}_t{i}", cat='Binary')
        
        # ペアの同卓回数を追跡する変数
        pair_meetings = {}
        for pair in self.all_pairs:
            pair_meetings[pair] = pulp.LpVariable(
                f"meetings_{pair[0]}_{pair[1]}", 
                lowBound=0, 
                cat='Integer'
            )
        
        # 最小・最大同卓回数の変数
        min_meetings = pulp.LpVariable("min_meetings", lowBound=0, cat='Integer')
        max_meetings = pulp.LpVariable("max_meetings", lowBound=0, cat='Integer')
        
        # ペア数カウント用の変数
        count_vars = {}
        for c in range(self.rounds + 1):
            count_vars[c] = pulp.LpVariable(f"count_{c}", lowBound=0, cat='Integer')
        
        # 目的関数：階層的最適化
        objective = 0
        
        # 優先度1: 最小同卓回数を最大化
        objective += min_meetings * 10000000
        
        # 優先度2: 最大同卓回数を最小化
        objective -= max_meetings * 100000
        
        # 優先度3: 最小回数のペア数を最小化
        objective -= count_vars.get(min_meetings, 0) * 1000
        
        # 優先度4: 最大回数のペア数を最小化
        objective -= count_vars.get(max_meetings, 0) * 10
        
        prob += objective
        
        # 制約1: 各ラウンドで各プレイヤーは最大1つの卓に入る
        for r in range(self.rounds):
            for player in self.player_ids:
                player_tables = []
                for i, table in enumerate(all_tables):
                    if player in table:
                        player_tables.append(table_vars[(r, i)])
                
                prob += pulp.lpSum(player_tables) <= 1
        
        # 制約2: ペアの同卓回数を正しくカウント
        for pair in self.all_pairs:
            pair_sum = []
            for r in range(self.rounds):
                for i, table in enumerate(all_tables):
                    if pair[0] in table and pair[1] in table:
                        pair_sum.append(table_vars[(r, i)])
            
            prob += pair_meetings[pair] == pulp.lpSum(pair_sum)
        
        # 制約3: 最小・最大同卓回数の定義
        for pair in self.all_pairs:
            prob += pair_meetings[pair] >= min_meetings
            prob += pair_meetings[pair] <= max_meetings
        
        # 制約4: 各同卓回数のペア数をカウント
        for c in range(self.rounds + 1):
            # バイナリ変数：各ペアがc回同卓するか
            is_count_c = {}
            for pair in self.all_pairs:
                is_count_c[pair] = pulp.LpVariable(
                    f"is_{pair[0]}_{pair[1]}_count_{c}", 
                    cat='Binary'
                )
                
                # pair_meetings[pair] == c の場合のみ is_count_c[pair] = 1
                prob += pair_meetings[pair] >= c - self.rounds * (1 - is_count_c[pair])
                prob += pair_meetings[pair] <= c + self.rounds * (1 - is_count_c[pair])
            
            # c回同卓するペア数をカウント
            prob += count_vars[c] == pulp.lpSum(is_count_c.values())
        
        # 制約5: 各ラウンドで適切な数の卓を使用
        for r in range(self.rounds):
            if not self.allow_five and self.players % 4 == 0:
                # 4人打ちのみで割り切れる場合
                table_count = pulp.lpSum([
                    table_vars[(r, i)] 
                    for i in range(len(all_tables))
                ])
                prob += table_count == self.players // 4
        
        # 最適化を実行
        print("最適化を実行中...")
        solver = pulp.PULP_CBC_CMD(msg=1, timeLimit=300)  # 5分のタイムリミット
        prob.solve(solver)
        
        print(f"\n最適化状態: {pulp.LpStatus[prob.status]}")
        
        # 結果を構築
        results = []
        for r in range(self.rounds):
            round_tables = []
            used_players = set()
            
            for i, table in enumerate(all_tables):
                if table_vars[(r, i)].varValue > 0.5:
                    round_tables.append(list(table))
                    used_players.update(table)
            
            # 待機プレイヤー
            waiting = [p for p in self.player_ids if p not in used_players]
            if waiting:
                round_tables.append(waiting)
            
            results.append(round_tables)
        
        # 結果の統計を表示
        print(f"\n最適化結果:")
        print(f"- 最小同卓回数: {int(min_meetings.varValue)}回")
        print(f"- 最大同卓回数: {int(max_meetings.varValue)}回")
        
        return results
    
    def print_results(self, results: List[List[List[int]]]):
        """結果を見やすく出力"""
        print(f"\n麻雀卓組結果（完全最適化版） (参加者: {self.players}人, {self.rounds}回戦)")
        print(f"5人打ち: {'あり' if self.allow_five else 'なし'}")
        print(f"最適化方法: {'完全列挙探索' if self.use_enumeration else '制約プログラミング'}")
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
    parser = argparse.ArgumentParser(description='麻雀卓組生成プログラム（完全最適化版）')
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
    
    generator = ExactTableGroupGenerator(args.players, args.rounds, args.five)
    results = generator.generate()
    generator.print_results(results)


if __name__ == "__main__":
    main()