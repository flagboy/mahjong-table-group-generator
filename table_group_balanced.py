#!/usr/bin/env python3
"""麻雀卓組生成プログラム（均衡最適化版）- 同卓回数の均等分配を重視"""

import argparse
from typing import List, Dict, Tuple, Set, Optional
from collections import defaultdict
from itertools import combinations
import pulp
import math
import time


class BalancedTableGroupGenerator:
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
            # 5人打ちありの場合の最適配置
            best_config = self._find_best_table_configuration()
            self.max_pairs_per_round = best_config['pairs_per_round']
            self.optimal_table_config = best_config
        else:
            # 4人打ちのみ
            tables_per_round = self.players // 4
            self.max_pairs_per_round = tables_per_round * 6
            self.optimal_table_config = {'four_tables': tables_per_round, 'five_tables': 0}
        
        self.max_total_pairs = self.max_pairs_per_round * self.rounds
        
        # 各ペアの理想的な同卓回数
        self.ideal_meetings_float = self.max_total_pairs / self.total_pairs
        self.ideal_meetings_floor = int(self.ideal_meetings_float)
        self.ideal_meetings_ceil = self.ideal_meetings_floor + 1
        
        # 理想的な分配
        # floor回のペア数とceil回のペア数を計算
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
        print(f"\n目標分配:")
        print(f"- {self.ideal_meetings_floor}回同卓: {floor_pairs}ペア")
        print(f"- {self.ideal_meetings_ceil}回同卓: {ceil_pairs}ペア")
        
        if self.ideal_meetings_floor == 0:
            print(f"\n警告: 全ペアを最低1回同卓させるには{math.ceil(self.total_pairs / self.max_pairs_per_round)}回戦以上必要です")
    
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
        print("\n均衡最適化を開始...")
        
        # グローバル最適化問題として解く
        if self.rounds <= 10 and self.players <= 16:
            # 小規模問題は完全最適化
            return self._solve_global_optimization()
        else:
            # 大規模問題は段階的最適化
            return self._solve_phased_optimization()
    
    def _solve_global_optimization(self) -> List[List[List[int]]]:
        """全ラウンドを同時に最適化（小規模問題用）"""
        print("完全最適化を実行中...")
        
        # 最適化問題の設定
        prob = pulp.LpProblem("BalancedTableGrouping", pulp.LpMinimize)
        
        # 各ラウンドの可能な卓構成を生成
        round_configs = []
        for r in range(self.rounds):
            configs = self._generate_all_valid_configurations()
            round_configs.append(configs)
        
        # 決定変数：各ラウンドでどの構成を選ぶか
        config_vars = {}
        for r in range(self.rounds):
            for c, config in enumerate(round_configs[r]):
                config_vars[(r, c)] = pulp.LpVariable(f"config_{r}_{c}", cat='Binary')
        
        # ペアの同卓回数を追跡する変数
        pair_meetings = {}
        for pair in self.all_pairs:
            pair_meetings[pair] = pulp.LpVariable(f"meetings_{pair[0]}_{pair[1]}", 
                                                  lowBound=0, cat='Integer')
        
        # 目的関数：理想分配からの偏差を最小化
        objective = 0
        
        # 優先度1: 最小同卓回数を最大化（最重要）
        min_meetings = pulp.LpVariable("min_meetings", lowBound=0, cat='Integer')
        for pair in self.all_pairs:
            prob += pair_meetings[pair] >= min_meetings
        objective -= min_meetings * 1000000
        
        # 優先度2: 最大同卓回数を最小化
        max_meetings = pulp.LpVariable("max_meetings", lowBound=0, cat='Integer')
        for pair in self.all_pairs:
            prob += pair_meetings[pair] <= max_meetings
        objective += max_meetings * 10000
        
        # 優先度3: 理想分配からの偏差を最小化
        for pair in self.all_pairs:
            # 理想値からの絶対偏差
            deviation_pos = pulp.LpVariable(f"dev_pos_{pair[0]}_{pair[1]}", lowBound=0)
            deviation_neg = pulp.LpVariable(f"dev_neg_{pair[0]}_{pair[1]}", lowBound=0)
            
            prob += pair_meetings[pair] - self.ideal_meetings_float <= deviation_pos
            prob += self.ideal_meetings_float - pair_meetings[pair] <= deviation_neg
            
            objective += (deviation_pos + deviation_neg) * 100
        
        prob += objective
        
        # 制約1: 各ラウンドで1つの構成を選択
        for r in range(self.rounds):
            config_sum = pulp.lpSum([config_vars[(r, c)] 
                                    for c in range(len(round_configs[r]))])
            prob += config_sum == 1
        
        # 制約2: ペアの同卓回数を正しくカウント
        for pair in self.all_pairs:
            total_meetings = []
            for r in range(self.rounds):
                for c, config in enumerate(round_configs[r]):
                    if self._config_contains_pair(config, pair):
                        total_meetings.append(config_vars[(r, c)])
            
            prob += pair_meetings[pair] == pulp.lpSum(total_meetings)
        
        # 最適化を実行
        prob.solve(pulp.PULP_CBC_CMD(msg=0, timeLimit=30))
        
        # 結果を構築
        if prob.status == pulp.LpStatusOptimal:
            print("最適解が見つかりました")
        else:
            print("準最適解を使用します")
        
        results = []
        for r in range(self.rounds):
            selected_config = None
            for c, config in enumerate(round_configs[r]):
                if config_vars.get((r, c)) and config_vars[(r, c)].varValue > 0.5:
                    selected_config = config
                    break
            
            if selected_config:
                results.append(selected_config)
            else:
                # フォールバック
                results.append(round_configs[r][0])
        
        return results
    
    def _solve_phased_optimization(self) -> List[List[List[int]]]:
        """段階的最適化（大規模問題用）"""
        print("段階的最適化を実行中...")
        
        all_rounds = []
        pair_count = defaultdict(int)
        
        for round_num in range(self.rounds):
            print(f"\rラウンド {round_num + 1}/{self.rounds} を最適化中...", end='', flush=True)
            
            # このラウンドの最適化問題
            prob = pulp.LpProblem(f"Round_{round_num + 1}", pulp.LpMinimize)
            
            # 可能な卓の組み合わせ
            possible_tables = self._generate_possible_tables()
            
            # 決定変数
            table_vars = {}
            for i, table in enumerate(possible_tables):
                table_vars[i] = pulp.LpVariable(f"table_{i}", cat='Binary')
            
            # 目的関数：均衡を重視
            objective = 0
            
            # 現在の進捗
            progress = (round_num + 1) / self.rounds
            
            for i, table in enumerate(possible_tables):
                table_score = 0
                
                for p1, p2 in combinations(table, 2):
                    pair = tuple(sorted([p1, p2]))
                    current_count = pair_count.get(pair, 0)
                    
                    # 現時点での理想的な同卓回数
                    ideal_now = self.ideal_meetings_float * progress
                    
                    # 偏差を計算
                    deviation_before = abs(current_count - ideal_now)
                    deviation_after = abs(current_count + 1 - ideal_now)
                    
                    # 改善度合い
                    improvement = deviation_before - deviation_after
                    
                    # 未実現ペアは最優先
                    if current_count == 0 and self.ideal_meetings_floor > 0:
                        table_score += 1000000
                    else:
                        # 改善する場合はプラス、悪化する場合はマイナス
                        table_score += improvement * 1000
                    
                    # 過度な同卓にペナルティ
                    if current_count >= self.ideal_meetings_ceil:
                        table_score -= 10000 * (current_count - self.ideal_meetings_ceil + 1)
                
                objective -= table_vars[i] * table_score
            
            prob += objective
            
            # 制約を追加
            self._add_constraints(prob, table_vars, possible_tables)
            
            # 解く
            prob.solve(pulp.PULP_CBC_CMD(msg=0))
            
            # 結果を取得
            round_tables = self._extract_solution(table_vars, possible_tables)
            all_rounds.append(round_tables)
            
            # ペアカウントを更新
            for table in round_tables:
                if len(table) >= 4:
                    for p1, p2 in combinations(table, 2):
                        pair = tuple(sorted([p1, p2]))
                        pair_count[pair] += 1
        
        print()  # 改行
        return all_rounds
    
    def _generate_all_valid_configurations(self) -> List[List[List[int]]]:
        """すべての有効な卓構成を生成（小規模用）"""
        import itertools
        
        configs = []
        all_tables = list(combinations(self.player_ids, 4))
        if self.allow_five:
            all_tables.extend(list(combinations(self.player_ids, 5)))
        
        # 制限付きで構成を生成
        max_configs = min(1000, len(all_tables) ** 2)  # 最大1000構成
        
        for _ in range(max_configs):
            config = self._generate_random_valid_config()
            if config and self._is_valid_config(config):
                configs.append(config)
        
        # 重複を除去
        unique_configs = []
        seen = set()
        for config in configs:
            # 構成を正規化（ソート）してハッシュ可能にする
            normalized = tuple(sorted(tuple(sorted(table)) for table in config))
            if normalized not in seen:
                seen.add(normalized)
                unique_configs.append(config)
        
        return unique_configs[:100]  # 最大100構成
    
    def _is_valid_config(self, config: List[List[int]]) -> bool:
        """構成が有効かチェック"""
        # すべてのプレイヤーが使用されているか
        used_players = set()
        for table in config:
            if len(table) >= 4:
                used_players.update(table)
        
        # 待機人数が適切か
        waiting = self.players - len(used_players)
        if not self.allow_five:
            return waiting == self.players % 4
        else:
            return waiting <= 3  # 5人打ちありなら最大3人待機
    
    def _generate_random_valid_config(self) -> List[List[int]]:
        """有効なランダム構成を生成"""
        import random
        players = self.player_ids.copy()
        random.shuffle(players)
        
        config = []
        
        if self.allow_five and hasattr(self, 'optimal_table_config'):
            # 最適な卓構成を使用
            for _ in range(self.optimal_table_config['five_tables']):
                if len(players) >= 5:
                    config.append(players[:5])
                    players = players[5:]
            
            for _ in range(self.optimal_table_config['four_tables']):
                if len(players) >= 4:
                    config.append(players[:4])
                    players = players[4:]
        else:
            # 4人卓のみ
            while len(players) >= 4:
                config.append(players[:4])
                players = players[4:]
        
        if players:
            config.append(players)
        
        return config
    
    def _config_contains_pair(self, config: List[List[int]], pair: Tuple[int, int]) -> bool:
        """構成がペアを含むかチェック"""
        for table in config:
            if len(table) >= 4 and pair[0] in table and pair[1] in table:
                return True
        return False
    
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
        
        # 制約2: 適切な数の卓を選択
        if self.allow_five and hasattr(self, 'optimal_table_config'):
            # 最適な卓数に近づける
            four_tables = pulp.lpSum([table_vars[i] for i, t in enumerate(possible_tables) if len(t) == 4])
            five_tables = pulp.lpSum([table_vars[i] for i, t in enumerate(possible_tables) if len(t) == 5])
            
            # ソフト制約として扱う
            prob += four_tables >= self.optimal_table_config['four_tables'] - 1
            prob += four_tables <= self.optimal_table_config['four_tables'] + 1
            prob += five_tables >= self.optimal_table_config['five_tables'] - 1
            prob += five_tables <= self.optimal_table_config['five_tables'] + 1
        else:
            # 4人卓のみの場合
            total_tables = pulp.lpSum(table_vars.values())
            prob += total_tables == self.players // 4
    
    def _extract_solution(self, table_vars: Dict[int, pulp.LpVariable],
                         possible_tables: List[Tuple[int, ...]]) -> List[List[int]]:
        """最適化結果から解を抽出"""
        round_tables = []
        used_players = set()
        
        for i, table in enumerate(possible_tables):
            if table_vars[i].varValue > 0.5:
                round_tables.append(list(table))
                used_players.update(table)
        
        # 待機プレイヤー
        waiting = [p for p in self.player_ids if p not in used_players]
        if waiting:
            round_tables.append(waiting)
        
        return round_tables
    
    def print_results(self, results: List[List[List[int]]]):
        """結果を見やすく出力"""
        print(f"\n麻雀卓組結果（均衡最適化版） (参加者: {self.players}人, {self.rounds}回戦)")
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
        
        # 理想分配との比較
        print(f"\n理想分配との比較:")
        for count, ideal_pairs in self.target_distribution.items():
            actual_pairs = count_distribution.get(count, 0)
            print(f"- {count}回同卓: 理想{ideal_pairs}ペア → 実際{actual_pairs}ペア (差{abs(ideal_pairs - actual_pairs)})")


def main():
    parser = argparse.ArgumentParser(description='麻雀卓組生成プログラム（均衡最適化版）')
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
    
    generator = BalancedTableGroupGenerator(args.players, args.rounds, args.five)
    results = generator.generate()
    generator.print_results(results)


if __name__ == "__main__":
    main()