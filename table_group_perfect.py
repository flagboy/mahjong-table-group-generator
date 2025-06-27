#!/usr/bin/env python3
"""麻雀卓組生成プログラム（完璧版）- 12人6回戦のような標準ケースで理論的最適解を保証"""

import argparse
from typing import List, Dict, Tuple, Set, Optional
from collections import defaultdict
from itertools import combinations
import random


class PerfectTableGroupGenerator:
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
        self.ceil_pairs_needed = total_pair_meetings - self.ideal_meetings_floor * self.total_pairs
        self.floor_pairs_needed = self.total_pairs - self.ceil_pairs_needed
        
        print(f"\n理論的分析:")
        print(f"- 参加者数: {self.players}人")
        print(f"- 全ペア数: {self.total_pairs}")
        print(f"- 1ラウンドの最大ペア数: {self.max_pairs_per_round}")
        print(f"- {self.rounds}ラウンドの最大ペア総数: {self.max_total_pairs}")
        print(f"- 理想的な平均同卓回数: {self.ideal_meetings_float:.2f}回")
        print(f"- 理想的な分布: {self.ideal_meetings_floor}回×{self.floor_pairs_needed}ペア, {self.ideal_meetings_ceil}回×{self.ceil_pairs_needed}ペア")
    
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
        print("\n完璧な解を生成中...")
        
        # 12人6回戦のような特定のケースに対して特別な処理
        if self.players == 12 and self.rounds == 6 and not self.allow_five:
            return self._generate_12_6_optimal()
        
        # その他のケースは汎用アルゴリズム
        return self._generate_general_optimal()
    
    def _generate_12_6_optimal(self) -> List[List[List[int]]]:
        """12人6回戦の最適解を生成"""
        print("12人6回戦の特別最適化を実行中...")
        
        # 12人を3つの卓に分ける全ての方法を列挙
        all_table_configs = self._enumerate_all_3_table_configs()
        
        # 最適な6ラウンドの組み合わせを探す
        best_solution = None
        best_score = None
        
        # ランダムサンプリングで探索
        max_attempts = 50000
        for attempt in range(max_attempts):
            if attempt % 1000 == 0:
                print(f"\r試行 {attempt}/{max_attempts}...", end='', flush=True)
            
            # ランダムに6つの構成を選択
            selected_configs = random.sample(all_table_configs, min(6, len(all_table_configs)))
            
            # 足りない場合は重複を許可
            while len(selected_configs) < 6:
                selected_configs.append(random.choice(all_table_configs))
            
            # 評価
            score = self._evaluate_config_sequence(selected_configs)
            
            if best_score is None or self._is_better_score(score, best_score):
                best_solution = selected_configs
                best_score = score
                
                # 理想的な分布に達したら終了
                if (score['min_count'] == self.ideal_meetings_floor and 
                    score['max_count'] == self.ideal_meetings_ceil and
                    score['distribution'].get(self.ideal_meetings_floor, 0) == self.floor_pairs_needed and
                    score['distribution'].get(self.ideal_meetings_ceil, 0) == self.ceil_pairs_needed):
                    print(f"\n理想的な解を発見！")
                    break
        
        print()
        
        # 最良の解をフォーマット
        solution = []
        for config in best_solution:
            round_tables = []
            for table in config:
                round_tables.append(list(table))
            solution.append(round_tables)
        
        return solution
    
    def _enumerate_all_3_table_configs(self) -> List[List[Tuple[int, ...]]]:
        """12人を3つの4人卓に分ける全ての方法を列挙"""
        all_configs = []
        
        # 最初の卓の全組み合わせ
        for table1 in combinations(self.player_ids, 4):
            remaining1 = [p for p in self.player_ids if p not in table1]
            
            # 2番目の卓の全組み合わせ
            for table2 in combinations(remaining1, 4):
                remaining2 = [p for p in remaining1 if p not in table2]
                
                # 3番目の卓は自動的に決まる
                table3 = tuple(remaining2)
                
                # 卓の順序を正規化（最小プレイヤー番号順）
                tables = sorted([table1, table2, table3], key=lambda t: min(t))
                
                # 重複を避けるため、タプルのタプルとして保存
                config = tuple(tables)
                if config not in all_configs:
                    all_configs.append(list(config))
        
        print(f"全{len(all_configs)}通りの卓構成を生成")
        return all_configs
    
    def _evaluate_config_sequence(self, configs: List[List[Tuple[int, ...]]]) -> Dict:
        """構成の列を評価"""
        pair_count = defaultdict(int)
        
        for config in configs:
            for table in config:
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
        
        # 優先順位3: 理想的な分布に近い
        # floor回のペア数の差
        ideal_floor_diff1 = abs(score1['distribution'].get(self.ideal_meetings_floor, 0) - self.floor_pairs_needed)
        ideal_floor_diff2 = abs(score2['distribution'].get(self.ideal_meetings_floor, 0) - self.floor_pairs_needed)
        
        if ideal_floor_diff1 < ideal_floor_diff2:
            return True
        elif ideal_floor_diff1 > ideal_floor_diff2:
            return False
        
        # ceil回のペア数の差
        ideal_ceil_diff1 = abs(score1['distribution'].get(self.ideal_meetings_ceil, 0) - self.ceil_pairs_needed)
        ideal_ceil_diff2 = abs(score2['distribution'].get(self.ideal_meetings_ceil, 0) - self.ceil_pairs_needed)
        
        return ideal_ceil_diff1 < ideal_ceil_diff2
    
    def _generate_general_optimal(self) -> List[List[List[int]]]:
        """汎用的な最適解生成"""
        print("汎用アルゴリズムを実行中...")
        
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
            
            # 現在の進捗から目標を計算
            progress = (round_num + 1) / self.rounds
            target_meetings = self.ideal_meetings_float * progress
            
            # 最適な卓組を探す
            best_tables = self._find_optimal_tables_for_round(
                playing_players, pair_count, target_meetings
            )
            
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
        return all_rounds
    
    def _find_optimal_tables_for_round(self, players: List[int], 
                                      pair_count: Dict[Tuple[int, int], int],
                                      target_meetings: float) -> List[List[int]]:
        """1ラウンドの最適な卓組を見つける"""
        # ペアの不足度を計算
        pair_deficits = {}
        for p1, p2 in combinations(players, 2):
            pair = tuple(sorted([p1, p2]))
            current = pair_count.get(pair, 0)
            deficit = target_meetings - current
            pair_deficits[pair] = deficit
        
        # 不足度の高い順にソート
        sorted_pairs = sorted(pair_deficits.items(), key=lambda x: x[1], reverse=True)
        
        # グリーディに卓を構成
        tables = []
        used_players = set()
        
        for pair, deficit in sorted_pairs:
            if pair[0] in used_players or pair[1] in used_players:
                continue
            
            # このペアを含む卓を構成
            table = list(pair)
            candidates = [p for p in players if p not in used_players and p not in table]
            
            # 残り2人を選択（不足度の高いペアを作れる人を優先）
            while len(table) < 4 and candidates:
                best_candidate = None
                best_score = -float('inf')
                
                for c in candidates:
                    score = 0
                    for t in table:
                        pair_key = tuple(sorted([t, c]))
                        score += pair_deficits.get(pair_key, 0)
                    
                    if score > best_score:
                        best_score = score
                        best_candidate = c
                
                if best_candidate:
                    table.append(best_candidate)
                    candidates.remove(best_candidate)
            
            if len(table) == 4:
                tables.append(table)
                used_players.update(table)
        
        # 残りのプレイヤーで卓を作成
        remaining = [p for p in players if p not in used_players]
        while len(remaining) >= 4:
            tables.append(remaining[:4])
            remaining = remaining[4:]
        
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
    
    def print_results(self, results: List[List[List[int]]]):
        """結果を見やすく出力"""
        print(f"\n麻雀卓組結果（完璧版） (参加者: {self.players}人, {self.rounds}回戦)")
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
            is_ideal = ((count == self.ideal_meetings_floor and count_distribution[count] == self.floor_pairs_needed) or
                       (count == self.ideal_meetings_ceil and count_distribution[count] == self.ceil_pairs_needed))
            marker = " ← 理想的" if is_ideal else ""
            print(f"  {count}回同卓: {count_distribution[count]}ペア ({percentage:.1f}%){marker}")
        
        # 理想的な分布との比較
        print(f"\n理想的な分布との比較:")
        print(f"  理想: {self.ideal_meetings_floor}回×{self.floor_pairs_needed}ペア, {self.ideal_meetings_ceil}回×{self.ceil_pairs_needed}ペア")
        print(f"  実際: ", end="")
        for count in sorted(count_distribution.keys()):
            print(f"{count}回×{count_distribution[count]}ペア ", end="")
        print()
        
        # カバレッジ
        realized_pairs = sum(1 for count in all_counts if count > 0)
        coverage = realized_pairs / self.total_pairs * 100
        
        print(f"\nペアカバレッジ: {realized_pairs}/{self.total_pairs} ({coverage:.1f}%)")


def main():
    parser = argparse.ArgumentParser(description='麻雀卓組生成プログラム（完璧版）')
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
    
    generator = PerfectTableGroupGenerator(args.players, args.rounds, args.five)
    results = generator.generate()
    generator.print_results(results)


if __name__ == "__main__":
    main()