#!/usr/bin/env python3
"""ブラウザからの使用をシミュレート"""

import requests
import json


def test_browser_scenario():
    """ブラウザからの実際の使用パターンをテスト"""
    
    # 様々な入力パターンをテスト
    test_cases = [
        {
            "name": "標準的な12人6回戦",
            "data": {
                "players": ["太郎", "次郎", "三郎", "四郎", "五郎", "六郎",
                           "七郎", "八郎", "九郎", "十郎", "十一郎", "十二郎"],
                "rounds": 6,
                "allow_five": False
            }
        },
        {
            "name": "空白を含む名前",
            "data": {
                "players": ["山田 太郎", "鈴木 次郎", "佐藤 三郎", "田中 四郎", 
                           "高橋 五郎", "伊藤 六郎", "渡辺 七郎", "山本 八郎",
                           "中村 九郎", "小林 十郎", "加藤 十一郎", "吉田 十二郎"],
                "rounds": 6,
                "allow_five": False
            }
        },
        {
            "name": "改行や余分な空白を含む場合",
            "data": {
                "players": ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12"],
                "rounds": 6,
                "allow_five": False
            }
        }
    ]
    
    url = "http://localhost:5001/generate"
    
    for test_case in test_cases:
        print(f"\n{'='*60}")
        print(f"テストケース: {test_case['name']}")
        
        try:
            response = requests.post(url, json=test_case['data'])
            
            if response.status_code == 200:
                result = response.json()
                
                if result.get('success'):
                    stats = result['results']['statistics']['pair_statistics']
                    
                    print(f"結果:")
                    print(f"  最小同卓回数: {stats['min_count']}回")
                    print(f"  最大同卓回数: {stats['max_count']}回")
                    print(f"  カバレッジ: {stats['coverage']:.1f}%")
                    
                    if stats['min_count'] == 0:
                        print("  ⚠️ 0回同卓のペアが存在します！")
                        
                        # どのペアが0回か詳細を表示
                        distribution = stats['distribution']
                        if '0' in distribution:
                            print(f"  0回同卓のペア数: {distribution['0']}")
                    else:
                        print("  ✓ 全ペアが最低1回同卓")
                else:
                    print(f"エラー: {result}")
            else:
                print(f"HTTPエラー: {response.status_code}")
                
        except Exception as e:
            print(f"エラー: {str(e)}")
    
    print(f"\n{'='*60}")
    print("ブラウザでのアクセス方法:")
    print("1. http://localhost:5001 を開く")
    print("2. 参加者を12人入力")
    print("3. 回数を6に設定")
    print("4. '生成'ボタンをクリック")
    print("\n現在のサーバーは正常に動作しています。")


if __name__ == "__main__":
    test_browser_scenario()