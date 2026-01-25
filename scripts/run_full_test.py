"""
フルテスト実行スクリプト（本番版）

統合された本番設定で学習・バックテストを実行
"""
import subprocess
import sys
from pathlib import Path
from datetime import datetime


def run_script(script_name):
    """スクリプトを実行"""
    script_path = Path(__file__).parent / script_name
    print(f"\n{'='*60}")
    print(f"実行中: {script_name}")
    print(f"開始時刻: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print('='*60)

    result = subprocess.run(
        [sys.executable, str(script_path)],
        capture_output=False,
        text=True
    )

    print(f"終了時刻: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    return result.returncode


def main():
    print("="*60)
    print("フルテスト開始（本番版）")
    print(f"開始時刻: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*60)
    print("\n【設定】")
    print("  利確/損切: ±10%")
    print("  保有期間: 20営業日")
    print("  スコア閾値: 0.55")
    print("  同時保有: 1銘柄")
    print("  フィルタ: ROE > 0%, 時価総額 > 100億円")

    # Phase 2: 学習（データは既に存在する前提）
    ret = run_script("phase2_train.py")
    if ret != 0:
        print("Phase 2 失敗")
        return

    # Phase 3: バックテスト
    ret = run_script("phase3_backtest.py")
    if ret != 0:
        print("Phase 3 失敗")
        return

    print("\n" + "="*60)
    print("フルテスト完了")
    print(f"終了時刻: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*60)


if __name__ == "__main__":
    main()
