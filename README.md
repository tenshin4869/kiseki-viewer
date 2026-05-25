# kiseki-data2-simple-pdr

`data/Data2` と `data/Data3` を対象にした、最小構成のPDR軌跡可視化プロジェクトです。

このプロジェクトでは、SmartPDR論文の磁気センサ融合や動的歩幅推定は入れていません。処理は以下だけです。

```text
Accelerometer.csvを読み込む
Gyroscope.csvを読み込む
加速度ノルムを計算する
移動平均で平滑化する
歩行ピークを検出する
指定ジャイロ軸を積分してheadingを推定する
データセットごとの固定歩幅で2D座標を更新する
試行ごとの結果と重ね合わせ図を出力する
```

## 座標系

```text
スタート地点: x=0, y=0
初期進行方向: y軸正方向
右回転: x軸正方向へ曲がる
左回転: x軸負方向へ曲がる
heading=0 rad: +y方向
```

この座標系に合うように `gyro_sign: -1.0` を初期設定にしています。

## セットアップ

```bash
python3 -m pip install -r requirements.txt
```

## 入力データ

入力データはこのプロジェクト内の `data` フォルダ配下です。現在は `data/Data2` と `data/Data3` を読み込みます。

`P` で始まる試行名はポケット条件、`P` で始まらない試行名は手持ち条件として扱います。

```text
data/Data2/001_walk/
data/Data2/PWalk_001/
data/Data2/LTurn_001/
data/Data2/PLTurn_001/
data/Data2/RTurn_001/
data/Data2/PRTurn_001/
data/Data3/Walk_002/
data/Data3/PWalk_002/
```

各フォルダにはphyphoxの `Accelerometer.csv` と `Gyroscope.csv` が必要です。

## 1試行を実行

```bash
python3 scripts/run_trial.py --trial-id Data2/TurnLeft_001 --config config.yaml
```

出力例:

```text
output/Data2/TurnLeft_001/processed/steps.csv
output/Data2/TurnLeft_001/processed/heading.csv
output/Data2/TurnLeft_001/processed/trajectory.csv
output/Data2/TurnLeft_001/figures/acc_norm.png
output/Data2/TurnLeft_001/figures/heading.png
output/Data2/TurnLeft_001/figures/trajectory.png
```

## 全試行を実行

```bash
python3 scripts/run_all.py --config config.yaml
```

重ね合わせ図:

```text
output/overlays/all/trajectory_overlay.png
output/overlays/hand/trajectory_overlay.png
output/overlays/pocket/trajectory_overlay.png
output/overlays/Data2/hand/trajectory_overlay.png
output/overlays/Data2/pocket/trajectory_overlay.png
output/overlays/Data3/hand/trajectory_overlay.png
output/overlays/Data3/pocket/trajectory_overlay.png
```

## 歩幅設定

歩幅は `config.yaml` で変更できます。全体の初期値は `0.65m` ですが、Data3は直進6mの距離に合うように `0.67m` にしています。

```yaml
pdr:
  step_length_m: 0.65
  step_length_by_dataset:
    Data3: 0.67
```

Data3の手持ちデータだけ歩行ピークが小さく、初期値の `12.0` では歩数を取り逃がしたため、Data3の手持ちだけ `11.5` にしています。ポケット条件は過検出を避けるため、通常の `12.0` のままです。

```yaml
step_detection:
  height: 12.0
  height_by_dataset:
    Data3:
      hand: 11.5
```

## 実装しないこと

```text
SmartPDR論文の磁気センサ融合
Gravity.csvを使った座標変換
動的歩幅推定
曲がり区間検出
直進区間補正
自己アンカー補正
map matching
Kalman filter
Webアプリ化
```
