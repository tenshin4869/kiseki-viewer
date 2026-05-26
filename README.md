# kiseki-data2-simple-pdr

`data/Data2` と `data/Data3` を対象にした、最小構成のPDR軌跡可視化プロジェクトです。

このプロジェクトでは、SmartPDR記事の処理構造を元にセンサーCSVから2D軌跡を生成します。重力方向に射影した加速度のHPF/LPF、傾き補償した磁気方位によるジャイロ方位の validation、動的歩幅推定（切替可能）、ステップごとの位置更新を実装しています。

```text
Accelerometer.csvを読み込む
Gravity.csvを読み込む
Gyroscope.csvを読み込む
Magnetometer.csvを読み込む
加速度を重力方向へ射影する
HPFで重力成分を取り除く
移動平均で平滑化する
歩行ピークを検出する
ジャイロの重力軸成分を積分し、磁気方位で妥当性を検証する
固定歩幅、またはpeak-to-valleyによる動的歩幅を推定する
ステップごとの歩幅とheadingで2D座標を更新する
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

各フォルダにはphyphoxの `Accelerometer.csv`、`Gravity.csv`、`Gyroscope.csv`、`Magnetometer.csv` が必要です。

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
output/Data2/TurnLeft_001/figures/trajectory_comparison.png
```

`trajectory_comparison.png` は、補正版の軌跡 (`corrected (SmartPDR)`) と、補正前の `acc_norm` + `gyro_z` 積分 + 固定歩幅によるシンプル版 (`simple (legacy)`) を同一座標上に表示します。シンプル版の再現設定は `comparison.simple` で変更できます。

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

## SmartPDR風の設定

記事では、GCSのz軸加速度から重力成分をHPFで除き、移動平均でLPFした信号をステップ検出に使います。このプロジェクトでは `Gravity.csv` の重力方向へ加速度を射影することで、端末姿勢の影響を抑えた縦方向信号を作ります。記事に記載された HPF 係数 `0.95` を既定値にしています。

```yaml
preprocessing:
  step_signal: "vertical_hpf"
  hpf_alpha: 0.95
```

方位は、記事と同様にジャイロの鉛直成分と傾き補償磁気方位を利用します。実装では角度を通常の実数として平均せず、`0/360` 度境界に安全な円周角差で磁気 validation を適用します。`heading.csv` には `heading_gyro_rad`、`heading_mag_rad`、`mag_validation_accepted` も出力されます。

```yaml
heading:
  use_smart_pdr: true
  bias_static_duration_s: 1.0
  mag_correction_gain: 0.05
```

初期バイアス区間を `1.0s` としているのは、現在の記録で1秒前後から歩行が始まるためです。ジャイロのみで比較する場合は `use_smart_pdr: false` とし、必要に応じて `gyro_scale` を校正できます。

この方位推定は端末の前方向と歩行者の進行方向が概ね一致する手持ち条件を前提にします。ポケット条件は端末姿勢と身体方向の対応が異なるため、姿勢モード別の方位補正を追加するまで同じ精度は期待できません。

原因切り分け中の既定設定では、歩幅を固定して方位の影響を見やすくしています。動的歩幅を試す場合は、記事の peak-to-valley と4乗根/対数モデルに切り替えられます。

```yaml
pdr:
  step_length_mode: "dynamic"
  dynamic_step_length:
    calibrate_to_fixed_step_length: true
```

## 基準歩幅設定

動的歩幅推定のキャリブレーション基準、または `step_length_mode: "fixed"` にした場合の固定歩幅は `config.yaml` で変更できます。現在の既定値は、10歩の直進が約6mになる `0.60m` です。

```yaml
pdr:
  step_length_m: 0.60
  step_length_by_dataset:
    Data3: 0.60
```

HPF後の縦方向加速度を使うため、歩行ピークの初期しきい値は `1.2` にしています。記事の3条件 detector も実装されていますが、`Data2/001_walk` の歩数を取りこぼしたため既定では `scipy.signal.find_peaks` を用います。

```yaml
step_detection:
  use_custom_algorithm: false
  height: 1.2
  height_by_dataset:
    Data3:
      hand: 1.0
```

## 実装しないこと

```text
曲がり区間検出
直進区間補正
自己アンカー補正
map matching
Kalman filter
Webアプリ化
```
