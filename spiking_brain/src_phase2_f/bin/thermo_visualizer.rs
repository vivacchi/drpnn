//! Phase 2 Fork F-G1-R1 リアルタイム可視化 (macroquad + GPU)
//!
//! ビルド・実行:
//!   cargo run --release --features visualizer --bin thermo_visualizer
//!
//! 描画要素:
//!   1. スパイク点灯  : 発火直後のニューロンが黄色く光る (spike_trace ベース)
//!   2. conductance   : シナプスを RdYlBu_r カラーで描画 (赤=強 / 青=弱)
//!   3. 生死と成長    : 死んだシナプス = 非表示、新生 = オレンジ枠
//!   4. entropy ヒート: ニューロン本体を entropy 強度で着色 (熱いほど赤)
//!
//! キー操作:
//!   SPACE      : pause / resume
//!   1 - 5      : 評価パターン A-E を 1 回提示
//!   W          : ホワイトノイズモード (ON/OFF トグル)
//!   N          : 訓練 1 trial 進める (pause 中)
//!   F / S      : シミュ速度 + / -
//!   T          : シナプス表示モード切替 (all → strong only → none)
//!   R          : リセット (新しいネットワーク)
//!   ESC        : 終了

use macroquad::prelude::*;
use ::rand::rngs::StdRng;
use ::rand::{Rng, SeedableRng};
use spiking_brain::phase2_f::thermo_network::{ThermoNetwork, ThermoNetworkConfig};

// ── シミュ定数 (thermo_m1_evaluation.rs と整合) ──
const TRIAL_DURATION_MS: f64 = 300.0;
const DT_MS: f64 = 0.5;
const PULSE_WIDTH_MS: f64 = 4.0;
const INPUT_CURRENT: i32 = 60;
const P_WHITE_NOISE: f64 = 1.0 / 600.0;
const TRIAL_STEPS: i32 = (TRIAL_DURATION_MS / DT_MS) as i32;

// ── 評価パターン (A-E、0-95ms 5ms step) ──
fn make_pattern_set(n_input: usize) -> Vec<(char, Vec<f64>)> {
    let step = 5.0;
    let a: Vec<f64> = (0..n_input).map(|i| i as f64 * step).collect();
    let b: Vec<f64> = (0..n_input).map(|i| (n_input - 1 - i) as f64 * step).collect();
    let mut c = vec![0.0; n_input];
    for i in 0..n_input {
        c[i] = if i % 2 == 0 {
            (i as f64 / 2.0) * step
        } else {
            ((n_input / 2 + i / 2) as f64) * step
        };
    }
    let d: Vec<f64> = (0..n_input)
        .map(|i| (n_input / 2 + i) as f64 * step % (n_input as f64 * step))
        .collect();
    let e: Vec<f64> = vec![60.0, 10.0, 50.0, 80.0, 15.0, 90.0, 55.0, 0.0,
                            25.0, 70.0, 35.0, 5.0, 45.0, 20.0, 75.0, 30.0,
                            65.0, 40.0, 85.0, 95.0]
        .into_iter().take(n_input).collect();
    vec![('A', a), ('B', b), ('C', c), ('D', d), ('E', e)]
}

// ── 入力スケジュール (今フレームで何を入れるか) ──
#[derive(Clone)]
enum InputMode {
    Idle,                                 // 入力なし
    Pattern { times: Vec<f64>, step: i32 },// 固定パターン提示中 (残り step)
    WhiteNoise,                           // 連続ホワイトノイズ
}

fn step_input_for_pattern(times: &[f64], step: i32, n_input: usize) -> Vec<i32> {
    let pulse_steps = (PULSE_WIDTH_MS / DT_MS).max(1.0) as i32;
    let mut ext = vec![0i32; n_input];
    for (k, &t) in times.iter().enumerate() {
        let fs = (t / DT_MS) as i32;
        if step >= fs && step < fs + pulse_steps {
            ext[k] = INPUT_CURRENT;
        }
    }
    ext
}

fn step_input_white_noise(n_input: usize, rng: &mut StdRng, pulse_remaining: &mut [i32]) -> Vec<i32> {
    let pulse_steps = (PULSE_WIDTH_MS / DT_MS).max(1.0) as i32;
    let mut ext = vec![0i32; n_input];
    for k in 0..n_input {
        if pulse_remaining[k] > 0 {
            pulse_remaining[k] -= 1;
            ext[k] = INPUT_CURRENT;
        } else if rng.gen::<f64>() < P_WHITE_NOISE {
            pulse_remaining[k] = pulse_steps - 1;
            ext[k] = INPUT_CURRENT;
        }
    }
    ext
}

// ── 色マッピング ──
fn conductance_color(c: i32) -> Color {
    // 0..100 を 青→緑→赤 (RdYlBu_r 風の単純化)
    let t = (c as f32 / 100.0).clamp(0.0, 1.0);
    if t < 0.5 {
        // 青 → 緑
        let s = t * 2.0;
        Color::new(0.1 + s * 0.3, 0.3 + s * 0.5, 0.9 - s * 0.5, 0.6)
    } else {
        // 緑 → 赤
        let s = (t - 0.5) * 2.0;
        Color::new(0.4 + s * 0.55, 0.8 - s * 0.5, 0.4 - s * 0.35, 0.6 + s * 0.3)
    }
}

fn entropy_color(e: i32) -> Color {
    // 0..200 を 灰 → 赤
    let t = (e as f32 / 200.0).clamp(0.0, 1.0);
    Color::new(0.35 + t * 0.65, 0.35 + (1.0 - t) * 0.2, 0.35 + (1.0 - t) * 0.2, 1.0)
}

#[derive(Clone, Copy, PartialEq)]
enum SynView { All, Strong, None }

fn window_conf() -> Conf {
    Conf {
        window_title: "Phase 2 Fork F-G1-R1 — Real-time Thermodynamic SNN".to_string(),
        window_width: 1320,
        window_height: 960,
        ..Default::default()
    }
}

#[macroquad::main(window_conf)]
async fn main() {
    let mut cfg = ThermoNetworkConfig::default();
    let grid_w = cfg.grid_width as f32;
    let grid_h = cfg.grid_height as f32;
    let mut net = ThermoNetwork::new(cfg.clone());
    let initial_syn_count = net.synapses.len();
    let n_input = net.input_neurons.len();
    let patterns = make_pattern_set(n_input);

    let mut rng = StdRng::seed_from_u64(42);
    let mut pulse_remaining = vec![0i32; n_input];
    let mut input_mode = InputMode::Idle;
    let mut paused = false;
    let mut steps_per_frame: i32 = 4;
    let mut syn_view = SynView::Strong;
    let mut trial_counter: u64 = 0;
    let mut total_pruned: u64 = 0;
    let mut total_grown: u64 = 0;

    // ── 描画レイアウト (grid_h に合わせて縦長対応) ──
    let panel_x = 30.0;
    let panel_y = 90.0;
    let panel_w = 760.0;
    let cell_size = panel_w / grid_w;
    let panel_h = cell_size * cfg.grid_height as f32;
    let neuron_radius = cell_size * 0.32;
    let _ = panel_h;

    let pos = |x: f32, y: f32| -> (f32, f32) {
        (panel_x + (x + 0.5) * cell_size, panel_y + (y + 0.5) * cell_size)
    };

    loop {
        clear_background(Color::new(0.08, 0.08, 0.10, 1.0));

        // ── キー入力 ──
        if is_key_pressed(KeyCode::Escape) { break; }
        if is_key_pressed(KeyCode::Space) { paused = !paused; }
        if is_key_pressed(KeyCode::W) {
            input_mode = match input_mode {
                InputMode::WhiteNoise => InputMode::Idle,
                _ => InputMode::WhiteNoise,
            };
        }
        if is_key_pressed(KeyCode::F) { steps_per_frame = (steps_per_frame + 2).min(120); }
        if is_key_pressed(KeyCode::S) { steps_per_frame = (steps_per_frame - 2).max(1); }
        if is_key_pressed(KeyCode::T) {
            syn_view = match syn_view {
                SynView::All => SynView::Strong,
                SynView::Strong => SynView::None,
                SynView::None => SynView::All,
            };
        }
        if is_key_pressed(KeyCode::R) {
            net = ThermoNetwork::new(cfg.clone());
            trial_counter = 0;
            total_pruned = 0;
            total_grown = 0;
            input_mode = InputMode::Idle;
        }
        let pattern_keys = [
            (KeyCode::Key1, 0), (KeyCode::Key2, 1), (KeyCode::Key3, 2),
            (KeyCode::Key4, 3), (KeyCode::Key5, 4),
        ];
        for (kc, pi) in pattern_keys.iter() {
            if is_key_pressed(*kc) {
                if let Some((_, times)) = patterns.get(*pi) {
                    input_mode = InputMode::Pattern { times: times.clone(), step: 0 };
                }
            }
        }

        // ── シミュレーション (paused でなければ steps_per_frame 回 step) ──
        if !paused {
            for _ in 0..steps_per_frame {
                let ext = match &mut input_mode {
                    InputMode::Idle => vec![0i32; n_input],
                    InputMode::Pattern { times, step } => {
                        let ext = step_input_for_pattern(times, *step, n_input);
                        *step += 1;
                        if *step >= TRIAL_STEPS {
                            trial_counter += 1;
                            input_mode = InputMode::Idle;
                        }
                        ext
                    }
                    InputMode::WhiteNoise => {
                        step_input_white_noise(n_input, &mut rng, &mut pulse_remaining)
                    }
                };
                let prev_grown = net.axons_grown;
                let prev_pruned = net.axons_pruned;
                let _fired = net.step(&ext);
                total_grown = net.axons_grown;
                total_pruned = net.axons_pruned;
                let _ = prev_grown; let _ = prev_pruned;
            }
        }

        // ── 描画: 背景パネル枠 ──
        let panel_w_draw = cell_size * grid_w;
        let panel_h_draw = cell_size * cfg.grid_height as f32;
        draw_rectangle_lines(panel_x - 4.0, panel_y - 4.0,
            panel_w_draw + 8.0, panel_h_draw + 8.0, 2.0, GRAY);

        // ── (1) シナプス描画 ──
        if syn_view != SynView::None {
            for (idx, s) in net.synapses.iter().enumerate() {
                if !s.alive { continue; }
                if syn_view == SynView::Strong && s.conductance < 50 { continue; }
                let pre = &net.neurons[s.pre];
                let post = &net.neurons[s.post];
                let (x1, y1) = pos(pre.position.0 as f32, pre.position.1 as f32);
                let (x2, y2) = pos(post.position.0 as f32, post.position.1 as f32);
                let is_new = idx >= initial_syn_count;
                let mut c = conductance_color(s.conductance);
                let w = if s.conductance >= 80 { 1.8 }
                         else if s.conductance >= 50 { 1.0 }
                         else { 0.4 };
                if is_new {
                    // 新生は橙縁取り (背景にもう一本太い線を引く)
                    draw_line(x1, y1, x2, y2, w + 1.2, Color::new(1.0, 0.55, 0.0, 0.45));
                }
                c.a = if syn_view == SynView::All { 0.25 } else { 0.9 };
                draw_line(x1, y1, x2, y2, w, c);
            }
        }

        // ── (2) ニューロン描画 (entropy ヒート + スパイク点灯) ──
        for (i, n) in net.neurons.iter().enumerate() {
            let (x, y) = pos(n.position.0 as f32, n.position.1 as f32);

            // 種類ごとに base 色を入れ替え
            let is_input = net.input_neurons.contains(&i);
            let is_output = net.output_neurons.contains(&i);
            let base = if is_input {
                Color::new(0.2, 0.5, 1.0, 1.0)
            } else if is_output {
                Color::new(0.2, 0.9, 0.3, 1.0)
            } else if n.is_inhibitory {
                Color::new(0.85, 0.2, 0.2, 1.0)
            } else {
                entropy_color(n.local_entropy)
            };
            draw_circle(x, y, neuron_radius, base);

            // 発火直後 (spike_trace が大きい) は黄色く光る
            if n.spike_trace > 0 {
                let glow = (n.spike_trace as f32 / 160.0).clamp(0.0, 1.0);
                draw_circle(x, y, neuron_radius * (1.0 + glow * 0.8),
                    Color::new(1.0, 0.95, 0.2, glow * 0.7));
                draw_circle(x, y, neuron_radius * 0.55,
                    Color::new(1.0, 1.0, 0.5, 1.0));
            }
        }

        // ── HUD (右側パネル) ──
        let hud_x = panel_x + panel_w_draw + 40.0;
        let mut hud_y = panel_y;
        let line = 22.0;

        let alive = net.synapses.iter().filter(|s| s.alive).count();
        let strong = net.synapses.iter().filter(|s| s.alive && s.conductance >= 80).count();
        let new_alive = net.synapses.iter().enumerate()
            .filter(|(i, s)| *i >= initial_syn_count && s.alive).count();
        let firing_now = net.neurons.iter().filter(|n| n.spike_trace > 150).count();

        draw_text("Phase 2 Fork F-G1-R1", hud_x, hud_y, 22.0, WHITE); hud_y += line + 4.0;
        draw_text(&format!("time step      : {}", net.current_time), hud_x, hud_y, 18.0, LIGHTGRAY); hud_y += line;
        draw_text(&format!("trials done    : {}", trial_counter), hud_x, hud_y, 18.0, LIGHTGRAY); hud_y += line;
        draw_text(&format!("steps/frame    : {}", steps_per_frame), hud_x, hud_y, 18.0, LIGHTGRAY); hud_y += line;
        draw_text(&format!("paused         : {}", if paused {"YES"} else {"NO"}), hud_x, hud_y, 18.0, if paused {YELLOW} else {LIGHTGRAY}); hud_y += line;
        hud_y += 6.0;
        draw_text(&format!("synapses alive : {} / {}", alive, net.synapses.len()), hud_x, hud_y, 18.0, LIGHTGRAY); hud_y += line;
        draw_text(&format!(" cond >= 80    : {}", strong), hud_x, hud_y, 18.0, ORANGE); hud_y += line;
        draw_text(&format!("axons grown    : {}", total_grown), hud_x, hud_y, 18.0, ORANGE); hud_y += line;
        draw_text(&format!(" of which alive: {}", new_alive), hud_x, hud_y, 18.0, ORANGE); hud_y += line;
        draw_text(&format!("axons pruned   : {}", total_pruned), hud_x, hud_y, 18.0, GRAY); hud_y += line;
        hud_y += 6.0;
        draw_text(&format!("firing now     : {}", firing_now), hud_x, hud_y, 18.0,
            if firing_now > 0 {YELLOW} else {LIGHTGRAY}); hud_y += line;

        let mode_label = match &input_mode {
            InputMode::Idle => "idle".to_string(),
            InputMode::Pattern { step, .. } => format!("pattern (step {}/{})", step, TRIAL_STEPS),
            InputMode::WhiteNoise => "white noise".to_string(),
        };
        draw_text(&format!("input mode     : {}", mode_label), hud_x, hud_y, 18.0, SKYBLUE); hud_y += line;

        let view_label = match syn_view {
            SynView::All => "all alive",
            SynView::Strong => "strong only (≥50)",
            SynView::None => "hidden",
        };
        draw_text(&format!("syn view (T)   : {}", view_label), hud_x, hud_y, 18.0, LIGHTGRAY); hud_y += line;

        hud_y += line * 0.6;
        draw_text("Keys:", hud_x, hud_y, 18.0, WHITE); hud_y += line;
        for s in &[
            "  SPACE  pause/resume",
            "  1..5   present pattern A-E",
            "  W      white-noise toggle",
            "  F/S    speed +/-",
            "  T      synapse view cycle",
            "  R      reset network",
            "  ESC    quit",
        ] {
            draw_text(s, hud_x, hud_y, 16.0, GRAY); hud_y += line - 2.0;
        }

        // タイトル
        draw_text(
            "Real-time Thermodynamic SNN (CPU sim + GPU render)",
            panel_x, 40.0, 28.0, WHITE,
        );
        draw_text(
            "GREEN = output, BLUE = input, RED = inhibitory, YELLOW glow = firing, ORANGE = new synapse",
            panel_x, 68.0, 16.0, LIGHTGRAY,
        );

        next_frame().await
    }
}
