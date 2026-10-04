#!/usr/bin/env bash
# Gradient-based planning (GBP) evaluation for all CartPole JEPA models.
# Usage: CUDA_VISIBLE_DEVICES=0 bash run_gbp_cartpole_eval.sh
set +e

GBP_ARGS="--trials 10 --primitive-budget 300 --planning-horizon 10 \
  --executed-steps 1 --gd-steps 50 --lr 0.1 --action-noise 0.05 \
  --objective last --success-threshold 0.7 --seed 123 --device cuda"

OUTDIR=/mnt/t7shield/results_landscape_paper_cartpole
mkdir -p "$OUTDIR"

run_timed() {
    local label="$1"; shift
    echo ""
    echo "════════════════════════════════════════════════════"
    echo "  START: $label  $(date '+%Y-%m-%d %H:%M:%S')"
    echo "════════════════════════════════════════════════════"
    local t0=$SECONDS
    "$@"
    local elapsed=$(( SECONDS - t0 ))
    echo "  DONE:  $label  $(date '+%Y-%m-%d %H:%M:%S')  (${elapsed}s)"
    echo "════════════════════════════════════════════════════"
}

# ══════════════════════════════════════════════════════════════════════════════
#  ACTION CONTEXT 1
# ══════════════════════════════════════════════════════════════════════════════

# ── 1. FWD + SR ───────────────────────────────────────────────────────────────
run_timed "act1 | FWD + SR" \
python experiments/compare_gd_smwm.py $GBP_ARGS \
  --ckpts /mnt/t7shield/jepa_results/cartpole_smwm_sigreg_act1_seed42/model_final.pt \
  --cfgs  configs/cartpole_smwm_sigreg_act1.yaml \
  --output "$OUTDIR"/GBP_act1_fwd_sr.json

# ── 2. FWD + AR + MS-AR ───────────────────────────────────────────────────────
run_timed "act1 | FWD + AR + MS-AR" \
python experiments/compare_gd_smwm.py $GBP_ARGS \
  --ckpts /mnt/t7shield/jepa_results/cartpole_smwm_fwd_ar_ms_1step_act1_seed42/model_final.pt \
  --cfgs  configs/cartpole_smwm_fwd_ar_ms_1step_act1.yaml \
  --output "$OUTDIR"/GBP_act1_fwd_ar_ms_ar.json

# ── 3. FWD + EP-AR ────────────────────────────────────────────────────────────
run_timed "act1 | FWD + EP-AR" \
python experiments/compare_gd_smwm.py $GBP_ARGS \
  --ckpts /mnt/t7shield/jepa_results/cartpole_smwm_fwd_endpoint_inv_act1_seed42/model_final.pt \
  --cfgs  configs/cartpole_smwm_fwd_endpoint_inverse_act1.yaml \
  --output "$OUTDIR"/GBP_act1_fwd_ep_ar.json

# ── 4. MS + SR ────────────────────────────────────────────────────────────────
run_timed "act1 | MS + SR" \
python experiments/compare_gd_smwm.py $GBP_ARGS \
  --ckpts /mnt/t7shield/jepa_results/cartpole_smwm_sigreg_rollout_act1_seed42/model_final.pt \
  --cfgs  configs/cartpole_smwm_sigreg_rollout_act1.yaml \
  --output "$OUTDIR"/GBP_act1_ms_sr.json

# ── 5. MS + EP-AR + SR ────────────────────────────────────────────────────────
run_timed "act1 | MS + EP-AR + SR" \
python experiments/compare_gd_smwm.py $GBP_ARGS \
  --ckpts /mnt/t7shield/jepa_results/cartpole_smwm_rollout_w05_endpoint_inv_sigreg_w01_act1_seed42/model_final.pt \
  --cfgs  configs/cartpole_smwm_rollout_w05_endpoint_inverse_sigreg_w01_act1.yaml \
  --output "$OUTDIR"/GBP_act1_ms_ep_ar_sr.json

# ── 6. MS + DINOv2 ────────────────────────────────────────────────────────────
run_timed "act1 | MS + DINOv2" \
python experiments/compare_gd_smwm.py $GBP_ARGS \
  --ckpts /mnt/t7shield/jepa_results/cartpole_smwm_dinov2_rollout_act1_seed42/model_final.pt \
  --cfgs  configs/cartpole_smwm_dinov2_rollout_act1.yaml \
  --output "$OUTDIR"/GBP_act1_ms_dinov2.json

# ── 7. FWD + DINOv2 ───────────────────────────────────────────────────────────
run_timed "act1 | FWD + DINOv2" \
python experiments/compare_gd_smwm.py $GBP_ARGS \
  --ckpts /mnt/t7shield/jepa_results/cartpole_smwm_dinov2_fwd_act1_seed42/model_final.pt \
  --cfgs  configs/cartpole_smwm_dinov2_fwd_act1.yaml \
  --output "$OUTDIR"/GBP_act1_fwd_dinov2.json

# ── 8. MS + iBOT ──────────────────────────────────────────────────────────────
run_timed "act1 | MS + iBOT" \
python experiments/compare_gd_smwm.py $GBP_ARGS \
  --ckpts /mnt/t7shield/jepa_results/cartpole_smwm_ibot_rollout_act1_seed42/model_final.pt \
  --cfgs  configs/cartpole_smwm_ibot_rollout_act1.yaml \
  --output "$OUTDIR"/GBP_act1_ms_ibot.json

# ── 9. FWD + iBOT ─────────────────────────────────────────────────────────────
run_timed "act1 | FWD + iBOT" \
python experiments/compare_gd_smwm.py $GBP_ARGS \
  --ckpts /mnt/t7shield/jepa_results/cartpole_smwm_ibot_fwd_act1_seed42/model_final.pt \
  --cfgs  configs/cartpole_smwm_ibot_fwd_act1.yaml \
  --output "$OUTDIR"/GBP_act1_fwd_ibot.json

# ── 10. FWD + iBOT + PR-EP-AR ─────────────────────────────────────────────────
run_timed "act1 | FWD + iBOT + PR-EP-AR" \
python experiments/compare_gd_smwm.py $GBP_ARGS \
  --ckpts /mnt/t7shield/jepa_results/cartpole_smwm_ibot_fwd_endpoint_inv_act1_seed42/model_final.pt \
  --cfgs  configs/cartpole_smwm_ibot_fwd_endpoint_inverse_act1.yaml \
  --output "$OUTDIR"/GBP_act1_fwd_ibot_proj_ep_ar.json

# ── 11. MS + iBOT + PR-AR ─────────────────────────────────────────────────────
run_timed "act1 | MS + iBOT + PR-AR" \
python experiments/compare_gd_smwm.py $GBP_ARGS \
  --ckpts /mnt/t7shield/jepa_results/cartpole_ibot_projector_ar_1step_act1_200ep_seed42/model_final.pt \
  --cfgs  configs/cartpole_smwm_ibot_projector_ar_1step_act1.yaml \
  --output "$OUTDIR"/GBP_act1_ms_ibot_proj_ar.json

# ── 12. MS + iBOT + PR-EP-AR ──────────────────────────────────────────────────
run_timed "act1 | MS + iBOT + PR-EP-AR" \
python experiments/compare_gd_smwm.py $GBP_ARGS \
  --ckpts /mnt/t7shield/jepa_results/cartpole_ibot_projector_rollout_endpoint_inv_act1_200ep_seed42/model_final.pt \
  --cfgs  configs/cartpole_smwm_ibot_projector_rollout_endpoint_inverse_act1.yaml \
  --output "$OUTDIR"/GBP_act1_ms_ibot_proj_ep_ar.json

# ── 13. MS + DINOv2 + PR-AR ───────────────────────────────────────────────────
run_timed "act1 | MS + DINOv2 + PR-AR" \
python experiments/compare_gd_smwm.py $GBP_ARGS \
  --ckpts /mnt/t7shield/jepa_results/cartpole_dinov2_projector_ar_1step_act1_200ep_seed42/model_final.pt \
  --cfgs  configs/cartpole_smwm_dinov2_projector_ar_1step_act1.yaml \
  --output "$OUTDIR"/GBP_act1_ms_dinov2_proj_ar.json

# ── 14. MS + DINOv2 + PR-SR ───────────────────────────────────────────────────
run_timed "act1 | MS + DINOv2 + PR-SR" \
python experiments/compare_gd_smwm.py $GBP_ARGS \
  --ckpts /mnt/t7shield/jepa_results/cartpole_dinov2_projector_sigreg_act1_200ep_seed42/model_final.pt \
  --cfgs  configs/cartpole_smwm_dinov2_projector_sigreg_act1.yaml \
  --output "$OUTDIR"/GBP_act1_ms_dinov2_proj_sr.json

# ── 15. MS + iBOT + PR-SR ─────────────────────────────────────────────────────
run_timed "act1 | MS + iBOT + PR-SR" \
python experiments/compare_gd_smwm.py $GBP_ARGS \
  --ckpts /mnt/t7shield/jepa_results/cartpole_ibot_projector_sigreg_act1_200ep_seed42/model_final.pt \
  --cfgs  configs/cartpole_smwm_ibot_projector_sigreg_act1.yaml \
  --output "$OUTDIR"/GBP_act1_ms_ibot_proj_sr.json

# ── 16. MS + DINOv2 + PR-EP-AR ────────────────────────────────────────────────
run_timed "act1 | MS + DINOv2 + PR-EP-AR" \
python experiments/compare_gd_smwm.py $GBP_ARGS \
  --ckpts /mnt/t7shield/jepa_results/cartpole_dinov2_projector_rollout_endpoint_inv_act1_200ep_seed42/model_final.pt \
  --cfgs  configs/cartpole_smwm_dinov2_projector_rollout_endpoint_inverse_act1.yaml \
  --output "$OUTDIR"/GBP_act1_ms_dinov2_proj_ep_ar.json

# ── 17. MS + SR (ViT,-I) ──────────────────────────────────────────────────────
run_timed "act1 | MS + SR (ViT,-I)" \
python experiments/compare_gd_smwm.py $GBP_ARGS \
  --ckpts /mnt/t7shield/jepa_results/cartpole_smwm_sigreg_rollout_statevit_act1_seed42/model_final.pt \
  --cfgs  configs/cartpole_smwm_sigreg_rollout_statevit_act1.yaml \
  --output "$OUTDIR"/GBP_act1_ms_sr_statevit.json

# ── 18. FWD + EP-AR (ViT,-I) ──────────────────────────────────────────────────
run_timed "act1 | FWD + EP-AR (ViT,-I)" \
python experiments/compare_gd_smwm.py $GBP_ARGS \
  --ckpts /mnt/t7shield/jepa_results/cartpole_smwm_fwd_endpoint_inverse_statevit_act1_seed42/model_final.pt \
  --cfgs  configs/cartpole_smwm_fwd_endpoint_inverse_statevit_act1.yaml \
  --output "$OUTDIR"/GBP_act1_fwd_ep_ar_statevit.json

# ── 19. MS + SR (MLP,-I) ──────────────────────────────────────────────────────
run_timed "act1 | MS + SR (MLP,-I)" \
python experiments/compare_gd_smwm.py $GBP_ARGS \
  --ckpts results/cartpole_smwm_sigreg_rollout_proprio_act1_seed42/model_final.pt \
  --cfgs  configs/cartpole_smwm_sigreg_rollout_proprio_act1.yaml \
  --output "$OUTDIR"/GBP_act1_ms_sr_mlp.json

# ── 20. FWD + EP-AR (MLP,-I) ──────────────────────────────────────────────────
run_timed "act1 | FWD + EP-AR (MLP,-I)" \
python experiments/compare_gd_smwm.py $GBP_ARGS \
  --ckpts results/cartpole_smwm_fwd_endpoint_inverse_proprio_act1_seed42/model_final.pt \
  --cfgs  configs/cartpole_smwm_fwd_endpoint_inverse_proprio_act1.yaml \
  --output "$OUTDIR"/GBP_act1_fwd_ep_ar_mlp.json

# ══════════════════════════════════════════════════════════════════════════════
#  ACTION CONTEXT 5
# ══════════════════════════════════════════════════════════════════════════════

# ── 21. MS + SR (act5) ────────────────────────────────────────────────────────
run_timed "act5 | MS + SR" \
python experiments/compare_gd_smwm.py $GBP_ARGS \
  --ckpts /mnt/t7shield/jepa_results/cartpole_sensorimotor_diff_proprio_sigreg_rollout_200ep_seed42/model_final.pt \
  --cfgs  configs/cartpole_sensorimotor_world_model_diff_proprio_sigreg_rollout.yaml \
  --output "$OUTDIR"/GBP_act5_ms_sr.json

# ── 22. MS + DINOv2 (act5) ────────────────────────────────────────────────────
run_timed "act5 | MS + DINOv2" \
python experiments/compare_gd_smwm.py $GBP_ARGS \
  --ckpts /mnt/t7shield/jepa_results/cartpole_sensorimotor_dinov2_rollout_200ep_seed42/model_final.pt \
  --cfgs  configs/cartpole_sensorimotor_world_model_dinov2_rollout.yaml \
  --output "$OUTDIR"/GBP_act5_ms_dinov2.json

# ── 23. MS + iBOT (act5) ──────────────────────────────────────────────────────
run_timed "act5 | MS + iBOT" \
python experiments/compare_gd_smwm.py $GBP_ARGS \
  --ckpts /mnt/t7shield/jepa_results/cartpole_sensorimotor_ibot_rollout_200ep_seed42/model_final.pt \
  --cfgs  configs/cartpole_sensorimotor_world_model_ibot_rollout.yaml \
  --output "$OUTDIR"/GBP_act5_ms_ibot.json

# ── 24. MS + iBOT + PR-MS-AR (act5) ──────────────────────────────────────────
run_timed "act5 | MS + iBOT + PR-MS-AR" \
python experiments/compare_gd_smwm.py $GBP_ARGS \
  --ckpts /mnt/t7shield/jepa_results/cartpole_ibot_projector_ar_1step_200ep_seed42/model_final.pt \
  --cfgs  configs/cartpole_smwm_ibot_projector_ar_1step.yaml \
  --output "$OUTDIR"/GBP_act5_ms_ibot_proj_ms_ar.json

# ── 25. MS + iBOT + PR-SR (act5) ──────────────────────────────────────────────
run_timed "act5 | MS + iBOT + PR-SR" \
python experiments/compare_gd_smwm.py $GBP_ARGS \
  --ckpts /mnt/t7shield/jepa_results/cartpole_ibot_projector_sigreg_200ep_seed42/model_final.pt \
  --cfgs  configs/cartpole_smwm_ibot_projector_sigreg.yaml \
  --output "$OUTDIR"/GBP_act5_ms_ibot_proj_sr.json

# ── 26. MS + iBOT + PR-AR (-P) (act5) ────────────────────────────────────────
run_timed "act5 | MS + iBOT + PR-AR (-P)" \
python experiments/compare_gd_smwm.py $GBP_ARGS \
  --ckpts /mnt/t7shield/jepa_results/cartpole_ibot_projector_ar_1step_noproprio_200ep_seed42/model_final.pt \
  --cfgs  configs/cartpole_smwm_ibot_projector_ar_1step_noproprio.yaml \
  --output "$OUTDIR"/GBP_act5_ms_ibot_proj_ar_noproprio.json

# ── 27. MS + iBOT + PR-SR (-P) (act5) ────────────────────────────────────────
run_timed "act5 | MS + iBOT + PR-SR (-P)" \
python experiments/compare_gd_smwm.py $GBP_ARGS \
  --ckpts /mnt/t7shield/jepa_results/cartpole_ibot_projector_sigreg_noproprio_200ep_seed42/model_final.pt \
  --cfgs  configs/cartpole_smwm_ibot_projector_sigreg_noproprio.yaml \
  --output "$OUTDIR"/GBP_act5_ms_ibot_proj_sr_noproprio.json

# ── 28. MS + iBOT (-P) (act5) ─────────────────────────────────────────────────
run_timed "act5 | MS + iBOT (-P)" \
python experiments/compare_gd_smwm.py $GBP_ARGS \
  --ckpts /mnt/t7shield/jepa_results/cartpole_sensorimotor_ibot_rollout_noproprio_200ep_seed42/model_final.pt \
  --cfgs  configs/cartpole_sensorimotor_world_model_ibot_rollout_noproprio.yaml \
  --output "$OUTDIR"/GBP_act5_ms_ibot_noproprio.json

# ══════════════════════════════════════════════════════════════════════════════
#  Summary table
# ══════════════════════════════════════════════════════════════════════════════
python - <<'EOF'
import json
from pathlib import Path

MODELS = [
    # act1
    ("act1 | FWD + SR",               "/mnt/t7shield/results_landscape_paper_cartpole/GBP_act1_fwd_sr.json"),
    ("act1 | FWD + AR + MS-AR",       "/mnt/t7shield/results_landscape_paper_cartpole/GBP_act1_fwd_ar_ms_ar.json"),
    ("act1 | FWD + EP-AR",            "/mnt/t7shield/results_landscape_paper_cartpole/GBP_act1_fwd_ep_ar.json"),
    ("act1 | MS + SR",                "/mnt/t7shield/results_landscape_paper_cartpole/GBP_act1_ms_sr.json"),
    ("act1 | MS + EP-AR + SR",        "/mnt/t7shield/results_landscape_paper_cartpole/GBP_act1_ms_ep_ar_sr.json"),
    ("act1 | MS + DINOv2",            "/mnt/t7shield/results_landscape_paper_cartpole/GBP_act1_ms_dinov2.json"),
    ("act1 | FWD + DINOv2",           "/mnt/t7shield/results_landscape_paper_cartpole/GBP_act1_fwd_dinov2.json"),
    ("act1 | MS + iBOT",              "/mnt/t7shield/results_landscape_paper_cartpole/GBP_act1_ms_ibot.json"),
    ("act1 | FWD + iBOT",             "/mnt/t7shield/results_landscape_paper_cartpole/GBP_act1_fwd_ibot.json"),
    ("act1 | FWD + iBOT + PR-EP-AR",  "/mnt/t7shield/results_landscape_paper_cartpole/GBP_act1_fwd_ibot_proj_ep_ar.json"),
    ("act1 | MS + iBOT + PR-AR",      "/mnt/t7shield/results_landscape_paper_cartpole/GBP_act1_ms_ibot_proj_ar.json"),
    ("act1 | MS + iBOT + PR-EP-AR",   "/mnt/t7shield/results_landscape_paper_cartpole/GBP_act1_ms_ibot_proj_ep_ar.json"),
    ("act1 | MS + DINOv2 + PR-AR",    "/mnt/t7shield/results_landscape_paper_cartpole/GBP_act1_ms_dinov2_proj_ar.json"),
    ("act1 | MS + DINOv2 + PR-SR",    "/mnt/t7shield/results_landscape_paper_cartpole/GBP_act1_ms_dinov2_proj_sr.json"),
    ("act1 | MS + iBOT + PR-SR",      "/mnt/t7shield/results_landscape_paper_cartpole/GBP_act1_ms_ibot_proj_sr.json"),
    ("act1 | MS + DINOv2 + PR-EP-AR", "/mnt/t7shield/results_landscape_paper_cartpole/GBP_act1_ms_dinov2_proj_ep_ar.json"),
    ("act1 | MS + SR (ViT,-I)",       "/mnt/t7shield/results_landscape_paper_cartpole/GBP_act1_ms_sr_statevit.json"),
    ("act1 | FWD + EP-AR (ViT,-I)",   "/mnt/t7shield/results_landscape_paper_cartpole/GBP_act1_fwd_ep_ar_statevit.json"),
    ("act1 | MS + SR (MLP,-I)",       "/mnt/t7shield/results_landscape_paper_cartpole/GBP_act1_ms_sr_mlp.json"),
    ("act1 | FWD + EP-AR (MLP,-I)",   "/mnt/t7shield/results_landscape_paper_cartpole/GBP_act1_fwd_ep_ar_mlp.json"),
    # act5
    ("act5 | MS + SR",                "/mnt/t7shield/results_landscape_paper_cartpole/GBP_act5_ms_sr.json"),
    ("act5 | MS + DINOv2",            "/mnt/t7shield/results_landscape_paper_cartpole/GBP_act5_ms_dinov2.json"),
    ("act5 | MS + iBOT",              "/mnt/t7shield/results_landscape_paper_cartpole/GBP_act5_ms_ibot.json"),
    ("act5 | MS + iBOT + PR-MS-AR",   "/mnt/t7shield/results_landscape_paper_cartpole/GBP_act5_ms_ibot_proj_ms_ar.json"),
    ("act5 | MS + iBOT + PR-SR",      "/mnt/t7shield/results_landscape_paper_cartpole/GBP_act5_ms_ibot_proj_sr.json"),
    ("act5 | MS + iBOT + PR-AR (-P)", "/mnt/t7shield/results_landscape_paper_cartpole/GBP_act5_ms_ibot_proj_ar_noproprio.json"),
    ("act5 | MS + iBOT + PR-SR (-P)", "/mnt/t7shield/results_landscape_paper_cartpole/GBP_act5_ms_ibot_proj_sr_noproprio.json"),
    ("act5 | MS + iBOT (-P)",         "/mnt/t7shield/results_landscape_paper_cartpole/GBP_act5_ms_ibot_noproprio.json"),
]

print(f"\n{'Model':<32} {'SR':>7} {'Held':>7} {'FinalErr':>10}")
print("=" * 59)
for name, path in MODELS:
    p = Path(path)
    if not p.exists():
        print(f"{name:<32} {'MISSING':>7}")
        continue
    data = json.loads(p.read_text())
    s = data['models'][0]['summary']
    print(f"{name:<32} {s['success_rate']:>7.1%} "
          f"{s['held_stable_rate']:>7.1%} {s['mean_final_error']:>10.5f}")
print("=" * 59)
EOF
