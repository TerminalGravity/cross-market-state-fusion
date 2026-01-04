# Phase 5 Temporal Architecture - Detailed Analysis

**Commit**: 1f49f6f
**Date**: 2025-12-31
**Author**: nikshepsvn
**Performance**: $3,289 PnL, 164% ROI on $50 initial capital (973 trades)

---

## Architecture Comparison

### Our Current Version (Pre-Phase 5)

**Actor**: Simple feedforward
```
Input: 18 features
→ Linear(18, 128) → tanh
→ Linear(128, 128) → tanh
→ Linear(128, 3) → softmax
```

**Critic**: Symmetric with actor
```
Input: 18 features
→ Linear(18, 128) → tanh
→ Linear(128, 128) → tanh
→ Linear(128, 1)
```

**Key Parameters**:
- Hidden size: 128 (both networks)
- gamma: 0.99
- buffer_size: 512
- entropy_coef: 0.10 (high to prevent collapse)
- No temporal processing
- No LayerNorm

---

### Phase 5 Upstream Version

**TemporalEncoder**: NEW! Captures momentum/trends
```
Input: Last 5 states (5 * 18 = 90 features)
→ Linear(90, 64) → LayerNorm → tanh
→ Linear(64, 32) → LayerNorm → tanh
Output: 32 temporal features
```

**Actor**: Temporal-aware, smaller network
```
Input: Current state (18) + Temporal features (32) = 50
→ Linear(50, 64) → LayerNorm → tanh
→ Linear(64, 64) → LayerNorm → tanh
→ Linear(64, 3) → softmax
```

**Critic**: Asymmetric (larger than actor)
```
Input: Current state (18) + Temporal features (32) = 50
→ Linear(50, 96) → LayerNorm → tanh
→ Linear(96, 96) → LayerNorm → tanh
→ Linear(96, 1)
```

**Key Parameters**:
- Actor hidden: 64 (smaller to prevent overfitting)
- Critic hidden: 96 (larger for better value estimation)
- **Asymmetric design**: Critic 50% larger than actor
- gamma: 0.95 (lower for 15-min horizon)
- buffer_size: 256 (smaller for faster adaptation)
- entropy_coef: 0.03 (lower to allow sparse policy - mostly HOLD)
- history_len: 5 (temporal window)
- temporal_dim: 32 (temporal encoder output)
- **LayerNorm** in all networks for training stability

---

## Key Innovations

### 1. Temporal Processing
**What**: Maintains history of last 5 states per asset
**Why**: Captures momentum, velocity, and trend direction
**How**:
- `_state_history[asset]` - deque per market
- `_get_temporal_state()` - stacks last N states
- TemporalEncoder compresses history to 32 features

**Benefit**: Model can see "where the price is going" not just "where it is"

### 2. Asymmetric Actor-Critic
**What**: Critic (96 hidden) is 50% larger than actor (64 hidden)
**Why**:
- Value estimation is harder than policy
- Critic doesn't overfit as easily (regresses to scalar)
- Better value estimates → better advantage computation → better policy

**Benefit**: More accurate value estimates improve PPO updates

### 3. Lower Gamma (0.95 vs 0.99)
**What**: Discount factor reduced to 0.95
**Why**: 15-minute markets have short horizon
**Benefit**: Agent focuses on immediate rewards, not distant future

### 4. Smaller Buffer (256 vs 512)
**What**: Experience replay buffer cut in half
**Why**: Faster regime adaptation in volatile crypto markets
**Benefit**: Model updates more frequently with fresh data

### 5. Lower Entropy (0.03 vs 0.10)
**What**: Entropy coefficient reduced dramatically
**Why**: Most states should be HOLD (sparse trading)
**Benefit**: Policy can collapse to mostly HOLD without penalty

### 6. LayerNorm Everywhere
**What**: Layer normalization in all networks
**Why**: Training stability with different feature scales
**Benefit**: Smoother gradient flow, faster convergence

---

## Performance Results

**Upstream Phase 5 (paper trading)**:
- 973 trades over training period
- $3,289 total PnL
- 164% ROI on $50 initial capital
- 22.8% win rate
- **By asset**:
  - XRP: +$3,126 (best performer)
  - SOL: +$757
  - BTC: +$509
  - ETH: -$1,102 (worst performer)

**Our Current Version**:
- Unknown (need to test)
- Currently running with Phase 4 or earlier architecture

---

## Code Changes Required

### 1. strategies/rl_mlx.py (~158 lines added)
- [ ] Add `TemporalEncoder` class
- [ ] Modify `Actor` to accept temporal features
- [ ] Modify `Critic` to accept temporal features (and make larger)
- [ ] Add `_state_history` dict for per-asset history
- [ ] Add `_get_temporal_state()` method
- [ ] Update `act()` to use temporal state
- [ ] Update `store()` to handle temporal state
- [ ] Update `Experience` dataclass with temporal fields
- [ ] Update hyperparameters

### 2. strategies/base.py (~109 lines changed)
- [ ] Add `get_confidence_size()` method to Action enum
  - Dynamic sizing: 0.25x at prob=0.5, up to 1.0x at extremes
  - Edge increases at extreme probabilities in binary markets
  - Formula: `base(0.25) + scale(0.75) × extremeness`

- [ ] Update `to_features()` with feature normalization
  - ALL 18 features now clamped to [-1, 1] range
  - Momentum: `returns × 50` then clamp
  - CVD: `cvd_acceleration × 10` then clamp
  - Spread: `spread_pct × 20` then clamp
  - Vol: `vol_5m × 20` then clamp
  - PnL: `position_pnl / 50` then clamp
  - Order flow features already in [-1, 1]

- [ ] Helper function: `clamp(x, min_val=-1.0, max_val=1.0)`
  - Prevents extreme values from dominating gradients
  - Ensures consistent input scaling for neural network

### 3. rl_model.safetensors
- [ ] Update with new trained weights
- Size: 154576 bytes → 158112 bytes (+3536 bytes for temporal encoder)

### 4. rl_model_stats.npz
- [ ] Update normalization stats
- Size: 808 bytes → 2642 bytes (+1834 bytes, likely temporal stats)

---

## Integration Challenges

### Challenge 1: State History Management
**Issue**: Need to maintain per-asset state history
**Our code**: `railway_worker.py` manages positions
**Solution**: Initialize `_state_history` when market discovered, clear on market close

### Challenge 2: Experience Dataclass Changes
**Issue**: Experience now includes temporal_state fields
**Our code**: Database stores trades, not raw experiences
**Solution**: No DB changes needed - temporal processing is internal to strategy

### Challenge 3: Model Loading
**Issue**: New model has different architecture (TemporalEncoder)
**Our code**: `load_model()` in rl_mlx.py
**Solution**: Model file includes all weights, loading should work automatically

### Challenge 4: Hyperparameter Conflicts
**Issue**: Different defaults (gamma, buffer_size, entropy_coef)
**Our code**: May have overridden these in railway_worker.py or run.py
**Solution**: Check all initialization points, update to Phase 5 values

---

## Testing Plan

### Phase 1: Standalone Testing
1. Replace strategies/rl_mlx.py with upstream version
2. Update rl_model.safetensors and rl_model_stats.npz
3. Run paper trading locally: `python run.py rl --load rl_model`
4. Verify temporal state history works
5. Check for errors in temporal encoder

### Phase 2: Integration with Live Infrastructure
1. Test with railway_worker.py in PAPER mode
2. Verify safety system still works
3. Check database integration
4. Monitor performance vs old model

### Phase 3: Live Deployment
1. Deploy to Fly.io in PAPER mode first
2. Monitor for 24-48 hours
3. Compare PnL to old model
4. If better, switch to LIVE mode

---

## Risk Assessment

**Low Risk**:
- ✅ Upstream architecture is well-tested (164% ROI)
- ✅ Changes are mostly internal to strategy
- ✅ No database schema changes needed
- ✅ Safety system independent of strategy

**Medium Risk**:
- ⚠️ New model may perform differently in live markets
- ⚠️ Hyperparameter changes may need tuning
- ⚠️ State history memory usage (5 states * 4 markets * 18 features)

**Mitigation**:
- Test in PAPER mode first
- Monitor memory usage
- Compare performance to baseline
- Keep old model as fallback

---

## Next Steps

1. ✅ Document Phase 5 architecture (this file)
2. ⏳ Review strategies/base.py changes
3. ⏳ Check requirements.txt for new dependencies
4. ⏳ Analyze crash protection commit (c537da8)
5. ⏳ Review all documentation changes
6. ⏳ Create integration branch
7. ⏳ Test Phase 5 locally
8. ⏳ Deploy and monitor

---

**Status**: Analysis complete, ready for base.py review
