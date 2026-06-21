//! Billing — the pure accounting math for the router<->worker payment-channel loop.
//!
//! The economic model maps an inference session onto a CE **payment channel** (off-chain receipts,
//! one on-chain settlement). The router is the **payer** (it opens the channel and signs receipts);
//! the worker is the **host** (it accumulates the highest receipt and redeems it with one
//! `channel_close`). This module is I/O-free so the accounting is unit-testable without a node:
//! it computes *what a session owes* from a [`PriceSheet`] plus the work done, tracks the monotonic
//! `cumulative` total a receipt must cover, and meters long streaming generations on fixed
//! **heartbeat intervals** (30s) so a multi-minute generation is billed as it runs, not only at the
//! end.
//!
//! All money is integer base units ([`Amount`]); never floats. `cumulative` only ever increases —
//! a channel receipt authorizes a running total, and the host redeems the highest one.

use ce_rs::{Amount, CREDIT};

/// The default heartbeat billing interval, matching CE's long-running-work cadence (30s).
pub const HEARTBEAT_INTERVAL_SECS: u64 = 30;

/// How a session's compute is priced. All fields are base units; zero disables that component.
///
/// Cost accrues from three sources, summed:
/// - `per_token` — base units per generated token (the dominant cost for short clinical queries);
/// - `per_gb_second` — base units per GB-second of model residency held while a session streams
///   (meters long generations that pin a large model);
/// - `per_request` — a flat base-unit floor charged once per request (covers dispatch overhead).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PriceSheet {
    /// Base units charged per generated token.
    pub per_token: Amount,
    /// Base units charged per GB-second of work (model-GB × wall-seconds).
    pub per_gb_second: Amount,
    /// Flat base-unit floor charged once per request.
    pub per_request: Amount,
}

impl PriceSheet {
    /// A sensible default: ~0.001 credit/request, a small per-token rate, and a per-GB-second rate
    /// for long streams. Operators override these via the worker/router CLI.
    pub fn default_sheet() -> PriceSheet {
        PriceSheet {
            // 0.00001 credit per token.
            per_token: Amount::from_base(CREDIT / 100_000),
            // 0.000001 credit per GB-second.
            per_gb_second: Amount::from_base(CREDIT / 1_000_000),
            // 0.001 credit flat per request.
            per_request: Amount::from_base(CREDIT / 1_000),
        }
    }

    /// Cost in base units for a single completed unit of work: a flat per-request floor plus the
    /// per-token cost of `tokens` produced plus the per-GB-second cost of holding `model_gb` of
    /// model for `seconds`. Saturating throughout — pricing never panics or overflows.
    pub fn cost(&self, tokens: u64, model_gb: u64, seconds: u64) -> Amount {
        let token_cost = self.per_token.base().saturating_mul(tokens as i128);
        let gb_seconds = (model_gb as i128).saturating_mul(seconds as i128);
        let gb_second_cost = self.per_gb_second.base().saturating_mul(gb_seconds);
        let total = self
            .per_request
            .base()
            .saturating_add(token_cost)
            .saturating_add(gb_second_cost);
        Amount::from_base(total)
    }
}

/// A running billing meter for one (payer, host) session over a single channel. Tracks the monotonic
/// `cumulative` total every receipt must cover, plus the heartbeat clock for long streams.
///
/// On the router (payer) it decides the `cumulative` to sign each receipt for; on the worker (host)
/// it tracks the highest received `cumulative` so the final `channel_close` redeems the right total.
#[derive(Debug, Clone)]
pub struct Meter {
    price: PriceSheet,
    /// The model's on-disk size in GB, used for the per-GB-second residency charge.
    model_gb: u64,
    /// Monotonic running total this channel has been billed for (base units). Receipts sign this.
    cumulative: Amount,
    /// The session capacity the channel was opened with (cumulative is clamped to it).
    capacity: Amount,
    /// Unix second of the last heartbeat tick charged (0 until the first request).
    last_heartbeat_secs: u64,
}

impl Meter {
    /// A fresh meter for a session priced by `price`, holding a `model_gb`-GB model, with a channel
    /// `capacity`. `cumulative` starts at zero.
    pub fn new(price: PriceSheet, model_gb: u64, capacity: Amount) -> Meter {
        Meter {
            price,
            model_gb,
            cumulative: Amount::ZERO,
            capacity,
            last_heartbeat_secs: 0,
        }
    }

    /// The current monotonic cumulative total (base units) — what the next receipt must cover.
    pub fn cumulative(&self) -> Amount {
        self.cumulative
    }

    /// The channel capacity this meter is bounded by.
    pub fn capacity(&self) -> Amount {
        self.capacity
    }

    /// How much capacity remains before the channel is exhausted.
    pub fn remaining(&self) -> Amount {
        Amount::from_base((self.capacity.base() - self.cumulative.base()).max(0))
    }

    /// True once the cumulative has reached the channel capacity — the session must reopen/refresh.
    pub fn exhausted(&self) -> bool {
        self.cumulative.base() >= self.capacity.base()
    }

    /// Charge one completed request producing `tokens`, taking `seconds` of wall time, and advance
    /// the cumulative. Returns the new cumulative the next receipt must cover. Clamped to capacity
    /// so a receipt is never signed beyond the locked funds.
    pub fn charge_request(&mut self, tokens: u64, seconds: u64) -> Amount {
        let delta = self.price.cost(tokens, self.model_gb, seconds);
        self.advance(delta.base());
        self.cumulative
    }

    /// Charge accrued **heartbeat** intervals for a long-running stream. Given the current unix time
    /// and the tokens produced since the last tick, it charges one heartbeat unit for **each whole
    /// [`HEARTBEAT_INTERVAL_SECS`] elapsed** since the last tick (so a 95s gap bills 3 intervals),
    /// pricing each interval's GB-second residency plus a share of the new tokens. Returns the
    /// number of heartbeat intervals charged (0 if not yet a full interval). Idempotent within an
    /// interval: calling it twice in the same 30s window charges nothing the second time.
    pub fn charge_heartbeats(&mut self, now_secs: u64, tokens_since_last: u64) -> u64 {
        if self.last_heartbeat_secs == 0 {
            // First observation arms the clock without charging (the request charge covers t0).
            self.last_heartbeat_secs = now_secs;
            return 0;
        }
        if now_secs <= self.last_heartbeat_secs {
            return 0;
        }
        let elapsed = now_secs - self.last_heartbeat_secs;
        let intervals = elapsed / HEARTBEAT_INTERVAL_SECS;
        if intervals == 0 {
            return 0;
        }
        // Charge per-GB-second residency across the whole elapsed heartbeat span, plus the new
        // tokens (attributed once, at this tick).
        let charged_secs = intervals * HEARTBEAT_INTERVAL_SECS;
        let delta = self.price.cost(tokens_since_last, self.model_gb, charged_secs).base();
        self.advance(delta);
        // Advance the clock by the whole intervals consumed (keep the sub-interval remainder).
        self.last_heartbeat_secs += charged_secs;
        intervals
    }

    /// Number of whole heartbeat intervals that have elapsed since the clock was armed, without
    /// charging — for bookkeeping/metrics.
    pub fn intervals_elapsed(&self, now_secs: u64) -> u64 {
        if self.last_heartbeat_secs == 0 || now_secs <= self.last_heartbeat_secs {
            return 0;
        }
        (now_secs - self.last_heartbeat_secs) / HEARTBEAT_INTERVAL_SECS
    }

    /// Advance the cumulative by `delta` base units, clamped to [0, capacity].
    fn advance(&mut self, delta: i128) {
        let next = self.cumulative.base().saturating_add(delta.max(0));
        self.cumulative = Amount::from_base(next.min(self.capacity.base()));
    }
}

/// Host-side receipt tracking: the worker keeps the **highest** cumulative a payer has signed for,
/// and redeems exactly that at channel close. Receipts can arrive out of order or be replayed; only
/// a strictly higher cumulative advances the redeemable total.
#[derive(Debug, Clone, Default)]
pub struct HighestReceipt {
    cumulative: Amount,
    payer_sig: String,
}

impl HighestReceipt {
    /// Offer a newly received `(cumulative, payer_sig)`. Accepts it only if it strictly exceeds the
    /// current highest (a higher running total authorized by the payer). Returns true if accepted.
    pub fn offer(&mut self, cumulative: Amount, payer_sig: &str) -> bool {
        if cumulative.base() > self.cumulative.base() {
            self.cumulative = cumulative;
            self.payer_sig = payer_sig.to_string();
            true
        } else {
            false
        }
    }

    /// The highest cumulative seen so far (what `channel_close` should redeem).
    pub fn cumulative(&self) -> Amount {
        self.cumulative
    }

    /// The payer signature that authorizes [`cumulative`](Self::cumulative).
    pub fn payer_sig(&self) -> &str {
        &self.payer_sig
    }

    /// True once at least one receipt has been accepted (there is something to redeem).
    pub fn is_redeemable(&self) -> bool {
        !self.payer_sig.is_empty() && self.cumulative.base() > 0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sheet() -> PriceSheet {
        PriceSheet {
            per_token: Amount::from_base(10),
            per_gb_second: Amount::from_base(2),
            per_request: Amount::from_base(100),
        }
    }

    #[test]
    fn cost_sums_request_token_and_gb_second_components() {
        let s = sheet();
        // 100 (request) + 5*10 (tokens) + 4gb*3s*2 (gb-seconds) = 100 + 50 + 24 = 174
        assert_eq!(s.cost(5, 4, 3).base(), 174);
    }

    #[test]
    fn cost_is_saturating_not_panicking() {
        let s = PriceSheet {
            per_token: Amount::from_base(i128::MAX),
            per_gb_second: Amount::from_base(i128::MAX),
            per_request: Amount::from_base(i128::MAX),
        };
        // Must not panic on overflow.
        let _ = s.cost(u64::MAX, u64::MAX, u64::MAX);
    }

    #[test]
    fn meter_cumulative_is_monotonic_and_capped() {
        let mut m = Meter::new(sheet(), 4, Amount::from_base(1_000));
        let c1 = m.charge_request(5, 3); // 174
        assert_eq!(c1.base(), 174);
        let c2 = m.charge_request(5, 3); // +174 = 348
        assert_eq!(c2.base(), 348);
        assert!(c2.base() > c1.base(), "cumulative must increase");
        // Drive past capacity: it clamps, never exceeds.
        for _ in 0..100 {
            m.charge_request(5, 3);
        }
        assert_eq!(m.cumulative().base(), 1_000);
        assert!(m.exhausted());
        assert_eq!(m.remaining().base(), 0);
    }

    #[test]
    fn heartbeats_charge_one_unit_per_whole_interval() {
        let mut m = Meter::new(sheet(), 4, Amount::from_base(1_000_000));
        // Arm the clock at t=1000 (no charge on first observation).
        assert_eq!(m.charge_heartbeats(1000, 0), 0);
        assert_eq!(m.cumulative().base(), 0);

        // 29s later: still inside the first interval, no charge.
        assert_eq!(m.charge_heartbeats(1029, 0), 0);
        assert_eq!(m.cumulative().base(), 0);

        // 30s after arming: exactly one interval -> one heartbeat charged.
        let n = m.charge_heartbeats(1030, 0);
        assert_eq!(n, 1);
        // cost(0 tokens, 4gb, 30s) = 100 + 0 + 4*30*2 = 100 + 240 = 340
        assert_eq!(m.cumulative().base(), 340);
    }

    #[test]
    fn heartbeats_bill_multiple_elapsed_intervals_at_once() {
        let mut m = Meter::new(sheet(), 1, Amount::from_base(1_000_000));
        m.charge_heartbeats(1000, 0); // arm
        // 95s later -> 3 whole intervals (90s), remainder 5s kept.
        let n = m.charge_heartbeats(1095, 10);
        assert_eq!(n, 3);
        // cost(10 tokens, 1gb, 90s) = 100 + 100 + 1*90*2 = 100 + 100 + 180 = 380
        assert_eq!(m.cumulative().base(), 380);
        // Next call 5s later (t=1100): only 5s since the advanced clock (1090) -> no full interval.
        assert_eq!(m.charge_heartbeats(1100, 0), 0);
    }

    #[test]
    fn heartbeats_are_idempotent_within_an_interval() {
        let mut m = Meter::new(sheet(), 1, Amount::from_base(1_000_000));
        m.charge_heartbeats(1000, 0); // arm
        assert_eq!(m.charge_heartbeats(1040, 0), 1); // one interval
        let after_first = m.cumulative().base();
        // Same window again, no new whole interval -> no charge.
        assert_eq!(m.charge_heartbeats(1041, 0), 0);
        assert_eq!(m.cumulative().base(), after_first);
    }

    #[test]
    fn intervals_elapsed_does_not_charge() {
        let mut m = Meter::new(sheet(), 1, Amount::from_base(1_000_000));
        m.charge_heartbeats(1000, 0); // arm
        assert_eq!(m.intervals_elapsed(1075), 2);
        // intervals_elapsed is read-only: cumulative untouched.
        assert_eq!(m.cumulative().base(), 0);
    }

    #[test]
    fn highest_receipt_only_advances_on_strictly_higher_cumulative() {
        let mut h = HighestReceipt::default();
        assert!(!h.is_redeemable());
        assert!(h.offer(Amount::from_base(100), "sigA"));
        assert_eq!(h.cumulative().base(), 100);
        assert_eq!(h.payer_sig(), "sigA");
        assert!(h.is_redeemable());
        // A lower/equal cumulative is ignored (replay / out-of-order).
        assert!(!h.offer(Amount::from_base(100), "sigStale"));
        assert!(!h.offer(Amount::from_base(50), "sigOld"));
        assert_eq!(h.payer_sig(), "sigA");
        // A strictly higher one advances.
        assert!(h.offer(Amount::from_base(250), "sigB"));
        assert_eq!(h.cumulative().base(), 250);
        assert_eq!(h.payer_sig(), "sigB");
    }

    #[test]
    fn default_sheet_prices_are_positive() {
        let s = PriceSheet::default_sheet();
        assert!(s.per_token.base() > 0);
        assert!(s.per_gb_second.base() > 0);
        assert!(s.per_request.base() > 0);
        // A 100-token, 8GB, 30s request costs a small but non-zero amount.
        assert!(s.cost(100, 8, 30).base() > 0);
    }
}
