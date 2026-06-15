// Paley tournament S_K candidate finder.
//
// Finds the smallest prime n ≡ 3 (mod 4) such that every K vertices in the
// Paley tournament on n vertices have a common dominator.  Uses a fast
// falsification pipeline (structural / random / local search) followed by a
// deterministic verifier.  Persists progress to a bookmark file so the search
// can be paused with Ctrl-C and resumed.
//
// OEIS A362137 indexing:
//   A(1) = 1   (trivial)
//   A(idx) = find_smallest_SK(idx - 1) for idx >= 2.
// Known: 1, 3, 7, 19, 67, 331, 1163  ↔  A(1..7).
//
//   cargo run --release -- A 8                           # find A(8)
//   cargo run --release -- A 8 --lo 2683                 # custom start
//   cargo run --release -- A 8 --bookmark a8.txt         # custom bookmark
//   cargo run --release -- SK 7 --threads 8              # by property index

use rand::seq::SliceRandom;
use rand::{Rng, SeedableRng};
use rand_xoshiro::Xoshiro256PlusPlus;
use rayon::prelude::*;
use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

// ---------------------------------------------------------------- BitSet

#[derive(Clone)]
struct BitSet {
    words: Vec<u64>,
}

impl BitSet {
    #[inline]
    fn zeros(nwords: usize) -> Self {
        Self { words: vec![0u64; nwords] }
    }
    #[inline]
    fn set(&mut self, i: usize) {
        self.words[i >> 6] |= 1u64 << (i & 63);
    }
    #[inline]
    fn copy_from(&mut self, src: &BitSet) {
        self.words.copy_from_slice(&src.words);
    }
    /// dst &= other; returns true iff result is non-empty.
    #[inline]
    fn and_assign(&mut self, other: &BitSet) -> bool {
        let mut acc = 0u64;
        for (a, b) in self.words.iter_mut().zip(&other.words) {
            *a &= b;
            acc |= *a;
        }
        acc != 0
    }
    /// out = a & b; returns true iff result is non-empty.
    #[inline]
    fn and_into(out: &mut BitSet, a: &BitSet, b: &BitSet) -> bool {
        let mut acc = 0u64;
        for ((o, x), y) in out.words.iter_mut().zip(&a.words).zip(&b.words) {
            *o = x & y;
            acc |= *o;
        }
        acc != 0
    }
    #[inline]
    fn popcount(&self) -> u32 {
        self.words.iter().map(|w| w.count_ones()).sum()
    }
}

// ---------------------------------------------------------------- Paley setup

struct Paley {
    n: usize,
    #[allow(dead_code)]
    nwords: usize,
    nb: Vec<BitSet>,         // nb[v] = {x : v - x ∈ QR}
    qr: Vec<bool>,           // qr[i] = i is a QR
    nrs: Vec<usize>,         // sorted nonresidues in 1..n-1
    qr_inv: Vec<usize>,      // qr_inv[q] = q^{-1} mod n if q ∈ QR; 0 otherwise
}

#[inline]
fn mod_pow(base: usize, exp: usize, m: usize) -> usize {
    let mut result = 1u128;
    let mut b = (base % m) as u128;
    let mm = m as u128;
    let mut e = exp;
    while e > 0 {
        if e & 1 == 1 { result = (result * b) % mm; }
        b = (b * b) % mm;
        e >>= 1;
    }
    result as usize
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum CanonicalKind { None, CaseA }

/// Returns false iff there is an alternative QR-arc (a, b) ≠ (S[0]=0, S[1]=1) in
/// S × S whose normalisation g_(a,b)(x) = (x - a) * (b - a)^{-1} mod n produces
/// a sorted list strictly lex-smaller than S — i.e. S is non-canonical and we
/// can prune.  Sound at any internal-node depth: if g(S) < S sorted then
/// g(S ∪ T) < (S ∪ T) sorted for every extension T (the smaller leading element
/// of g(S) is also a leading element of sort(g(S ∪ T))).
#[inline]
fn case_a_canonical(s: &[usize], p: &Paley) -> bool {
    let n = p.n;
    let k = s.len();
    if k < 3 { return true; }
    let mut t = [0usize; 16];
    for i in 0..k {
        let a = s[i];
        for j in 0..k {
            if i == j { continue; }
            if i == 0 && j == 1 { continue; }   // skip the (s[0]=0, s[1]=1) self-arc
            let b = s[j];
            let d = if b >= a { b - a } else { b + n - a };
            if d == 0 || !p.qr[d] { continue; }
            let d_inv = p.qr_inv[d];
            for kk in 0..k {
                let x = if s[kk] >= a { s[kk] - a } else { s[kk] + n - a };
                t[kk] = ((x as u128 * d_inv as u128) % n as u128) as usize;
            }
            (&mut t[..k]).sort_unstable();
            // strict-lex compare: t < s ?
            for kk in 0..k {
                if t[kk] < s[kk] { return false; }
                if t[kk] > s[kk] { break; }
            }
        }
    }
    true
}

impl Paley {
    fn new(n: usize) -> Self {
        debug_assert!(n % 4 == 3 && is_prime(n));
        let nwords = (n + 63) >> 6;
        let mut qr = vec![false; n];
        for i in 1..n {
            qr[(i * i) % n] = true;
        }
        let nrs: Vec<usize> = (1..n).filter(|&j| !qr[j]).collect();

        // nb[0] = NR; nb[v] = nb[0] cyclically rotated left by v.  Build a
        // doubled buffer (nb0 ∥ nb0) and slice out length-n windows for each v.
        let mut nb0 = BitSet::zeros(nwords);
        for &j in &nrs {
            nb0.set(j);
        }

        let src_words = ((2 * n + 63) >> 6) + 1;
        let mut src = vec![0u64; src_words];
        for i in 0..nwords {
            src[i] = nb0.words[i];
        }
        let word_off = n >> 6;
        let bit_off = n & 63;
        if bit_off == 0 {
            for i in 0..nwords {
                src[word_off + i] |= nb0.words[i];
            }
        } else {
            for i in 0..nwords {
                src[word_off + i]     |= nb0.words[i] << bit_off;
                src[word_off + i + 1] |= nb0.words[i] >> (64 - bit_off);
            }
        }

        let last_bits = n & 63;
        let last_mask: u64 = if last_bits == 0 { !0 } else { (1u64 << last_bits) - 1 };

        let mut nb = Vec::with_capacity(n);
        for v in 0..n {
            let mut bs = BitSet::zeros(nwords);
            let start = n - v;          // for v=0, start=n -> reads second copy = nb0 itself
            let wo = start >> 6;
            let bo = start & 63;
            if bo == 0 {
                for i in 0..nwords {
                    bs.words[i] = src[wo + i];
                }
            } else {
                let inv = 64 - bo;
                for i in 0..nwords {
                    bs.words[i] = (src[wo + i] >> bo) | (src[wo + i + 1] << inv);
                }
            }
            if last_bits != 0 {
                bs.words[nwords - 1] &= last_mask;
            }
            nb.push(bs);
        }

        // qr_inv[q] = q^{-1} mod n for q ∈ QR (Fermat: q^{n-2}).  Used by
        // the case-A canonical filter.
        let mut qr_inv = vec![0usize; n];
        for q in 1..n {
            if qr[q] { qr_inv[q] = mod_pow(q, n - 2, n); }
        }

        Paley { n, nwords, nb, qr, nrs, qr_inv }
    }
}

// ---------------------------------------------------------------- primes

fn is_prime(n: usize) -> bool {
    if n < 2 { return false; }
    if n < 4 { return true; }
    if n % 2 == 0 { return false; }
    let mut i = 3usize;
    while i * i <= n {
        if n % i == 0 { return false; }
        i += 2;
    }
    true
}

fn next_prime_3mod4_at_least(n: usize) -> usize {
    let mut m = n.max(3);
    if m % 4 != 3 {
        m += (3 + 4 - (m % 4)) % 4;
    }
    while !is_prime(m) { m += 4; }
    m
}

// ---------------------------------------------------------------- Falsifiers

/// AND-chain check: returns true iff (prefix ∩ ⋂ nb[v] for v in t) is non-empty.
/// `scratch` is a reusable buffer.  Bails the moment the running mask hits zero.
#[inline]
fn has_dominator(p: &Paley, prefix: &BitSet, t: &[usize], scratch: &mut BitSet) -> bool {
    scratch.copy_from(prefix);
    for &v in t {
        if !scratch.and_assign(&p.nb[v]) {
            return false;
        }
    }
    true
}

fn structural_witness(p: &Paley, k: usize) -> Option<Vec<usize>> {
    if k < 2 { return None; }
    let mut scratch = BitSet::zeros(p.nwords);
    // Arithmetic progressions through 0
    for d in 1..p.n {
        scratch.copy_from(&p.nb[0]);
        let mut s = vec![0usize];
        let mut bad = false;
        for j in 1..k {
            let v = (j * d) % p.n;
            s.push(v);
            if !scratch.and_assign(&p.nb[v]) { bad = true; break; }
        }
        if bad { return Some(s); }
    }
    // Geometric progressions {0, 1, g, g², ...}
    for g in 2..p.n {
        scratch.copy_from(&p.nb[0]);
        if !scratch.and_assign(&p.nb[1]) { return Some(vec![0, 1]); }
        let mut s = vec![0usize, 1];
        let mut x = 1usize;
        let mut bad = false;
        for _ in 2..k {
            x = (x * g) % p.n;
            if x == 0 { break; }
            s.push(x);
            if !scratch.and_assign(&p.nb[x]) { bad = true; break; }
        }
        if bad { return Some(s); }
    }
    None
}

/// Seeded-restart falsifier: try a transformation of the previous prime's
/// counterexample, plus small perturbations.  Cheap (~ms).  Counterexamples
/// often cluster across nearby primes, so re-using the prior fail set under
/// {mod n, rescale by n/n_prev, small jitter} catches a chunk of "candidate"
/// primes pre-verify.
fn seeded_witness(
    p: &Paley,
    rng: &mut Xoshiro256PlusPlus,
    k: usize,
    seed: &Option<(usize, Vec<usize>)>,
) -> Option<Vec<usize>> {
    let (n_prev, prev_s) = seed.as_ref()?;
    if k < 2 || prev_s.is_empty() { return None; }
    let n = p.n;
    let mut scratch = BitSet::zeros(p.nwords);
    let mut try_pad = |s_raw: &[usize]| -> Option<Vec<usize>> {
        let mut t: Vec<usize> = Vec::with_capacity(k);
        for &v in s_raw {
            let v = v % n;
            if !t.contains(&v) { t.push(v); }
            if t.len() == k { break; }
        }
        let mut nxt = 0usize;
        while t.len() < k && nxt < n {
            if !t.contains(&nxt) { t.push(nxt); }
            nxt += 1;
        }
        scratch.copy_from(&p.nb[t[0]]);
        for &v in &t[1..] {
            if !scratch.and_assign(&p.nb[v]) { return Some(t); }
        }
        None
    };
    if let Some(s) = try_pad(prev_s) { return Some(s); }
    let scale_num = n as u128;
    let scale_den = *n_prev as u128;
    let mapped: Vec<usize> = prev_s.iter()
        .map(|&v| ((v as u128 * scale_num) / scale_den) as usize % n)
        .collect();
    if let Some(s) = try_pad(&mapped) { return Some(s); }
    for _ in 0..16 {
        let jittered: Vec<usize> = prev_s.iter().map(|&v| {
            let j = rng.gen_range(0..9i64) - 4;
            ((v as i64 + j).rem_euclid(n as i64)) as usize
        }).collect();
        if let Some(s) = try_pad(&jittered) { return Some(s); }
    }
    None
}

fn random_witness(p: &Paley, rng: &mut Xoshiro256PlusPlus, k: usize, budget: u64) -> Option<Vec<usize>> {
    if k < 2 { return None; }
    if p.n - 1 < k - 1 { return None; }
    let mut scratch = BitSet::zeros(p.nwords);
    let mut s: Vec<usize> = Vec::with_capacity(k - 1);
    let want = k - 1;
    for _ in 0..budget {
        // Fast distinct sampling: rejection sampling is O(k) for k ≪ n.
        s.clear();
        while s.len() < want {
            let c = rng.gen_range(1..p.n);
            if !s.contains(&c) { s.push(c); }
        }
        if !has_dominator(p, &p.nb[0], &s, &mut scratch) {
            let mut out = vec![0usize];
            out.extend(&s);
            return Some(out);
        }
    }
    None
}

/// Targeted "consecutive-pair" structural sweep: for each pair (a, b) with
/// b - a in 1..=PAIR_DELTA_MAX, build {0, 1, a, b} and greedily extend with
/// elements that minimise popcount.  Catches the very common pattern where
/// counterexamples include a near-consecutive pair like (1007, 1008).
const PAIR_DELTA_MAX: usize = 12;
const PAIR_GREEDY_TRIALS: usize = 32;
fn pair_witness(p: &Paley, rng: &mut Xoshiro256PlusPlus, k: usize) -> Option<Vec<usize>> {
    if k < 4 { return None; }
    let mut m = BitSet::zeros(p.nwords);
    let mut trial = BitSet::zeros(p.nwords);
    for a in 2..p.n - 1 {
        for delta in 1..=PAIR_DELTA_MAX {
            let b = a + delta;
            if b >= p.n { break; }
            // {0, 1, a, b}
            m.copy_from(&p.nb[0]);
            if !m.and_assign(&p.nb[1]) { return Some(vec![0, 1]); }
            if !m.and_assign(&p.nb[a]) {
                let mut s = vec![0, 1, a]; pad(&mut s, p.n, k); return Some(s);
            }
            if !m.and_assign(&p.nb[b]) {
                let mut s = vec![0, 1, a, b]; pad(&mut s, p.n, k); return Some(s);
            }
            // Greedy extension: add (k-4) more elements, each chosen from a
            // small random sample to minimise running popcount.
            let mut s = vec![0, 1, a, b];
            let mut ok = true;
            for _ in 4..k {
                let mut best_v: usize = 0;
                let mut best_pop = u32::MAX;
                for _ in 0..PAIR_GREEDY_TRIALS {
                    let c = rng.gen_range(2..p.n);
                    if s.contains(&c) { continue; }
                    BitSet::and_into(&mut trial, &m, &p.nb[c]);
                    let pp = trial.popcount();
                    if pp < best_pop { best_pop = pp; best_v = c; }
                }
                if best_v == 0 { ok = false; break; }
                s.push(best_v);
                if !m.and_assign(&p.nb[best_v]) {
                    pad(&mut s, p.n, k);
                    return Some(s);
                }
            }
            // Even if not collapsed, keep going through other (a, b)
            let _ = ok;
        }
    }
    None
}

fn pad(s: &mut Vec<usize>, n: usize, k: usize) {
    let mut nxt = 2usize;
    while s.len() < k && nxt < n {
        if !s.contains(&nxt) { s.push(nxt); }
        nxt += 1;
    }
}

fn local_witness(p: &Paley, rng: &mut Xoshiro256PlusPlus, k: usize, restarts: u32, max_swaps: u32) -> Option<Vec<usize>> {
    if k < 2 { return None; }
    if p.n - 1 < k - 1 { return None; }
    let want = k - 1;
    let mut m = BitSet::zeros(p.nwords);
    let mut without = BitSet::zeros(p.nwords);
    let mut trial = BitSet::zeros(p.nwords);
    let mut s: Vec<usize> = Vec::with_capacity(want);
    for _ in 0..restarts {
        s.clear();
        while s.len() < want {
            let c = rng.gen_range(1..p.n);
            if !s.contains(&c) { s.push(c); }
        }
        m.copy_from(&p.nb[0]);
        let mut good = true;
        for &v in &s {
            if !m.and_assign(&p.nb[v]) { good = false; break; }
        }
        if !good {
            let mut out = vec![0usize]; out.extend(&s); return Some(out);
        }
        for _ in 0..max_swaps {
            let i = rng.gen_range(0..want);
            without.copy_from(&p.nb[0]);
            for (j, &v) in s.iter().enumerate() {
                if j == i { continue; }
                without.and_assign(&p.nb[v]);
            }
            let mut best_v = s[i];
            let mut best_pop = u32::MAX;
            for _ in 0..16 {
                let c = rng.gen_range(1..p.n);
                if s.contains(&c) { continue; }
                BitSet::and_into(&mut trial, &without, &p.nb[c]);
                let pp = trial.popcount();
                if pp < best_pop { best_pop = pp; best_v = c; }
            }
            if best_v != s[i] {
                s[i] = best_v;
                m.copy_from(&p.nb[0]);
                let mut ok = true;
                for &v in &s {
                    if !m.and_assign(&p.nb[v]) { ok = false; break; }
                }
                if !ok {
                    let mut out = vec![0usize]; out.extend(&s); return Some(out);
                }
            }
        }
    }
    None
}

// ---------------------------------------------------------------- Verifier

/// Recursive ordered DFS: at each level, score every available pool element by
/// popcount(prefix ∩ nb[v]) and descend in ascending order.  This is greedy
/// "head toward the empty mask" — for primes that *do* have a counterexample,
/// it finds one in O(depth) descents instead of O(C(n, depth)) leaves.
///
/// For primes with no counterexample this enumerates the same sorted subsets
/// as the naive DFS but pays the per-level sort overhead.  The caller is
/// expected to bound this case with `deadline` (otherwise true-pass primes can
/// take effectively forever at K ≥ 7).
///
/// Returns Some(t) if a length-`depth` t collapses prefix to empty (a
/// counterexample), or None for "no counterexample found in this subtree
/// before abort/deadline".
fn enumerate_serial(
    p: &Paley,
    prefix: &BitSet,
    pool: &[usize],
    depth: usize,
    cur_subset: &mut Vec<usize>,
    canonical: CanonicalKind,
    abort: &AtomicBool,
    interrupted: &AtomicBool,
    deadline: Option<Instant>,
) -> Option<Vec<usize>> {
    if depth == 0 {
        return if prefix.popcount() == 0 { Some(vec![]) } else { None };
    }
    if pool.len() < depth { return None; }
    if abort.load(Ordering::Relaxed) || interrupted.load(Ordering::Relaxed) { return None; }
    if let Some(dl) = deadline {
        if Instant::now() >= dl {
            abort.store(true, Ordering::Relaxed);
            return None;
        }
    }

    // Score each candidate by popcount(prefix ∩ nb[v]); sort ascending.
    let mut tmp = BitSet::zeros(p.nwords);
    let mut scored: Vec<(u32, usize)> = Vec::with_capacity(pool.len());
    for (i, &v) in pool.iter().enumerate() {
        BitSet::and_into(&mut tmp, prefix, &p.nb[v]);
        scored.push((tmp.popcount(), i));
    }
    scored.sort_unstable();

    for &(pop, idx) in &scored {
        if abort.load(Ordering::Relaxed) { return None; }
        let v = pool[idx];
        cur_subset.push(v);
        // Canonical-form prune (orbit deduplication under the affine group)
        let cf_ok = match canonical {
            CanonicalKind::None => true,
            CanonicalKind::CaseA => case_a_canonical(cur_subset, p),
        };
        if !cf_ok { cur_subset.pop(); continue; }

        if pop == 0 {
            let mut t = vec![v];
            let mut nxt = idx + 1;
            while t.len() < depth {
                if nxt >= pool.len() { cur_subset.pop(); return None; }
                t.push(pool[nxt]); nxt += 1;
            }
            cur_subset.pop();
            return Some(t);
        }
        if depth == 1 { cur_subset.pop(); continue; }
        let mut new_prefix = prefix.clone();
        new_prefix.and_assign(&p.nb[v]);
        let sub_pool = &pool[idx + 1..];
        if let Some(sub) = enumerate_serial(p, &new_prefix, sub_pool, depth - 1, cur_subset, canonical, abort, interrupted, deadline) {
            cur_subset.pop();
            let mut t = vec![v]; t.extend(sub);
            return Some(t);
        }
        cur_subset.pop();
    }
    None
}

/// Parallel verifier: split on the first element of t, in popcount-ascending
/// order so that low-popcount branches start first.  Short-circuit on any
/// branch that finds a counterexample.
fn enumerate_parallel(
    p: &Paley,
    prefix: &BitSet,
    pool: &[usize],
    depth: usize,
    initial_subset: &[usize],   // [0,1] for case A, [0,g] for case B
    canonical: CanonicalKind,
    abort: &Arc<AtomicBool>,
    interrupted: &Arc<AtomicBool>,
    deadline: Option<Instant>,
) -> Option<Vec<usize>> {
    if depth == 0 {
        return if prefix.popcount() == 0 { Some(vec![]) } else { None };
    }
    let pool_len = pool.len();
    if depth > pool_len { return None; }
    let n_top = pool_len - (depth - 1);

    let mut top_scored: Vec<(u32, usize)> = (0..n_top)
        .map(|i| {
            let mut tmp = BitSet::zeros(p.nwords);
            BitSet::and_into(&mut tmp, prefix, &p.nb[pool[i]]);
            (tmp.popcount(), i)
        })
        .collect();
    top_scored.sort_unstable();

    let result = top_scored.par_iter().find_map_any(|&(pop, i)| {
        if abort.load(Ordering::Relaxed) || interrupted.load(Ordering::Relaxed) { return None; }
        if let Some(dl) = deadline {
            if Instant::now() >= dl {
                abort.store(true, Ordering::Relaxed);
                return None;
            }
        }
        let v = pool[i];

        // Canonical check on partial [initial_subset, v]
        let mut cur_subset: Vec<usize> = Vec::with_capacity(initial_subset.len() + depth);
        cur_subset.extend_from_slice(initial_subset);
        cur_subset.push(v);
        let cf_ok = match canonical {
            CanonicalKind::None => true,
            CanonicalKind::CaseA => case_a_canonical(&cur_subset, p),
        };
        if !cf_ok { return None; }

        if pop == 0 {
            let mut t = vec![v];
            let mut nxt = i + 1;
            while t.len() < depth {
                if nxt >= pool_len { return None; }
                t.push(pool[nxt]); nxt += 1;
            }
            abort.store(true, Ordering::Relaxed);
            return Some(t);
        }
        if depth == 1 { return None; }
        let mut new_prefix = prefix.clone();
        new_prefix.and_assign(&p.nb[v]);
        let sub = enumerate_serial(
            p, &new_prefix, &pool[i + 1..], depth - 1,
            &mut cur_subset, canonical, abort, interrupted, deadline,
        )?;
        abort.store(true, Ordering::Relaxed);
        let mut t = vec![v]; t.extend(sub);
        Some(t)
    });
    result
}

enum VerifyResult {
    Fail(Vec<usize>),
    Pass,
    Timeout,
    Interrupted,
}

fn verify(p: &Paley, k: usize, deadline: Option<Instant>, interrupted: &Arc<AtomicBool>) -> VerifyResult {
    if k == 0 { return VerifyResult::Pass; }
    if k == 1 {
        return if p.nb[0].popcount() == 0 { VerifyResult::Fail(vec![0]) }
               else { VerifyResult::Pass };
    }
    let g = p.nrs[0];
    // sibling-abort: set by any branch that finds a counterexample (or by
    // Ctrl-C / deadline) so other branches give up early.
    let abort = Arc::new(AtomicBool::new(false));

    // Case A: S ⊇ {0, 1}.  McKay canonical filter is currently disabled — it
    // empirically loses to ordered DFS on false positives at our K (the
    // popcount-greedy path keeps hitting non-canonical witnesses that get
    // pruned, so we re-explore far more of the tree).  Kept as scaffolding.
    let mut pa = p.nb[0].clone(); pa.and_assign(&p.nb[1]);
    let pool_a: Vec<usize> = (2..p.n).collect();
    if let Some(t) = enumerate_parallel(
        p, &pa, &pool_a, k - 2, &[0, 1], CanonicalKind::None,
        &abort, interrupted, deadline,
    ) {
        let mut s = vec![0usize, 1]; s.extend(t);
        return VerifyResult::Fail(s);
    }
    if interrupted.load(Ordering::Relaxed) { return VerifyResult::Interrupted; }
    if let Some(dl) = deadline { if Instant::now() >= dl { return VerifyResult::Timeout; } }

    // Case B: S ⊇ {0, g}, S \ {0} ⊆ NR.  No canonical filter for now (smaller pool).
    abort.store(false, Ordering::Relaxed);
    let mut pb = p.nb[0].clone(); pb.and_assign(&p.nb[g]);
    let pool_b: Vec<usize> = p.nrs.iter().skip(1).copied().collect();
    if let Some(t) = enumerate_parallel(
        p, &pb, &pool_b, k - 2, &[0, g], CanonicalKind::None,
        &abort, interrupted, deadline,
    ) {
        let mut s = vec![0usize, g]; s.extend(t);
        return VerifyResult::Fail(s);
    }
    if interrupted.load(Ordering::Relaxed) { return VerifyResult::Interrupted; }
    if let Some(dl) = deadline { if Instant::now() >= dl { return VerifyResult::Timeout; } }

    VerifyResult::Pass
}

// ---------------------------------------------------------------- Bookmark

#[derive(Clone, Debug)]
enum Bookmark {
    Empty,
    Pending(usize),
    Done(usize),
}

fn read_bookmark(path: &Path) -> Bookmark {
    let Ok(text) = fs::read_to_string(path) else { return Bookmark::Empty; };
    let t = text.trim();
    if let Some(rest) = t.strip_prefix("DONE:") {
        if let Ok(n) = rest.trim().parse::<usize>() { return Bookmark::Done(n); }
    }
    if let Ok(n) = t.parse::<usize>() { return Bookmark::Pending(n); }
    Bookmark::Empty
}

fn write_bookmark(path: &Path, n: usize) {
    let _ = fs::write(path, n.to_string());
}

fn write_done(path: &Path, n: usize) {
    let _ = fs::write(path, format!("DONE: {}", n));
}

// ---------------------------------------------------------------- Driver

#[derive(Clone, Copy, Debug)]
enum Mode { OEIS, SK }

struct Args {
    mode: Mode,
    target: usize,
    lo: Option<usize>,
    hi: Option<usize>,
    sample_budget: u64,
    threads: Option<usize>,
    bookmark: Option<PathBuf>,
    verify_timeout_secs: u64,
}

fn parse_args() -> Args {
    let mut argv = env::args().skip(1);
    let cmd = argv.next().unwrap_or_else(|| "A".to_string());
    let mode = match cmd.as_str() {
        "A" | "a" | "oeis" => Mode::OEIS,
        "SK" | "sk" => Mode::SK,
        _ => panic!("first arg must be `A` (OEIS index) or `SK` (property index)"),
    };
    let target: usize = argv.next().expect("need target index").parse().expect("target must be integer");
    let mut lo = None;
    let mut hi = None;
    let mut sample_budget: u64 = 100_000;
    let mut threads = None;
    let mut bookmark = None;
    let mut verify_timeout_secs: u64 = 0;   // 0 = no deadline; verify runs to completion
    while let Some(flag) = argv.next() {
        match flag.as_str() {
            "--lo" => lo = Some(argv.next().expect("--lo N").parse().unwrap()),
            "--hi" => hi = Some(argv.next().expect("--hi N").parse().unwrap()),
            "--budget" => sample_budget = argv.next().expect("--budget N").parse().unwrap(),
            "--threads" => threads = Some(argv.next().expect("--threads N").parse().unwrap()),
            "--bookmark" => bookmark = Some(PathBuf::from(argv.next().expect("--bookmark PATH"))),
            "--verify-timeout" => verify_timeout_secs = argv.next().expect("--verify-timeout SEC").parse().unwrap(),
            other => panic!("unknown flag {}", other),
        }
    }
    Args { mode, target, lo, hi, sample_budget, threads, bookmark, verify_timeout_secs }
}

/// Find smallest prime n ≡ 3 (mod 4) with property S_k.
/// Returns Some(n) if found (PASS or TIMEOUT-CANDIDATE), None if interrupted.
fn find_smallest_sk(
    k: usize,
    lo: usize,
    hi: usize,
    sample_budget: u64,
    verify_timeout_secs: u64,
    bookmark_path: &Path,
    interrupted: &Arc<AtomicBool>,
) -> Option<usize> {
    if k == 0 { return Some(1); }
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(0xC0_FF_EE);
    let mut seed: Option<(usize, Vec<usize>)> = None;

    // Resume from bookmark if it points past `lo`
    let mut start = lo;
    match read_bookmark(bookmark_path) {
        Bookmark::Pending(saved) if saved > start => {
            println!("[resume] bookmark says n={}, lo was {}", saved, start);
            start = saved;
        }
        Bookmark::Done(n) => {
            println!("[done] bookmark already says A = {}", n);
            return Some(n);
        }
        _ => {}
    }

    let mut n = next_prime_3mod4_at_least(start);
    while n < hi {
        if interrupted.load(Ordering::Relaxed) {
            println!("[interrupt] stopping at n={} (bookmark saved)", n);
            return None;
        }
        write_bookmark(bookmark_path, n);
        let t0 = Instant::now();
        let p = Paley::new(n);
        let stage: &str;
        let elapsed_after;

        if k >= 2 {
            if let Some(s) = seeded_witness(&p, &mut rng, k, &seed) {
                seed = Some((n, s));
                stage = "seeded";
                elapsed_after = t0.elapsed();
                println!("n={:>6}  fail ({:<10})  {:.3}s", n, stage, elapsed_after.as_secs_f64());
                n = next_prime_3mod4_at_least(n + 4);
                continue;
            }
        }
        if k >= 2 && structural_witness(&p, k).is_some() {
            stage = "structural";
            elapsed_after = t0.elapsed();
        } else if k >= 4 && pair_witness(&p, &mut rng, k).is_some() {
            stage = "pair";
            elapsed_after = t0.elapsed();
        } else if k >= 2 && random_witness(&p, &mut rng, k, sample_budget).is_some() {
            stage = "random";
            elapsed_after = t0.elapsed();
        } else if k >= 2 && local_witness(&p, &mut rng, k, 1000, 100).is_some() {
            stage = "local";
            elapsed_after = t0.elapsed();
        } else {
            // verify with deadline
            print!("n={:>6}  candidate, verifying ...  ", n);
            use std::io::Write as _;
            std::io::stdout().flush().ok();
            let deadline = if verify_timeout_secs > 0 {
                Some(Instant::now() + Duration::from_secs(verify_timeout_secs))
            } else { None };
            match verify(&p, k, deadline, interrupted) {
                VerifyResult::Pass => {
                    println!("PASS  ({:.3}s)", t0.elapsed().as_secs_f64());
                    write_done(bookmark_path, n);
                    return Some(n);
                }
                VerifyResult::Timeout => {
                    println!("TIMEOUT after {}s — bookmarking and continuing past", verify_timeout_secs);
                    // Don't mark DONE; bookmark stays at this n so we can re-verify later
                    // with a longer deadline.  Continue past for now to find more candidates.
                    n = next_prime_3mod4_at_least(n + 4);
                    continue;
                }
                VerifyResult::Interrupted => {
                    println!("interrupted mid-verify");
                    return None;
                }
                VerifyResult::Fail(s) => {
                    stage = "verify";
                    elapsed_after = t0.elapsed();
                    println!("fail @ {:?}  ({:.3}s)", s, elapsed_after.as_secs_f64());
                    seed = Some((n, s));
                }
            }
        }
        if stage != "verify" {
            println!("n={:>6}  fail ({:<10})  {:.3}s", n, stage, elapsed_after.as_secs_f64());
        }
        n = next_prime_3mod4_at_least(n + 4);
    }
    None
}

fn weil_hi(k: usize) -> usize {
    // Ceiling above which Weil bound proves S_k unconditionally.
    if k <= 1 { 8 } else { k * k * 4usize.pow((k - 1) as u32) }
}

fn lower_bound(k: usize) -> usize {
    if k == 0 { 1 } else { 1usize << k.saturating_sub(1) }
}

fn main() {
    let args = parse_args();

    if let Some(t) = args.threads {
        rayon::ThreadPoolBuilder::new().num_threads(t).build_global().unwrap();
    }

    // Translate target -> property index `k` and label.
    let (k, label) = match args.mode {
        Mode::OEIS => {
            // A(idx) = find_smallest_SK(idx-1)
            if args.target == 0 { panic!("OEIS index must be >= 1"); }
            if args.target == 1 {
                println!("A(1) = 1   (trivial)");
                return;
            }
            (args.target - 1, format!("A({})", args.target))
        }
        Mode::SK => (args.target, format!("SK({})", args.target)),
    };

    let lo = args.lo.unwrap_or_else(|| lower_bound(k));
    let hi = args.hi.unwrap_or_else(|| weil_hi(k));
    let bookmark = args.bookmark.unwrap_or_else(|| {
        PathBuf::from(format!("bookmark_{}.txt",
            match args.mode { Mode::OEIS => format!("a{}", args.target),
                              Mode::SK   => format!("sk{}", args.target) }))
    });

    println!("[paley] {} = find_smallest_SK({})", label, k);
    println!("[paley] threads = {}, lo = {}, hi = {}, budget = {}, verify_timeout = {}s, bookmark = {}",
             rayon::current_num_threads(), lo, hi, args.sample_budget,
             args.verify_timeout_secs, bookmark.display());

    let interrupted = Arc::new(AtomicBool::new(false));
    let int_clone = interrupted.clone();
    ctrlc::set_handler(move || {
        int_clone.store(true, Ordering::Relaxed);
        eprintln!("\n[ctrl-c] requesting graceful stop after current prime ...");
    }).ok();

    let t0 = Instant::now();
    match find_smallest_sk(k, lo, hi, args.sample_budget, args.verify_timeout_secs, &bookmark, &interrupted) {
        Some(n) => println!("\n{} = {}   (total {:.3}s)", label, n, t0.elapsed().as_secs_f64()),
        None => println!("\n[paley] no answer found (or interrupted)"),
    }
}
