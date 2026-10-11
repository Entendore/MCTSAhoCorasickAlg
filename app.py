# ============================================================================
# AC-MCTS v9  (v8 + 4 features)
#   VocabAligner   : statistic-proposed, human-curated explicit token maps
#   RiskBudget     : episode-level per-bucket CVaR; beta escalation + option
#                    suspension, hysteresis recovery
#   GrammarGenEnv  : compositional S->NP VP pushdown domain; per-slot buckets
#   port_multi     : several source stores, majority-sign consensus weighting
#   (BatchComboGrid, DuelGrid, fork-parallel: unchanged from v7/v8, omitted)
# ============================================================================
import json, math, random, time
from collections import deque, defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

INF = float("inf")
STORE_FORMAT, AUTO_FORMAT, PLANNER_FORMAT, VERIFY_FORMAT, FORMAT_VERSION = \
    "ac-mcts-pattern-store", "ac-mcts-automaton", "ac-mcts-planner", "ac-mcts-verifier", 9


class Kind(Enum):
    NEUTRAL = 0
    SUCCESS = 1
    FAILURE = -1


def _enc(b): return list(b) if isinstance(b, tuple) else b
def _dec(b): return tuple(b) if isinstance(b, list) else b
def replay(ac, ctx):
    q = 0
    for t in ctx:
        q = ac.step(q, t)
    return q
def subtree_size(node):
    seen, stack, n = set(), [node], 0
    while stack:
        x = stack.pop()
        if id(x) in seen:
            continue
        seen.add(id(x)); n += 1
        stack.extend(e[2] for e in x.children.values())
    return n
def _fp(patterns):
    return frozenset((tuple(p), round(w, 3), k, round(c, 3)) for p, w, k, c in patterns)


# ---------------------------------------------------------------------------
# 1. Aho-Corasick — unchanged from v8 (typed, dense, distances, VI, wildcard-src)
# ---------------------------------------------------------------------------
class AhoCorasick:
    CONF_MIN = 0.25

    def __init__(self) -> None:
        self._g: List[Dict[Any, int]] = [{}]
        self._f: List[int] = [0]
        self._o: List[List[int]] = [[]]
        self._p: List[Tuple[tuple, float, Kind, float]] = []
        self._src: List[Optional[tuple]] = [None]
        self._s: List[float] = [0.0]
        self._kind: List[Kind] = [Kind.NEUTRAL]
        self._conf: List[float] = [0.0]
        self._d_fail: List[float] = [INF]
        self._d_succ: List[float] = [INF]
        self._par: List[int] = [0]
        self._cht: List[Any] = [""]
        self._depth: List[int] = [0]
        self._sigma: set = set()
        self._toks: List[Any] = []
        self._tok_idx: Dict[Any, int] = {}
        self._trans: np.ndarray = np.zeros((1, 0), dtype=np.int32)
        self._vi: Optional[np.ndarray] = None
        self._vi_key: Optional[tuple] = None
        self.verified = False
        self.version = 0

    def reset(self) -> None:
        self.__init__()

    def add_pattern(self, tokens, weight: float, outcome: Kind = Kind.NEUTRAL,
                    confidence: float = 0.0, source: Optional[tuple] = None) -> int:
        tokens = tuple(tokens)
        pid = len(self._p)
        v = 0
        for t in tokens:
            self._sigma.add(t)
            nxt = self._g[v].get(t)
            if nxt is None:
                self._g.append({}); self._f.append(0); self._o.append([])
                self._s.append(0.0); self._kind.append(Kind.NEUTRAL)
                self._conf.append(0.0); self._d_fail.append(INF); self._d_succ.append(INF)
                self._par.append(v); self._cht.append(t); self._depth.append(0)
                self._src.append(None)
                nxt = len(self._g) - 1
                self._g[v][t] = nxt
            v = nxt
        self._o[v].append(pid)
        self._p.append((tokens, float(weight), outcome, float(confidence)))
        self._src.append(source if source is not None else tokens)
        return pid

    def build(self) -> None:
        q: deque = deque()
        for v in self._g[0].values():
            self._f[v] = 0; self._depth[v] = 1; q.append(v)
        while q:
            u = q.popleft()
            self._o[u].extend(self._o[self._f[u]])
            self._s[u] = sum(self._p[pid][1] for pid in self._o[u])
            for t, v in self._g[u].items():
                f = self._f[u]
                while f and t not in self._g[f]:
                    f = self._f[f]
                self._f[v] = self._g[f].get(t, 0)
                if self._f[v] == v:
                    self._f[v] = 0
                self._depth[v] = self._depth[u] + 1
                q.append(v)
        self._type_states(); self._build_dense(); self._compute_distances()
        self.version += 1

    def _type_states(self) -> None:
        for v in range(len(self._g)):
            fconf = [self._p[i][3] for i in self._o[v]
                     if self._p[i][2] is Kind.FAILURE and self._p[i][3] >= self.CONF_MIN]
            sconf = [self._p[i][3] for i in self._o[v]
                     if self._p[i][2] is Kind.SUCCESS and self._p[i][3] >= self.CONF_MIN]
            if fconf:   self._kind[v], self._conf[v] = Kind.FAILURE, max(fconf)
            elif sconf: self._kind[v], self._conf[v] = Kind.SUCCESS, max(sconf)
            else:       self._kind[v], self._conf[v] = Kind.NEUTRAL, 0.0

    def _build_dense(self) -> None:
        self._toks = sorted(self._sigma, key=str)
        self._tok_idx = {t: i for i, t in enumerate(self._toks)}
        n, m = len(self._g), len(self._toks)
        trans = np.zeros((n, m), dtype=np.int32)
        for v in sorted(range(n), key=lambda x: self._depth[x]):
            for i in range(m):
                u = self._g[v].get(self._toks[i])
                trans[v, i] = u if u is not None else \
                    (0 if v == 0 else int(trans[self._f[v], i]))
        self._trans = trans

    def _compute_distances(self) -> None:
        n = len(self._g)
        rev: List[List[int]] = [[] for _ in range(n)]
        for v in range(n):
            for t in self._toks:
                rev[self.step(v, t)].append(v)
        self._d_fail = self._bfs(rev, Kind.FAILURE)
        self._d_succ = self._bfs(rev, Kind.SUCCESS)

    def _bfs(self, rev, target: Kind) -> List[float]:
        d = [INF] * len(self._g); q: deque = deque()
        for v, k in enumerate(self._kind):
            if k is target:
                d[v] = 0; q.append(v)
        while q:
            u = q.popleft()
            for p in rev[u]:
                if d[p] > d[u] + 1:
                    d[p] = d[u] + 1; q.append(p)
        return d

    def step(self, v: int, token) -> int:
        i = self._tok_idx.get(token)
        return 0 if i is None else int(self._trans[v, i])

    def score(self, v): return self._s[v]
    def kind(self, v): return self._kind[v]
    def conf(self, v): return self._conf[v]

    def dist(self, v, t: Kind):
        return self._d_fail[v] if t is Kind.FAILURE else self._d_succ[v]

    def proximity(self, v, t: Kind) -> float:
        d = self.dist(v, t)
        return 0.0 if math.isinf(d) else 1.0 / (1.0 + d)

    def patterns_at(self, v): return [self._p[i][0] for i in self._o[v]]
    def pattern_weights(self, v): return [(self._p[i][0], self._p[i][1]) for i in self._o[v]]
    def patterns_src_at(self, v):
        return [self._src[i] if self._src[i] is not None else self._p[i][0]
                for i in self._o[v]]
    def pattern_list(self): return [(t, w, k.name, c) for (t, w, k, c) in self._p]

    def value_iteration(self, gamma=0.9, iters=80, tol=1e-6) -> np.ndarray:
        key = (self.version, gamma)
        if self._vi_key != key:
            V = np.zeros(len(self._g))
            tr, s = self._trans, np.asarray(self._s, dtype=np.float64)
            for _ in range(iters):
                Vn = (s[tr] + gamma * V[tr]).max(axis=1)
                if float(np.max(np.abs(Vn - V))) < tol:
                    V = Vn; break
                V = Vn
            self._vi, self._vi_key = V, key
        return self._vi

    def lookahead(self, v: int, h: int, gamma: float = 1.0) -> float:
        if h <= 0 or not self._toks:
            return 0.0
        best = 0.0
        for t in self._toks:
            v2 = self.step(v, t)
            best = max(best, self._s[v2] + gamma * self.lookahead(v2, h - 1, gamma))
        return best

    def label(self, v: int, maxlen: int = 10) -> str:
        out = []
        while v and len(out) < maxlen:
            out.append(str(self._cht[v])); v = self._par[v]
        s = "".join(reversed(out))
        return s + ("…" if v else "")

    def kind_counts(self):
        return (sum(1 for k in self._kind if k is Kind.SUCCESS),
                sum(1 for k in self._kind if k is Kind.FAILURE))

    @property
    def n_states(self): return len(self._g)

    def save(self, path: str) -> None:
        data = {"format": AUTO_FORMAT, "version": FORMAT_VERSION,
                "patterns": [{"tokens": list(t), "weight": w, "kind": k.name,
                              "conf": c,
                              "source": (list(s) if s is not None and
                                         any(x is None for x in s) else None)}
                             for (t, w, k, c), s in zip(self._p, self._src)],
                "tokens": list(self._toks), "trans": self._trans.tolist()}
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, path: str, verify: bool = True) -> "AhoCorasick":
        with open(path) as f:
            data = json.load(f)
        ac = cls()
        for pd in data["patterns"]:
            src = tuple(pd["source"]) if pd.get("source") else None
            ac.add_pattern(pd["tokens"], pd["weight"], Kind[pd["kind"]],
                           pd["conf"], source=src)
        ac.build(); ac.verified = True
        if verify and data.get("tokens") and data.get("trans"):
            saved = np.array(data["trans"], dtype=np.int32)
            ac.verified = bool(list(map(str, ac._toks)) == list(map(str, data["tokens"]))
                               and saved.shape == ac._trans.shape
                               and (saved == ac._trans).all())
        return ac


# ---------------------------------------------------------------------------
# 2. Store (kernel share, regression, wildcards)  ★ + port_multi
# ---------------------------------------------------------------------------
@dataclass
class _Evidence:
    pos: float = 0.0
    neg: float = 0.0


def _bucket_dist(a, b):
    if isinstance(a, tuple) and isinstance(b, tuple) and len(a) == len(b):
        return sum(abs(x - y) for x, y in zip(a, b))
    return None


class PatternStore:
    def __init__(self, min_len=2, max_len=3, focus=6, min_abs_reward=1.0,
                 min_obs=3.0, dominance=0.70, conf_min_pat=0.30,
                 gain=1.0, decay=0.97, w_threshold=0.12,
                 credit_window=2, credit=0.5, global_share=0.25, beta_z=1.0,
                 use_regression=True, regress_min_steps=8, regress_min_rows=10,
                 ridge_lambda=1.0, t_cap=3.0, kernel_sigma=0.0):
        self.min_len, self.max_len = min_len, max_len
        self.focus, self.min_abs_reward = focus, min_abs_reward
        self.min_obs, self.dominance, self.conf_min_pat = min_obs, dominance, conf_min_pat
        self.gain, self.decay, self.w_threshold = gain, decay, w_threshold
        self.credit_window, self.credit = credit_window, credit
        self.global_share, self.beta_z = global_share, beta_z
        self.use_regression = use_regression
        self.regress_min_steps, self.regress_min_rows = regress_min_steps, regress_min_rows
        self.ridge_lambda, self.t_cap = ridge_lambda, t_cap
        self.kernel_sigma = kernel_sigma
        self.ev: Dict[Tuple[Any, tuple], _Evidence] = {}
        self.typed: Dict[Tuple[Any, tuple], Tuple[float, Kind, float]] = {}
        self.last_reg: Dict[tuple, Tuple[float, float]] = {}

    def disp(self, pat) -> str:
        return "".join("?" if t is None else str(t) for t in pat)

    def buckets(self):
        return {b for (b, _) in self.ev}

    def alphabet(self) -> set:
        return {t for (_, pat) in self.ev for t in pat if t is not None}

    def learn_episode(self, tokens, rewards, sigs=None, final_return=None) -> None:
        sigs = list(sigs) if sigs else ["G"] * len(tokens)
        T = len(tokens)
        self.last_reg = {}
        vocab: List[tuple] = []
        seen = set()
        for i in range(T):
            for n in range(self.min_len, self.max_len + 1):
                if i - n + 1 >= 0:
                    pat = tuple(tokens[i - n + 1:i + 1])
                    if pat not in seen:
                        seen.add(pat); vocab.append(pat)
        if self.use_regression and T >= self.regress_min_steps and vocab:
            rows: Dict[Any, List[int]] = defaultdict(list)
            for t, b in enumerate(sigs):
                rows[b].append(t)
            pooled: List[int] = []
            used = False
            for b, idxs in rows.items():
                if len(idxs) >= self.regress_min_rows:
                    self._regress(tokens, rewards, vocab, idxs, b)
                    used = True
                else:
                    pooled.extend(idxs)
            if pooled:
                self._regress(tokens, rewards, vocab, pooled, ACPool.GLOBAL)
                used = True
            if used:
                return
        self._legacy_learn(tokens, rewards, sigs, final_return)

    def _legacy_learn(self, tokens, rewards, sigs, final_return) -> None:
        hot = sorted(range(len(tokens)), key=lambda i: -abs(rewards[i]))[: self.focus]
        for i in hot:
            if abs(rewards[i]) >= self.min_abs_reward:
                self._credit(tokens, i, 1.0 if rewards[i] > 0 else -1.0, sigs[i])
        if final_return is not None and abs(final_return) >= self.min_abs_reward:
            for i in range(max(0, len(tokens) - self.credit_window), len(tokens)):
                self._credit(tokens, i, self.credit if final_return > 0 else -self.credit,
                             sigs[i])

    def _credit(self, tokens, i, label, b) -> None:
        for n in range(self.min_len, self.max_len + 1):
            lo = i - n + 1
            if lo < 0:
                continue
            e = self.ev.setdefault((b, tuple(tokens[lo:i + 1])), _Evidence())
            e.pos = e.pos * self.decay + max(label, 0.0)
            e.neg = e.neg * self.decay + max(-label, 0.0)

    def _regress(self, tokens, rewards, vocab, idxs, bucket) -> None:
        n_rows, V = len(idxs), len(vocab)
        X = np.zeros((n_rows, V)); y = np.empty(n_rows)
        for r, t in enumerate(idxs):
            y[r] = rewards[t]
            for j, pat in enumerate(vocab):
                k = len(pat)
                if t + 1 >= k and tuple(tokens[t + 1 - k:t + 1]) == pat:
                    X[r, j] = 1.0
        y -= y.mean()
        A = X.T @ X + self.ridge_lambda * np.eye(V)
        try:
            beta = np.linalg.solve(A, X.T @ y)
            Ai = np.linalg.inv(A)
        except np.linalg.LinAlgError:
            return
        resid = y - X @ beta
        dof = max(1.0, n_rows - float(np.trace(X @ Ai @ X.T)))
        sigma2 = float(resid @ resid) / dof
        se = np.sqrt(np.clip(np.diag(sigma2 * Ai), 0.0, None))
        for j, pat in enumerate(vocab):
            if se[j] <= 1e-9:
                continue
            tstat = float(beta[j] / se[j])
            self.last_reg.setdefault(pat, (float(beta[j]), tstat))
            if abs(tstat) <= 1.0:
                continue
            cred = min(abs(tstat), self.t_cap)
            e = self.ev.setdefault((bucket, pat), _Evidence())
            if tstat > 0:
                e.pos = e.pos * self.decay + cred
            else:
                e.neg = e.neg * self.decay + cred

    def generalize(self, min_ev=2.0) -> List[Tuple[Any, tuple, tuple, tuple]]:
        merged = []
        while True:
            by_b: Dict[Any, List[tuple]] = {}
            for (b, pat) in self.ev:
                by_b.setdefault(b, []).append(pat)
            applied = False
            for b, pats in by_b.items():
                for i in range(len(pats)):
                    a = pats[i]
                    if (b, a) not in self.ev:
                        continue
                    for j in range(i + 1, len(pats)):
                        c = pats[j]
                        if (b, c) not in self.ev:
                            continue
                        if len(a) != len(c):
                            continue
                        diff = [k for k in range(len(a)) if a[k] != c[k]]
                        if len(diff) != 1:
                            continue
                        k = diff[0]
                        if a[k] is None or c[k] is None:
                            continue
                        ea, ec = self.ev[(b, a)], self.ev[(b, c)]
                        if ea.pos + ea.neg < min_ev or ec.pos + ec.neg < min_ev:
                            continue
                        if (ea.pos - ea.neg) * (ec.pos - ec.neg) <= 0:
                            continue
                        w = a[:k] + (None,) + a[k + 1:]
                        self.ev[(b, w)] = _Evidence(ea.pos + ec.pos, ea.neg + ec.neg)
                        del self.ev[(b, a)]; del self.ev[(b, c)]
                        merged.append((b, a, c, w))
                        applied = True
                        break
                    if applied:
                        break
                if applied:
                    break
            if not applied:
                break
        return merged

    def _global_ev(self) -> Dict[tuple, _Evidence]:
        g: Dict[tuple, _Evidence] = {}
        for (_, pat), e in self.ev.items():
            ge = g.setdefault(pat, _Evidence())
            ge.pos += e.pos; ge.neg += e.neg
        return g

    def _effective(self, b, pat, e, g) -> _Evidence:
        pos, neg, wsum = e.pos, e.neg, 1.0
        if self.kernel_sigma > 0 and b != ACPool.GLOBAL:
            for (bb, p2), e2 in self.ev.items():
                if p2 != pat or bb == b or bb == ACPool.GLOBAL:
                    continue
                d = _bucket_dist(b, bb)
                if d is None:
                    continue
                w = math.exp(-((d / self.kernel_sigma) ** 2))
                if w > 1e-6:
                    pos += w * e2.pos; neg += w * e2.neg; wsum += w
            pos /= wsum; neg /= wsum
        ge = g.get(pat)
        if ge is not None and b != ACPool.GLOBAL:
            pos += self.global_share * ge.pos
            neg += self.global_share * ge.neg
        return _Evidence(pos, neg)

    def port_from(self, other: "PatternStore", token_map: Optional[Dict[Any, Any]] = None,
                  bucket_map: Optional[Dict[Any, Any]] = None, scale: float = 0.5) -> int:
        token_map = token_map or {}
        bucket_map = bucket_map or {}
        n = 0
        for (b, pat), e in other.ev.items():
            if any(t not in token_map for t in pat):
                continue
            mpat = tuple(token_map[t] for t in pat)
            bb = bucket_map.get(b, b)
            tgt = self.ev.setdefault((bb, mpat), _Evidence())
            tgt.pos += scale * e.pos
            tgt.neg += scale * e.neg
            n += 1
        return n

    # ---- ★ multi-source port with disagreement weighting ----
    def port_multi(self, sources, base_scale=0.5, unanimous_boost=0.3,
                   disagree_pen=0.4) -> Tuple[int, List[tuple]]:
        """sources: list of (PatternStore, token_map|None, bucket_map|None).
           Per mapped (bucket, pattern): majority sign vote; agreeing sources
           keep full weight, dissenting ones scaled by (1-disagree_pen);
           unanimous (>=2 sources) patterns boosted. Returns (n, report)."""
        per: Dict[Tuple[Any, tuple], Dict[int, _Evidence]] = defaultdict(dict)
        for si, (st, tm, bm) in enumerate(sources):
            tm = tm or {}
            bm = bm or {}
            for (b, pat), e in st.ev.items():
                if tm and any(t not in tm for t in pat):
                    continue
                mp_ = tuple(tm[t] for t in pat) if tm else pat
                bb = bm.get(b, b)
                per[(bb, mp_)][si] = _Evidence(e.pos, e.neg)
        report, n = [], 0
        for (bb, mp_), persrc in sorted(per.items(), key=str):
            signs = {si: (1 if v.pos > v.neg else -1 if v.neg > v.pos else 0)
                     for si, v in persrc.items()}
            vals = [s for s in signs.values() if s != 0]
            if not vals:
                continue
            tot = sum(vals)
            majority = 1 if tot > 0 else -1 if tot < 0 else 0
            agree = sum(1 for s in vals if s == majority)
            disagree = sum(1 for s in vals if s not in (0, majority))
            boost = (1 + unanimous_boost) if (disagree == 0 and agree >= 2) else 1.0
            for si, e in persrc.items():
                rel = 1.0 if signs[si] in (0, majority) else (1.0 - disagree_pen)
                tgt = self.ev.setdefault((bb, mp_), _Evidence())
                tgt.pos += base_scale * boost * rel * e.pos
                tgt.neg += base_scale * boost * rel * e.neg
            n += 1
            report.append((bb, self.disp(mp_), len(persrc), agree, disagree, boost))
        return n, report

    def posterior(self, bucket, pat):
        e = self.ev.get((bucket, tuple(pat)))
        if e is None or (e.pos + e.neg) <= 0:
            return None
        n = e.pos + e.neg
        p = (e.pos + 1.0) / (n + 2.0)
        sd = math.sqrt(max(p * (1 - p), 1e-12) / (n + 3.0))
        return n, p, sd

    def thompson(self, bucket, pat, rng):
        ps = self.posterior(bucket, pat)
        if ps is None:
            return None
        e = self.ev[(bucket, tuple(pat))]
        return rng.betavariate(e.pos + 1.0, e.neg + 1.0)

    def ci(self, bucket, pat, z=1.96):
        ps = self.posterior(bucket, pat)
        if ps is None:
            return None
        n, p, _ = ps
        h = z * math.sqrt(max(p * (1 - p), 1e-12) / (n + 2.0))
        return (max(0.0, p - h), min(1.0, p + h))

    def _classify(self, pos, neg) -> Tuple[float, Kind, float]:
        n = pos + neg
        if n <= 1e-9:
            return 0.0, Kind.NEUTRAL, 0.0
        p = (pos + 1.0) / (n + 2.0)
        sd = math.sqrt(max(p * (1 - p), 1e-12) / (n + 3.0))
        conf = abs(2 * p - 1) * min(1.0, n / self.min_obs)
        if n >= self.min_obs:
            if p - self.beta_z * sd >= self.dominance:
                return self.gain * conf, Kind.SUCCESS, conf
            if p + self.beta_z * sd <= 1.0 - self.dominance:
                return -self.gain * conf, Kind.FAILURE, conf
        return (self.gain if p >= 0.5 else -self.gain) * 0.5 * conf, Kind.NEUTRAL, conf

    def scrub(self, pat: tuple) -> int:
        keys = [k for k in self.ev if k[1] == tuple(pat)]
        for k in keys:
            del self.ev[k]
        return len(keys)

    def halve(self, bucket=None) -> None:
        for (b, _), e in self.ev.items():
            if bucket is None or b == bucket:
                e.pos *= 0.5; e.neg *= 0.5

    _PARAMS = ("min_len", "max_len", "focus", "min_abs_reward", "min_obs", "dominance",
               "conf_min_pat", "gain", "decay", "w_threshold", "credit_window",
               "credit", "global_share", "beta_z", "use_regression",
               "regress_min_steps", "regress_min_rows", "ridge_lambda", "t_cap",
               "kernel_sigma")

    def save(self, path: str) -> None:
        data = {"format": STORE_FORMAT, "version": FORMAT_VERSION,
                "params": {k: getattr(self, k) for k in self._PARAMS},
                "evidence": [{"b": _enc(b),
                              "p": [None if t is None else t for t in pat],
                              "pos": e.pos, "neg": e.neg}
                             for (b, pat), e in self.ev.items()]}
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, path: str) -> "PatternStore":
        with open(path) as f:
            data = json.load(f)
        if data.get("format") != STORE_FORMAT:
            raise ValueError(f"not an {STORE_FORMAT} file: {path}")
        store = cls(**data.get("params", {}))
        for d in data["evidence"]:
            store.ev[(_dec(d["b"]), tuple(d["p"]))] = _Evidence(d["pos"], d["neg"])
        return store


def _expand_wildcard(pat, sigma) -> List[tuple]:
    slots = [i for i, t in enumerate(pat) if t is None]
    if not slots:
        return [pat]
    outs: List[tuple] = []
    def rec(i, cur):
        if i == len(pat):
            outs.append(tuple(cur)); return
        if pat[i] is None:
            for t in sigma:
                cur.append(t); rec(i + 1, cur); cur.pop()
        else:
            cur.append(pat[i]); rec(i + 1, cur)
    rec(0, [])
    return outs


class ACPool:
    GLOBAL = "G"

    def __init__(self, store: PatternStore, min_bucket_ev: float = 6.0):
        self.store, self.min_bucket_ev = store, min_bucket_ev
        self.acs: Dict[Any, AhoCorasick] = {}
        self._last_key: Optional[frozenset] = None

    def get(self, b) -> AhoCorasick:
        return self.acs.get(b) or self.acs[self.GLOBAL]

    def rebuild(self) -> bool:
        store, g = self.store, self.store._global_ev()
        store.typed = {}
        sigma = store.alphabet()
        ac_g = AhoCorasick()
        for pat, e in g.items():
            w, k, c = store._classify(e.pos, e.neg)
            if abs(w) >= store.w_threshold:
                store.typed[(self.GLOBAL, pat)] = (w, k, c)
                for inst in _expand_wildcard(pat, sigma):
                    ac_g.add_pattern(inst, w, k, c, source=pat if pat != inst else None)
        ac_g.build()
        newacs = {self.GLOBAL: ac_g}
        for b in store.buckets():
            if b == self.GLOBAL:
                continue
            ac = AhoCorasick(); total = 0.0
            for (bb, pat), e in store.ev.items():
                if bb != b:
                    continue
                total += e.pos + e.neg
                eff = store._effective(b, pat, e, g)
                w, k, c = store._classify(eff.pos, eff.neg)
                if abs(w) >= store.w_threshold:
                    store.typed[(b, pat)] = (w, k, c)
                    for inst in _expand_wildcard(pat, sigma):
                        ac.add_pattern(inst, w, k, c,
                                       source=pat if pat != inst else None)
            if total >= self.min_bucket_ev:
                ac.build()
                newacs[b] = ac
        key = frozenset((b, p, round(w, 3), k.name, round(c, 3))
                        for (b, p), (w, k, c) in store.typed.items())
        changed = key != self._last_key
        self._last_key = key
        self.acs = newacs
        return changed


def rebuild_obs(store_obs: PatternStore, ac_obs: AhoCorasick) -> frozenset:
    ac_obs.reset()
    sigma = store_obs.alphabet()
    for (_, pat), e in store_obs.ev.items():
        w, k, c = store_obs._classify(e.pos, e.neg)
        if abs(w) >= store_obs.w_threshold:
            for inst in _expand_wildcard(pat, sigma):
                ac_obs.add_pattern(inst, w, k, c,
                                   source=pat if pat != inst else None)
    ac_obs.build()
    return _fp(ac_obs.pattern_list())


# ---------------------------------------------------------------------------
# 3. Environments: ComboGrid, TemplateGenEnv, ★GrammarGenEnv
#    (RenamedGrid = ComboGrid subclass for the alignment demo)
# ---------------------------------------------------------------------------
class ComboGrid:
    MOVES = {'U': (-1, 0), 'D': (1, 0), 'L': (0, -1), 'R': (0, 1)}

    def __init__(self, n=6,
                 combos=((('R', 'R', 'R'), 5.0, None), (('D', 'D', 'D'), -5.0, None)),
                 step_cost=-0.05, goal_reward=10.0, ctx=3,
                 hazards=(), hazard_cost=-2.0, bucketed=False,
                 noisy: Optional[Dict[tuple, float]] = None, seed=0):
        self.n, self.ctx = n, ctx
        self.step_cost, self.goal_reward = step_cost, goal_reward
        self.combos, self.hazards, self.hazard_cost = combos, set(hazards), hazard_cost
        self.bucketed = bucketed
        self.noisy = noisy or {}
        self.rng = random.Random(seed)
        self.signs = [1.0] * len(combos)
        self.start, self.goal = (0, 0), (n - 1, n - 1)

    def set_sign(self, idx, s):
        self.signs[idx] = s

    def reset(self):
        return self.start, ()

    def actions(self, s):
        return list(self.MOVES)

    def is_terminal(self, s):
        return s[0] == self.goal

    def signature(self, s):
        if not self.bucketed:
            return ACPool.GLOBAL
        (i, j), _ = s
        return (i * 2 // self.n, j * 2 // self.n)

    def observe(self, s):
        (i, j), _ = s
        if s in self.hazards:
            return 'H'
        gi, gj = self.goal
        return 'G' if abs(i - gi) + abs(j - gj) <= 2 else '.'

    def step(self, s, a):
        (i, j), ctx = s
        di, dj = self.MOVES[a]
        i2 = min(max(i + di, 0), self.n - 1)
        j2 = min(max(j + dj, 0), self.n - 1)
        moved = (i2, j2) != (i, j)
        ctx2 = tuple((ctx + (a,))[-self.ctx:]) if moved else ctx
        r = self.step_cost
        if (i2, j2) in self.hazards:
            r += self.hazard_cost
        for idx, (pat, w, region) in enumerate(self.combos):
            k = len(pat)
            if len(ctx2) >= k and ctx2[-k:] == pat:
                if region is None or (region[0] <= i2 <= region[1]
                                      and region[2] <= j2 <= region[3]):
                    pay = w * self.signs[idx]
                    sg = self.noisy.get(pat)
                    if sg:
                        pay += self.rng.gauss(0.0, sg)
                    r += pay
        ns = ((i2, j2), ctx2)
        done = self.is_terminal(ns)
        if done:
            r += self.goal_reward
        return ns, r, done


class RenamedGrid(ComboGrid):
    """Same MDP as ComboGrid, compass-token vocabulary — the alignment demo's
       target env."""
    MOVES = {'N': (-1, 0), 'S': (1, 0), 'W': (0, -1), 'E': (0, 1)}


GEN_CFG = {
    'log': dict(
        vocab=['ERROR', 'WARN', 'INFO', 'disk', 'user', 'net', 'full', 'login',
               'down', 'END'],
        families=(('SYS', (('ERROR', 'disk', 'full'), ('WARN', 'disk', 'down'))),
                  ('USR', (('INFO', 'user', 'login'), ('WARN', 'user', 'down'))),
                  ('NET', (('WARN', 'net', 'down'), ('INFO', 'net', 'full')))),
        switch_every=3, max_len=9, fam_reward=5.0, end_bonus=2.0, step_cost=-0.05),
}


class TemplateGenEnv:
    stop_token = 'END'
    CTX_CAP = 5

    def __init__(self, cfg: dict, seed=0):
        self.cfg = cfg
        self.vocab = list(cfg['vocab'])
        self.fams = [(name, [tuple(t) for t in temps]) for name, temps in cfg['families']]
        self.rng = random.Random(seed)

    @property
    def max_len(self):
        return self.cfg['max_len']

    def reset(self):
        return (0, (), 0, 0)

    def actions(self, s):
        return list(self.vocab)

    def is_terminal(self, s):
        return s[3] >= self.cfg['max_len']

    def signature(self, s):
        return ACPool.GLOBAL

    def observe(self, s):
        return str(s[0])

    def step(self, s, a):
        fam, ctx, pays, t = s
        if a == 'END':
            r = self.cfg['end_bonus'] if pays > 0 else 0.0
            return (fam, ctx, pays, self.cfg['max_len']), r, True
        ctx2 = tuple((ctx + (a,))[-self.CTX_CAP:])
        r = self.cfg['step_cost']
        pays2 = pays
        for tp in self.fams[fam][1]:
            k = len(tp)
            if len(ctx2) >= k and ctx2[-k:] == tp:
                r += self.cfg['fam_reward']
                pays2 += 1
        t2 = t + 1
        nfam = fam
        if t2 % self.cfg['switch_every'] == 0 and t2 < self.cfg['max_len']:
            nfam = (fam + 1) % len(self.fams)
        return (nfam, ctx2, pays2, t2), r, self.is_terminal((nfam, ctx2, pays2, t2))


# ---- ★ GrammarGenEnv: compositional S -> NP VP, VP -> verb (NP)? ----
class GrammarGenEnv:
    """Pushdown-style generation. Episode length & reward are grammar-set
       (safety cap is never binding). signature = top-of-stack slot, so the
       bucketed automatons learn syntactically-conditioned patterns."""
    stop_token = None
    EXP = {'S': ('NP', 'VP'), 'NP': ('det', 'adj', 'noun'), 'VP': ('verb', 'VOPT')}
    SLOTS = {'det': ('the', 'a'), 'adj': ('quick', 'calm'),
             'noun': ('fox', 'sea'), 'verb': ('runs', 'rests')}
    VOCAB = ['the', 'a', 'quick', 'calm', 'fox', 'sea', 'runs', 'rests', 'OBJ', 'END']

    def __init__(self, seed=0, slot_pay=0.5, obj_bonus=7.0, short_bonus=4.0,
                 vp_close=2.0, wrong_pen=-0.3, hard_cap=24):
        self.max_len = hard_cap
        self.slot_pay, self.obj_bonus = slot_pay, obj_bonus
        self.short_bonus, self.vp_close, self.wrong_pen = short_bonus, vp_close, wrong_pen
        self.rng = random.Random(seed)

    def _expand(self, stack) -> tuple:
        stack = list(stack)
        changed = True
        while changed:
            changed = False
            for i, x in enumerate(stack):
                if x in self.EXP:
                    stack[i:i + 1] = list(reversed(self.EXP[x]))   # top-last
                    changed = True
                    break
        return tuple(stack)

    def reset(self):
        return (self._expand(('S',)), (), False, 0)

    def actions(self, s):
        top = s[0][-1] if s[0] else ''
        if top == 'VOPT':
            return ['OBJ', 'END']
        return list(self.SLOTS.get(top, ()))

    def is_terminal(self, s):
        return not s[0] or s[3] >= self.max_len

    def signature(self, s):
        return s[0][-1] if s[0] else 'DONE'

    def observe(self, s):
        return self.signature(s)

    def step(self, s, a):
        stack, words, has_obj, t = s
        if not stack:
            return s, 0.0, True
        top = stack[-1]
        if top == 'VOPT':
            if a == 'OBJ':
                ns = (self._expand(('NP',)) + stack[:-1], words + ('OBJ',), True, t + 1)
                return ns, 0.0, self.is_terminal(ns)
            if a == 'END':
                ns = (stack[:-1], words + ('END',), has_obj, t + 1)
                return ns, self.vp_close, self.is_terminal(ns)
            return s, self.wrong_pen, False
        ok = self.SLOTS.get(top, ())
        if a in ok:
            ns = (stack[:-1], words + (a,), has_obj, t + 1)
            r = self.slot_pay
            done = self.is_terminal(ns)
            if done:
                r += self.obj_bonus if has_obj else self.short_bonus
            return ns, r, done
        return s, self.wrong_pen, False


def bootstrap_corpus(store: PatternStore, families, bonus=1.2) -> int:
    n = 0
    for _, temps in families:
        for tp in temps:
            for ln in range(store.min_len, store.max_len + 1):
                for i in range(len(tp) - ln + 1):
                    store.ev.setdefault(('G', tuple(tp[i:i + ln])),
                                        _Evidence()).pos += bonus
                    n += 1
    return n


# ---------------------------------------------------------------------------
# 4. Verifier (CVaR options) + cvar_tail — unchanged from v8
# ---------------------------------------------------------------------------
def cvar_tail(diffs, alpha=0.25) -> float:
    s = sorted(diffs)
    k = max(1, math.ceil(alpha * len(s)))
    return sum(s[:k]) / k


@dataclass
class Verdict:
    ok: Optional[bool]
    diffs: List[float]
    sigs: List[Any]
    mean: float = 0.0
    p25: float = 0.0
    cvar: float = 0.0
    alpha: float = 0.25


class Verifier:
    HOT_CAP, COLD_CAP = 12, 4

    def __init__(self, env, k=8, depth=8, cvar_alpha=0.25, seed=0):
        self.env, self.k, self.depth = env, k, depth
        self.cvar_alpha = cvar_alpha
        self.rng = random.Random(seed)
        self.hot: Dict[Any, List[Tuple[float, Any]]] = defaultdict(list)
        self.cold: Dict[Any, List[Tuple[float, Any]]] = defaultdict(list)
        self.records: Dict[Tuple[tuple, Any], Verdict] = {}
        self.cache: Dict[Tuple[tuple, Any, int], Verdict] = {}
        self.cache_opt: Dict[Tuple[tuple, Any], Verdict] = {}
        self.attempts: Dict[Tuple[tuple, Any, int], int] = defaultdict(int)

    def remember(self, s, hotness: float = 0.0) -> None:
        if self.env.is_terminal(s):
            return
        b = self.env.signature(s)
        pool = self.hot if hotness > 0 else self.cold
        lst = pool[b]
        entry = (hotness, s)
        cap = self.HOT_CAP if pool is self.hot else self.COLD_CAP
        if len(lst) < cap:
            lst.append(entry)
            lst.sort(key=lambda x: -x[0])
        elif hotness > lst[-1][0]:
            lst[-1] = entry
            lst.sort(key=lambda x: -x[0])
        elif self.rng.random() < 0.1:
            lst[self.rng.randrange(len(lst))] = entry

    def _candidates(self, bucket):
        cands = [s for (_, s) in self.hot.get(bucket, [])] + \
                [s for (_, s) in self.cold.get(bucket, [])]
        cands = [s for s in cands if not self.env.is_terminal(s)]
        self.rng.shuffle(cands)
        return cands

    def _paired(self, tokens, bucket, k):
        cands = self._candidates(bucket)
        if len(cands) < 3:
            return None, None
        diffs, sigs = [], []
        for s in cands[:k]:
            rp = self._run(s, list(tokens))
            rr = self._run(s, [self.rng.choice(self.env.actions(s)) for _ in tokens])
            diffs.append(rp - rr)
            sigs.append(self.env.signature(s))
        return diffs, sigs

    def test(self, tokens, bucket, expect: int) -> Optional[Verdict]:
        key = (tuple(tokens), bucket, expect)
        if key in self.cache:
            return self.cache[key]
        if self.attempts[key] >= 2:
            return None
        self.attempts[key] += 1
        diffs, sigs = self._paired(tokens, bucket, self.k)
        if diffs is None:
            return None
        good = sum(1 for d in diffs if (d > 1e-9 if expect > 0 else d < -1e-9))
        v = Verdict(ok=good >= 0.7 * len(diffs), diffs=diffs, sigs=sigs,
                    mean=sum(diffs) / len(diffs))
        self.cache[key] = v
        self.records[(tuple(tokens), bucket)] = v
        return v

    def test_option(self, tokens, bucket, k=10) -> Optional[Verdict]:
        key = (tuple(tokens), bucket)
        if key in self.cache_opt:
            return self.cache_opt[key]
        diffs, sigs = self._paired(tokens, bucket, k)
        if diffs is None:
            return None
        mean = sum(diffs) / len(diffs)
        v = Verdict(ok=(mean > 0 and cvar_tail(diffs, self.cvar_alpha) > 0),
                    diffs=diffs, sigs=sigs, mean=mean,
                    p25=sorted(diffs)[max(0, int(math.ceil(0.25 * len(diffs))) - 1)],
                    cvar=cvar_tail(diffs, self.cvar_alpha), alpha=self.cvar_alpha)
        self.cache_opt[key] = v
        self.records[(tuple(tokens), bucket)] = v
        return v

    def clear_bucket(self, bucket) -> None:
        self.cache.clear(); self.cache_opt.clear()
        for kk in [k for k in self.records if k[1] == bucket]:
            del self.records[kk]

    def _run(self, s, prefix) -> float:
        G, seq = 0.0, list(prefix)
        for t in range(self.depth):
            a = seq[t] if t < len(seq) else self.rng.choice(self.env.actions(s))
            s, r, done = self.env.step(s, a)
            G += r
            if done:
                break
        return G

    def save(self, path: str) -> None:
        data = {"format": VERIFY_FORMAT, "version": FORMAT_VERSION,
                "records": [[list(p), _enc(b), v.ok, [round(d, 4) for d in v.diffs],
                             [_enc(x) for x in v.sigs], round(v.mean, 4),
                             round(v.p25, 4), round(v.cvar, 4), v.alpha]
                            for (p, b), v in self.records.items()]}
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    def load_records(self, path: str) -> int:
        with open(path) as f:
            data = json.load(f)
        if data.get("format") != VERIFY_FORMAT:
            raise ValueError(f"not an {VERIFY_FORMAT} file: {path}")
        for p, b, ok, diffs, sigs, mean, p25, cvar, alpha in data["records"]:
            self.records[(tuple(p), _dec(b))] = Verdict(ok, diffs,
                                                        [_dec(x) for x in sigs],
                                                        mean, p25, cvar, alpha)
        return len(self.records)


# ---------------------------------------------------------------------------
# 5. Monitors: Drift, GridBandit, ParamBandits, ★RiskBudget
# ---------------------------------------------------------------------------
class DriftMonitor:
    def __init__(self, af=0.4, as_=0.05, warmup=5):
        self.af, self.as_, self.warmup = af, as_, warmup
        self.fast: Dict[Any, float] = {}
        self.slow: Dict[Any, float] = {}
        self.n: Dict[Any, int] = {}

    def update(self, b, x) -> bool:
        self.n[b] = self.n.get(b, 0) + 1
        f, sl = self.fast.get(b), self.slow.get(b)
        self.fast[b] = x if f is None else (1 - self.af) * f + self.af * x
        self.slow[b] = x if sl is None else (1 - self.as_) * sl + self.as_ * x
        if self.n[b] <= self.warmup:
            return False
        if self.fast[b] < self.slow[b] - max(1.0, 0.4 * abs(self.slow[b])):
            self.fast[b] = self.slow[b]
            return True
        return False


class GridBandit:
    DEFAULT_STORE = [{},
                     {"min_obs": 5.0, "dominance": 0.75, "beta_z": 1.5},
                     {"min_obs": 3.0, "dominance": 0.65, "beta_z": 0.7}]

    def __init__(self, c_arms=(1.0, 2.0), store_arms=None, commit=3, eps=0.15, seed=0):
        store_arms = store_arms or self.DEFAULT_STORE
        self.cfgs = [(c, dict(s)) for c in c_arms for s in store_arms]
        self.commit, self.eps = commit, eps
        self.n = [0.0] * len(self.cfgs)
        self.m = [0.0] * len(self.cfgs)
        self.cur: Optional[int] = None
        self.pend: List[float] = []
        self.rng = random.Random(seed)

    def start_episode(self):
        if self.cur is None or len(self.pend) >= self.commit:
            if self.cur is not None and self.pend:
                r = sum(self.pend) / len(self.pend)
                self.n[self.cur] += 1
                self.m[self.cur] += (r - self.m[self.cur]) / self.n[self.cur]
            if self.rng.random() < self.eps or min(self.n) == 0:
                self.cur = self.rng.randrange(len(self.cfgs))
            else:
                tot = sum(self.n)
                self.cur = max(range(len(self.cfgs)),
                               key=lambda j: self.m[j]
                               + math.sqrt(2 * math.log(tot + 1) / max(self.n[j], 1e-9)))
            self.pend = []
        return self.cfgs[self.cur]

    def observe(self, r: float) -> None:
        self.pend.append(r)

    def apply(self, planner, store) -> str:
        c, s = self.cfgs[self.cur]
        planner.c_puct = c
        for k, v in s.items():
            setattr(store, k, v)
        return f"c={c:g}/" + (",".join(f"{k[:3]}={v:g}" for k, v in s.items()) or "mid")

    def summary(self) -> str:
        return " | ".join(f"[{c:g}] n={int(nn)} Q={mm:+.2f}"
                          for (c, _), nn, mm in zip(self.cfgs, self.n, self.m))


class ParamBandits:
    def __init__(self, eps_arms=(0.05, 0.15, 0.30), temp_arms=(0.35, 0.5, 0.8),
                 eps=0.2, seed=0):
        self.ea, self.ta, self.eps = list(eps_arms), list(temp_arms), eps
        self.n: Dict[tuple, float] = {}
        self.m: Dict[tuple, float] = {}
        self.rng = random.Random(seed)

    def _pick(self, b, arms, kind):
        seen = [self.n.get((b, kind, i), 0.0) for i in range(len(arms))]
        if self.rng.random() < self.eps or min(seen) == 0:
            return self.rng.randrange(len(arms))
        tot = sum(seen) + 1e-9
        return max(range(len(arms)),
                   key=lambda j: self.m.get((b, kind, j), 0.0)
                   + math.sqrt(2 * math.log(tot) / max(seen[j], 1e-9)))

    def pick(self, b):
        return self._pick(b, self.ea, 'e'), self._pick(b, self.ta, 't')

    def update(self, b, ie, it, r) -> None:
        for kind, i in (('e', ie), ('t', it)):
            k = (b, kind, i)
            self.n[k] = self.n.get(k, 0.0) + 1
            self.m[k] = self.m.get(k, 0.0) + (r - self.m[k]) / self.n[k]


class RiskBudget:
    """★ Episode-level per-bucket CVaR. Breach: CVaR_alpha(bucket episode
       contributions) < budget. Effects while breached (per bucket):
       rollout-leaf risk_beta escalated, options suspended. Hysteresis
       margin prevents flapping at the threshold."""

    def __init__(self, alpha=0.25, budget=-1.5, buffer=10, hysteresis=0.5,
                 beta_max=0.8, beta_step=0.2):
        self.alpha, self.budget, self.buffer = alpha, budget, buffer
        self.hyst, self.beta_max, self.beta_step = hysteresis, beta_max, beta_step
        self.buf: Dict[Any, deque] = defaultdict(lambda: deque(maxlen=buffer))
        self.breached: Dict[Any, bool] = {}

    def cvar(self, b) -> Optional[float]:
        vals = list(self.buf.get(b, []))
        return cvar_tail(vals, self.alpha) if len(vals) >= 4 else None

    def update(self, b, x) -> Tuple[bool, bool]:
        """returns (newly_breached, recovered)"""
        self.buf[b].append(x)
        cv = self.cvar(b)
        was = self.breached.get(b, False)
        now = was
        if cv is not None:
            now = cv < (self.budget + (self.hyst if was else 0.0))
        self.breached[b] = now
        return (now and not was), (was and not now)

    def beta_for(self, b, base) -> float:
        if not self.breached.get(b, False):
            return base
        cv = self.cvar(b)
        if cv is None:
            return base
        depth = max(0.0, self.budget - cv)
        unit = max(abs(self.budget) * 0.5, 1e-9)
        steps = min(3, int(depth / unit) + 1)
        return min(self.beta_max, base + self.beta_step * steps)

    def options_ok(self, b) -> bool:
        return not self.breached.get(b, False)

    def save(self) -> dict:
        return {"buffers": [[_enc(b), list(v)] for b, v in self.buf.items()],
                "breached": [[_enc(b), x] for b, x in self.breached.items()]}

    def load(self, d) -> None:
        self.buf = defaultdict(lambda: deque(maxlen=self.buffer),
                               {_dec(b): deque(v, maxlen=self.buffer)
                                for b, v in d["buffers"]})
        self.breached = {_dec(b): x for b, x in d["breached"]}


# ---------------------------------------------------------------------------
# 6. ★ VocabAligner: statistic-proposed, human-curated explicit maps
# ---------------------------------------------------------------------------
class VocabAligner:
    """Two-pass alignment from episode logs:
         pass 1 seeds candidates by reward-profile closeness;
         pass 2 refines by bigram-context overlap, mapping neighbor tokens
         through the current map (resolves the vocabulary chicken-and-egg).
       Output is a PROPOSAL REPORT for human curation — no embeddings, no
       hidden representation: every pair ships with score + evidence."""

    def __init__(self, min_score=0.45, w_reward=0.6, w_ctx=0.4, rew_scale=2.0):
        self.min_score, self.w_reward, self.w_ctx, self.rew_scale = \
            min_score, w_reward, w_ctx, rew_scale

    def _profiles(self, episodes):
        prof: Dict[Any, dict] = {}
        for tokens, rewards in episodes:
            for i, t in enumerate(tokens):
                p = prof.setdefault(t, {'ctx': defaultdict(float), 'r': 0.0, 'n': 0})
                prev = tokens[i - 1] if i > 0 else '∘'
                nxt = tokens[i + 1] if i + 1 < len(tokens) else '∘'
                p['ctx'][(prev, nxt)] += 1.0
                p['r'] += rewards[i]; p['n'] += 1
        for p in prof.values():
            p['r'] /= max(p['n'], 1)
        return prof

    def _ctx_mapped(self, prof, token, mapping):
        acc: Dict[tuple, float] = {}
        for (p, n), c in prof[token]['ctx'].items():
            k = (mapping.get(p, p), mapping.get(n, n))
            acc[k] = acc.get(k, 0.0) + c
        return acc

    def propose(self, src_eps, tgt_eps) -> Tuple[Dict[Any, Any], List[tuple]]:
        sp, tp = self._profiles(src_eps), self._profiles(tgt_eps)
        # pass 1 — reward seeds, greedy 1-1
        pairs = sorted(((math.exp(-abs(sv['r'] - tv['r']) / self.rew_scale), s, t)
                        for s, sv in sp.items() for t, tv in tp.items()), reverse=True)
        seed: Dict[Any, Any] = {}
        us, ut = set(), set()
        for sc, s, t in pairs:
            if s in us or t in ut:
                continue
            us.add(s); ut.add(t); seed[s] = t
        # pass 2 — context overlap under the current map
        inv = {v: k for k, v in seed.items()}
        report, out = [], {}
        for s, t in seed.items():
            cs = self._ctx_mapped(sp, s, seed)
            ct = self._ctx_mapped(tp, t, inv)
            keys = set(cs) | set(ct)
            inter = sum(min(cs.get(k, 0.0), ct.get(k, 0.0)) for k in keys)
            union = sum(cs.values()) + sum(ct.values()) - inter
            ctxsim = inter / union if union > 0 else 0.0
            rew = math.exp(-abs(sp[s]['r'] - tp[t]['r']) / self.rew_scale)
            score = self.w_reward * rew + self.w_ctx * ctxsim
            report.append((s, t, score, rew, ctxsim))
            if score >= self.min_score:
                out[s] = t
        return out, sorted(report, key=lambda x: -x[2])


# ---------------------------------------------------------------------------
# 7. Planner (risk hooks; VI priors; otherwise v8)
# ---------------------------------------------------------------------------
@dataclass
class _Node:
    state: Any
    ac_key: Any
    q: int
    ctx: tuple
    q_obs: int = 0
    q_opp: int = 0
    to_move: int = 0
    action: Any = None
    prior: float = 1.0
    kind: Kind = Kind.NEUTRAL
    conf: float = 0.0
    is_option: bool = False
    children: Dict[Any, Tuple[float, float, "_Node"]] = field(default_factory=dict)
    untried: Optional[List[Any]] = None
    blocked: List[str] = field(default_factory=list)
    priors: Optional[Dict[Any, float]] = None
    N: int = 0
    W: float = 0.0


class ACMCTS:
    def __init__(self, env, pool: ACPool, ac_obs: Optional[AhoCorasick] = None,
                 n_simulations=120, rollout_depth=25, gamma=0.99,
                 c_puct=1.5, prior_temp=0.5,
                 look_gamma=0.9, look_kappa=1.0,
                 shaping=0.5, succ_bonus=3.0, fail_pen=3.0,
                 prox_s=0.6, prox_f=0.9, shield=True, max_ctx=3,
                 risk_beta=0.4, risk_min_n=12, risk_alpha=0.25,
                 root_noise=0.25, dir_alpha=1.0,
                 r_eps=0.3, r_alpha=0.2, thompson=True,
                 opp_temp=0.5, opp_pen=0.8, seed=0):
        self.env, self.pool, self.ac_obs = env, pool, ac_obs
        self.n_simulations, self.rollout_depth = n_simulations, rollout_depth
        self.gamma, self.c_puct, self.prior_temp = gamma, c_puct, prior_temp
        self.look_gamma, self.look_kappa = look_gamma, look_kappa
        self.shaping = shaping
        self.succ_bonus, self.fail_pen = succ_bonus, fail_pen
        self.prox_s, self.prox_f = prox_s, prox_f
        self.shield, self.max_ctx = shield, max_ctx
        self.risk_beta, self.risk_min_n, self.risk_alpha = risk_beta, risk_min_n, risk_alpha
        self.root_noise, self.dir_alpha = root_noise, dir_alpha
        self.r_eps, self.r_alpha, self.thompson = r_eps, r_alpha, thompson
        self.opp_temp, self.opp_pen = opp_temp, opp_pen
        self.rng = random.Random(seed)
        self.TT: Dict[Tuple[Any, Any], _Node] = {}
        self.qsamples: Dict[Tuple[Any, int], List[float]] = {}
        self.R: Dict[Tuple[Any, int, Any], float] = {}
        self.verified_fail: set = set()
        self.verified_succ: set = set()
        self.options: Dict[Any, List[Tuple[str, tuple]]] = {}
        self.grid: Optional[GridBandit] = None
        self.opp: Optional[ACPool] = None
        self.risk: Optional[RiskBudget] = None        # ★ episode-level budgets
        self.sims_last = 0
        self.last_decision = None

    def attach_opp(self, opp_pool: Optional[ACPool]) -> None:
        self.opp = opp_pool

    def new_episode(self, keep_tt=False) -> None:
        if not keep_tt:
            self.TT.clear()

    def refresh_options(self, store: PatternStore) -> None:
        self.options = {}
        for (b, pat), (w, k, c) in store.typed.items():
            if any(t is None for t in pat):
                continue
            if k is Kind.SUCCESS and (pat, b) in self.verified_succ:
                name = "OPT:" + "".join(str(x) for x in pat)
                lst = self.options.setdefault(b, [])
                if all(n != name for n, _ in lst):
                    lst.append((name, pat))

    def _option_walk(self, node, tokens):
        ac = self.pool.get(node.ac_key)
        v = node.q
        for t in tokens:
            v = ac.step(v, t)
            if ac.kind(v) is Kind.FAILURE:
                return None
        return v

    def _untried(self, node) -> List[Any]:
        acts = list(self.env.actions(node.state))
        node.blocked = []
        if self.risk is not None and not self.risk.options_ok(node.ac_key):
            return acts                                # ★ options suspended here
        for name, tokens in self.options.get(node.ac_key, []):
            if self._option_walk(node, tokens) is not None:
                acts.append(name)
            else:
                node.blocked.append(name)
        return acts

    def _admissible(self, node) -> List[Any]:
        acts = self.env.actions(node.state)
        if not self.shield:
            return acts
        ac = self.pool.get(node.ac_key)
        safe = []
        for a in acts:
            v = ac.step(node.q, a)
            bad = any((src, node.ac_key) in self.verified_fail
                      for src in ac.patterns_src_at(v) if ac.kind(v) is Kind.FAILURE)
            if not bad:
                safe.append(a)
        return safe or acts

    def _opp_dist(self, state, q_opp):
        acts = self.env.actions(state)
        ac = self.opp.get(ACPool.GLOBAL) if self.opp is not None else None
        if ac is None or ac.n_states <= 1:
            return acts, [1.0 / len(acts)] * len(acts)
        scs = [ac.score(ac.step(q_opp, t)) + self.look_kappa
               * ac.lookahead(ac.step(q_opp, t), 2, self.look_gamma) for t in acts]
        m = max(scs)
        ex = [math.exp((s - m) / self.opp_temp) for s in scs]
        z = sum(ex) or 1.0
        return acts, [e / z for e in ex]

    def _build_root(self, root_state, ac_key, q, ctx, q_obs, q_opp, to_move) -> _Node:
        ac = self.pool.get(ac_key)
        root = _Node(state=root_state, ac_key=ac_key, q=q, ctx=ctx,
                     q_obs=q_obs, q_opp=q_opp, to_move=to_move)
        root.kind, root.conf = ac.kind(q), ac.conf(q)
        root.untried = self._untried(root)
        return root

    def plan(self, root_state, ac_key, q, ctx=(), q_obs=0, q_opp=0, to_move=0,
             root: Optional[_Node] = None, budget_ms: Optional[float] = None,
             explain: bool = True, tau: float = 0.0):
        if root is not None and (root.state != root_state or root.ac_key != ac_key
                                 or root.q != q or root.ctx != ctx
                                 or root.to_move != to_move or root.q_opp != q_opp):
            root = None
        if root is None:
            root = self._build_root(root_state, ac_key, q, ctx, q_obs, q_opp, to_move)
        deadline = time.perf_counter() + budget_ms / 1000.0 if budget_ms else None
        n = 0
        while n < self.n_simulations:
            if deadline is not None and time.perf_counter() >= deadline:
                break
            self._iterate(root)
            n += 1
        self.sims_last = n
        if self.root_noise > 0 and root.children:
            self._mix_noise(root)
        action = self._choose(root, tau)
        if explain:
            self.last_decision = self._explain(root, action)
        return action, root

    def _mix_noise(self, root) -> None:
        pr = self._priors(root)
        prim = [a for a in root.children if not isinstance(a, str)]
        if len(prim) < 2:
            return
        z = sum(pr.get(a, 1e-6) for a in prim)
        if z <= 0:
            return
        noise = [self.rng.gammavariate(self.dir_alpha, 1.0) for _ in prim]
        nz = sum(noise) or 1.0
        for a, x in zip(prim, noise):
            pr[a] = (1 - self.root_noise) * pr.get(a, 0.0) / z \
                + self.root_noise * x / nz

    def _choose(self, root, tau=0.0) -> Any:
        if not root.children:
            return self.rng.choice(self._admissible(root))
        cands = [e[2] for e in root.children.values() if e[2].kind is not Kind.FAILURE] \
            or [e[2] for e in root.children.values()]
        if tau and tau > 1e-3:
            ws = [max(c.N, 0) ** (1.0 / tau) for c in cands]
            tot = sum(ws)
            if tot <= 0:
                return self.rng.choice(cands).action
            x = self.rng.random() * tot
            for c, w in zip(cands, ws):
                x -= w
                if x <= 0:
                    return c.action
            return cands[-1].action
        return max(cands, key=lambda c: c.N).action

    def _iterate(self, root: _Node) -> None:
        node, edges, on_path = root, [], {id(root)}
        while True:
            if self.env.is_terminal(node.state):
                break
            if node.untried:
                a = node.untried.pop(self.rng.randrange(len(node.untried)))
                child = self._expand(node, a)
                if id(child) not in on_path:
                    edges.append((node, a)); on_path.add(id(child))
                break
            if node.children:
                sel = self._select(node, on_path)
                if sel is None:
                    break
                a, edge = sel
                edges.append((node, a)); node = edge[2]; on_path.add(id(node))
                continue
            break
        terminal = self.env.is_terminal(node.state)
        G = 0.0 if terminal else self._rollout(node)
        if not terminal:
            rb = self.risk_beta
            if self.risk is not None:                       # ★ per-bucket escalation
                rb = self.risk.beta_for(node.ac_key, self.risk_beta)
            if rb > 0:
                key = (node.ac_key, node.q)
                samp = self.qsamples.setdefault(key, [])
                if len(samp) >= self.risk_min_n:
                    srt = sorted(samp)
                    kk = max(1, int(self.risk_alpha * len(srt)))
                    G = (1 - rb) * G + rb * sum(srt[:kk]) / kk
                samp.append(G)
                if len(samp) > 64:
                    samp.pop(0)
        self._backup(edges, node, G)

    def _select(self, node, on_path):
        sqrt_n = math.sqrt(max(node.N, 1))
        best_edge, best_val = None, -math.inf
        for a, e in node.children.items():
            ch = e[2]
            if id(ch) in on_path:
                continue
            val = ch.W / ch.N + self.c_puct * ch.prior * sqrt_n / (1 + ch.N)
            if val > best_val:
                best_edge, best_val = (a, e), val
        return best_edge

    def _expand(self, node, a):
        ac = self.pool.get(node.ac_key)
        if isinstance(a, str) and a.startswith("OPT:"):
            pat = next(p for n, p in self.options[node.ac_key] if n == a)
            s, R, d, done = node.state, 0.0, 1.0, False
            for t in pat:
                s, r, done = self.env.step(s, t)
                R += d * r; d *= self.gamma
                if done:
                    break
            r_in, disc, ns, new_tokens = R, d, s, pat
        else:
            ns, r_in, done = self.env.step(node.state, a)
            disc = self.gamma
            new_tokens = (a,)
        key2 = self.env.signature(ns)
        ac2 = self.pool.get(key2)
        ctx2 = tuple((node.ctx + new_tokens)[-self.max_ctx:])
        q2 = replay(ac2, ctx2)
        child = self.TT.get((key2, ns))
        if child is None:
            qo2 = 0
            if self.ac_obs is not None:
                qo2 = self.ac_obs.step(node.q_obs, self.env.observe(ns))
            child = _Node(state=ns, ac_key=key2, q=q2, ctx=ctx2, q_obs=qo2,
                          action=a, is_option=isinstance(a, str))
            child.kind, child.conf = ac2.kind(q2), ac2.conf(q2)
            child.untried = [] if (done or self.env.is_terminal(ns)) \
                else self._untried(child)
            self.TT[(key2, ns)] = child
        node.children[a] = (r_in, disc, child)
        child.prior = self._priors(node).get(a, 1.0)
        return child

    def _priors(self, node) -> Dict[Any, float]:
        if node.priors is None:
            ac = self.pool.get(node.ac_key)
            store = self.pool.store
            vi = ac.value_iteration(gamma=self.look_gamma)
            raw = {}
            for a in node.untried or []:
                is_opt = isinstance(a, str) and a.startswith("OPT:")
                if is_opt:
                    pat = next(p for n, p in self.options[node.ac_key] if n == a)
                    v = self._option_walk(node, pat)
                    if v is None:
                        continue
                    k, cf = ac.kind(v), ac.conf(v)
                    b = ac.score(v) + (self.succ_bonus * cf if k is Kind.SUCCESS else 0.0)
                else:
                    v = ac.step(node.q, a)
                    k, cf = ac.kind(v), ac.conf(v)
                    if self.thompson and ac.pattern_weights(v):
                        b = 0.0
                        for pat, w_built in ac.pattern_weights(v):
                            pth = store.thompson(node.ac_key, pat, self.rng)
                            b += store.gain * (2 * pth - 1) if pth is not None else w_built
                    else:
                        b = ac.score(v)
                    b += self.look_kappa * float(vi[v])
                    if k is Kind.SUCCESS:
                        b += self.succ_bonus * cf
                    elif k is Kind.FAILURE:
                        b -= self.fail_pen * cf
                b += self.prox_s * ac.proximity(v, Kind.SUCCESS)
                b -= self.prox_f * ac.proximity(v, Kind.FAILURE)
                raw[a] = math.exp(b / self.prior_temp)
            z = sum(raw.values()) or 1.0
            node.priors = {a: x / z for a, x in raw.items()}
        return node.priors

    def _rollout(self, node) -> float:
        s, ac_key, ac = node.state, node.ac_key, self.pool.get(node.ac_key)
        ctx, q, qo = node.ctx, node.q, node.q_obs
        G, disc, phi = 0.0, 1.0, 0.0
        for _ in range(self.rollout_depth):
            if self.env.is_terminal(s):
                break
            a = self._rollout_action(ac_key, q, self.env.actions(s))
            ns, r, done = self.env.step(s, a)
            ctx2 = tuple((ctx + (a,))[-self.max_ctx:])
            q2 = replay(ac, ctx2)
            ph2 = phi + ac.score(q2)
            if self.ac_obs is not None:
                qo2 = self.ac_obs.step(qo, self.env.observe(ns))
                ph2 += self.ac_obs.score(qo2)
            else:
                qo2 = qo
            G += disc * (r + self.shaping * (self.gamma * ph2 - phi))
            phi, disc, s, ctx, q, qo = ph2, disc * self.gamma, ns, ctx2, q2, qo2
            if done:
                break
            kf = ac.kind(q2)
            ko = self.ac_obs.kind(qo2) if self.ac_obs is not None else Kind.NEUTRAL
            conf = max(ac.conf(q2), self.ac_obs.conf(qo2) if self.ac_obs else 0.0)
            if kf is Kind.FAILURE or ko is Kind.FAILURE:
                G -= disc * self.fail_pen * conf
                break
            if kf is Kind.SUCCESS or ko is Kind.SUCCESS:
                G += disc * self.succ_bonus * conf
                break
        return G

    def _rollout_action(self, b, q, acts):
        if self.rng.random() < self.r_eps:
            return self.rng.choice(acts)
        best, bv = acts[0], -INF
        for a in acts:
            v = self.R.get((b, q, a), 0.0) + self.rng.random() * 1e-6
            if v > bv:
                best, bv = a, v
        return best

    def _backup(self, edges, leaf: _Node, G: float) -> None:
        leaf.N += 1; leaf.W += G
        g = G
        for parent, a in reversed(edges):
            r_in, disc, child = parent.children[a]
            g = r_in + disc * g
            if not isinstance(a, str):
                key = (parent.ac_key, parent.q, a)
                tgt = r_in + disc * (child.W / max(child.N, 1))
                old = self.R.get(key, 0.0)
                self.R[key] = old + self.r_alpha * (tgt - old)
            parent.N += 1; parent.W += g

    def _explain(self, root, action):
        ac = self.pool.get(root.ac_key)
        store = self.pool.store
        adm = set(self._admissible(root))
        cands, shielded = [], []
        for a, (r_in, disc, ch) in root.children.items():
            is_opt = isinstance(a, str)
            v = self._option_walk(root, next(p for n, p in
                                             self.options.get(root.ac_key, [])
                                             if n == a)) if is_opt else ac.step(root.q, a)
            pws = ac.pattern_weights(v) if v is not None else []
            ci = store.ci(root.ac_key, max(pws, key=lambda pw: abs(pw[1]))[0]) if pws else None
            item = dict(a=a, qlabel=ac.label(v) if v is not None else "?",
                        kind=ac.kind(v) if v is not None else Kind.NEUTRAL,
                        conf=ac.conf(v) if v is not None else 0.0,
                        visits=ch.N, share=min(1.0, ch.N / max(root.N, 1)),
                        value=(ch.W / ch.N) if ch.N else 0.0,
                        prior=self._priors(root).get(a, 0.0), option=is_opt, ci=ci)
            if not is_opt and a not in adm:
                shielded.append(item)
            elif ch.N:
                cands.append(item)
        cands.sort(key=lambda x: -x['visits'])
        return cands, shielded


# ---------------------------------------------------------------------------
# 8. Serialization (+ risk budget persistence)
# ---------------------------------------------------------------------------
class PlannerState:
    @staticmethod
    def save(pl: ACMCTS, path: str) -> None:
        data = {"format": PLANNER_FORMAT, "version": FORMAT_VERSION,
                "verified_fail": [[list(p), _enc(b)] for (p, b) in pl.verified_fail],
                "verified_succ": [[list(p), _enc(b)] for (p, b) in pl.verified_succ],
                "qsamples": [[_enc(b), q, list(v)] for (b, q), v in pl.qsamples.items()],
                "rollout": [[_enc(b), q, a, round(v, 6)] for (b, q, a), v in pl.R.items()],
                "risk": pl.risk.save() if pl.risk else None}
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    @staticmethod
    def load(pl: ACMCTS, path: str) -> None:
        with open(path) as f:
            data = json.load(f)
        if data.get("format") != PLANNER_FORMAT:
            raise ValueError(f"not an {PLANNER_FORMAT} file: {path}")
        pl.verified_fail = {(tuple(p), _dec(b)) for p, b in data["verified_fail"]}
        pl.verified_succ = {(tuple(p), _dec(b)) for p, b in data["verified_succ"]}
        pl.qsamples = {(_dec(b), q): list(v) for b, q, v in data["qsamples"]}
        pl.R = {(_dec(b), q, a): v for b, q, a, v in data["rollout"]}
        if data.get("risk") and pl.risk is not None:
            pl.risk.load(data["risk"])


# ---------------------------------------------------------------------------
# 9. Episodes / training (risk events surfaced)
# ---------------------------------------------------------------------------
def run_episode(env, planner, store, store_obs, ac_obs, verifier, analytics, ep,
                max_steps=60, explain_steps=0, warm=None,
                temp=0.0, temp_moves=0, param_b: Optional[ParamBandits] = None):
    L = store.max_len
    s = env.reset()
    ac_key = env.signature(s)
    ctx = ()
    q_obs = ac_obs.step(0, env.observe(s)) if ac_obs else 0
    planner.new_episode(keep_tt=warm is not None)
    root = warm if (warm is not None and warm.state == s and warm.ctx == ctx) else None
    first_root = None
    tokens, rewards, sigs, obs_after = [], [], [], []
    agg, cnt = defaultdict(float), defaultdict(int)
    picks: Dict[Any, Tuple[int, int]] = {}
    total, done = 0.0, False
    risk_events: List[tuple] = []
    for t in range(max_steps):
        if param_b is not None and ac_key not in picks:
            ie, it = param_b.pick(ac_key)
            picks[ac_key] = (ie, it)
            planner.r_eps = param_b.ea[ie]
            planner.prior_temp = param_b.ta[it]
        q = replay(planner.pool.get(ac_key), ctx)
        a, root = planner.plan(s, ac_key, q, ctx=ctx, q_obs=q_obs, root=root,
                               explain=(t < explain_steps),
                               tau=(temp if t < temp_moves else 0.0))
        if first_root is None:
            first_root = root
        if isinstance(a, str) and a.startswith("OPT:"):
            seq = next(p for n, p in planner.options[ac_key] if n == a)
        else:
            seq = (a,)
        child_edge = root.children.get(a) if root else None
        for tk in seq:
            s0 = s
            s, r, done = env.step(s, tk)
            verifier.remember(s0, abs(r))
            tokens.append(tk); rewards.append(r); total += r
            b = env.signature(s)
            sigs.append(b); agg[b] += r; cnt[b] += 1
            obs_after.append(env.observe(s))
            if ac_obs:
                q_obs = ac_obs.step(q_obs, obs_after[-1])
        ac_key = env.signature(s)
        ctx = tuple((ctx + seq)[-L:])
        if not done and child_edge is not None and child_edge[2].state == s \
                and child_edge[2].ac_key == ac_key:
            root = child_edge[2]
        else:
            root = None
        if done:
            break
    store.learn_episode(tokens, rewards, sigs, final_return=total)
    ofp = None
    if ac_obs is not None:
        store_obs.learn_episode(obs_after, rewards, final_return=total)
        ofp = rebuild_obs(store_obs, ac_obs)
    if param_b is not None:
        for b, m in agg.items():
            if b in picks:
                param_b.update(b, picks[b][0], picks[b][1], m / max(cnt[b], 1))
    if planner.risk is not None:                        # ★ episode-level CVaR update
        for b in agg:
            nb, rec = planner.risk.update(b, agg[b])
            if nb:
                risk_events.append(('breach', b, planner.risk.cvar(b)))
            elif rec:
                risk_events.append(('recover', b, planner.risk.cvar(b)))
    return total, done, len(tokens), first_root, bmeans, ofp, risk_events


def verification_pass(store, pool, verifier, planner, budget=4, log=None):
    tested = 0
    for (b, pat), (w, k, c) in sorted(store.typed.items(), key=str):
        if k is Kind.NEUTRAL or tested >= budget:
            continue
        tested += 1
        if k is Kind.FAILURE:
            v = verifier.test(pat, b, -1)
            if v is not None:
                if v.ok:
                    planner.verified_fail.add((pat, b))
                else:
                    planner.verified_fail.discard((pat, b))
        else:
            v = verifier.test_option(pat, b)
            if v is not None:
                planner.verified_succ.discard((pat, b))
                if v.ok:
                    planner.verified_succ.add((pat, b))
                if log:
                    log(store.disp(pat), v)
    planner.refresh_options(store)


def train(env, episodes, use_obs=False, n_sims=120, seed=7,
          hook=None, budget=-1.5, param_bandits_on=False, kernel_sigma=0.0,
          store: Optional[PatternStore] = None, verbose=True):
    store = store or PatternStore(kernel_sigma=kernel_sigma)
    pool = ACPool(store); pool.rebuild()
    store_obs = ac_obs = None
    prev_ofp = None
    if use_obs:
        store_obs, ac_obs = PatternStore(min_abs_reward=1.5), AhoCorasick()
        ac_obs.build()
    verifier = Verifier(env, seed=seed)
    planner = ACMCTS(env, pool, ac_obs=ac_obs, n_simulations=n_sims, seed=seed)
    planner.risk = RiskBudget(budget=budget, seed_unused=None) if False \
        else RiskBudget(budget=budget)
    param_b = ParamBandits(seed=seed) if param_bandits_on else None
    drift = DriftMonitor()
    best, warm = -INF, None
    for ep in range(1, episodes + 1):
        if hook:
            hook(ep, env)
        ret, done, steps, first_root, bmeans, ofp, revents = run_episode(
            env, planner, store, store_obs, ac_obs, verifier, None, ep, warm=warm,
            param_b=param_b)
        for kind_, b, cv in revents:
            if kind_ == 'breach':
                print(f"  ⛔ risk breach in bucket {b}: CVaR.25 {cv:+.2f} < "
                      f"{planner.risk.budget:+.2f} → risk_beta "
                      f"{planner.risk_beta:.2f}→{planner.risk.beta_for(b, planner.risk_beta):.2f}, "
                      f"OPTIONS SUSPENDED")
            else:
                print(f"  ✅ bucket {b} back within budget (CVaR {cv:+.2f}) "
                      f"→ options restored")
        merged = store.generalize()
        changed = pool.rebuild() or (ofp != prev_ofp)
        prev_ofp = ofp
        for b, m in bmeans.items():
            if drift.update(b, m):
                print(f"  ⚠ drift in bucket {b} — halving ITS evidence only")
                store.halve(b)
                verifier.clear_bucket(b)
                planner.verified_fail = {k for k in planner.verified_fail if k[1] != b}
                planner.verified_succ = {k for k in planner.verified_succ if k[1] != b}
                warm = None
        verification_pass(store, pool, verifier, planner)
        warm = None if changed else first_root
        best = max(best, ret)
        if verbose:
            ns_, nf_ = pool.get(ACPool.GLOBAL).kind_counts()
            print(f"ep {ep:02d} | return {ret:7.2f} | steps {steps:3d} | S:{ns_:2d} "
                  f"F:{nf_:2d} | opts {sum(len(v) for v in planner.options.values())} | "
                  f"verif F:{len(planner.verified_fail)} S:{len(planner.verified_succ)} | "
                  f"warm {'Y' if warm else '-'} | best {best:7.2f}")
    return store, pool, planner, ac_obs, store_obs, verifier


# ---------------------------------------------------------------------------
# 10. Generic generation + trainers
# ---------------------------------------------------------------------------
def generate(env, planner, store, explain_last=True):
    planner.new_episode()
    s = env.reset()
    ctx: tuple = ()
    qo = planner.ac_obs.step(0, env.observe(s)) if planner.ac_obs else 0
    root = None
    out: List[Any] = []
    L = store.max_len
    for t in range(env.max_len):
        q = replay(planner.pool.get(ACPool.GLOBAL), ctx)
        a, root = planner.plan(s, ACPool.GLOBAL, q, ctx=ctx, q_obs=qo, root=root,
                               explain=(explain_last and t == env.max_len - 1))
        if env.stop_token is not None and a == env.stop_token:
            out.append(a)
            break
        s, r, done = env.step(s, a)
        out.append(a)
        ctx = tuple((ctx + (a,))[-L:])
        if planner.ac_obs is not None:
            qo = planner.ac_obs.step(qo, env.observe(s))
        if done:
            break
    return out


def train_env_gen(env, episodes=8, n_sims=90, seed=3, bootstrap=None, verbose=True):
    store = PatternStore()
    pool = ACPool(store); pool.rebuild()
    store_obs, ac_obs = PatternStore(min_abs_reward=1.5), AhoCorasick()
    ac_obs.build()
    if bootstrap:
        nb = bootstrap_corpus(store, bootstrap)
        pool.rebuild()
        if verbose:
            print(f"  corpus bootstrap: {nb} n-gram priors")
    verifier = Verifier(env, seed=seed)

    def vlog(pat, v: Verdict):
        print(f"    ✓ OPTION '{pat}': mean {v.mean:+.2f} CVaR{v.alpha:.2f} "
              f"{v.cvar:+.2f} → {'OK' if v.ok else 'REJECTED'}")
    planner = ACMCTS(env, pool, ac_obs=ac_obs, n_simulations=n_sims,
                     rollout_depth=16, seed=seed)
    best = -INF
    for ep in range(1, episodes + 1):
        ret, done, steps, *_rest = run_episode(env, planner, store, store_obs,
                                               ac_obs, verifier, None, ep)[:4] \
            + (None, None, None, [])
        pool.rebuild()
        rebuild_obs(store_obs, ac_obs)
        verification_pass(store, pool, verifier, planner, log=vlog)
        best = max(best, ret)
        ns_, nf_ = pool.get(ACPool.GLOBAL).kind_counts()
        if verbose:
            print(f"  ep {ep:02d} | return {ret:6.2f} | S:{ns_} F:{nf_} | "
                  f"opts {sum(len(v) for v in planner.options.values())} | best {best:.2f}")
    return planner, store, store_obs, ac_obs, verifier


# ---------------------------------------------------------------------------
# 11. Demo
# ---------------------------------------------------------------------------
def collect(env, n_ep=6, max_steps=25, seed=0):
    """Scripted distance-greedy episodes for alignment statistics."""
    rng = random.Random(seed)
    eps = []
    for _ in range(n_ep):
        s = env.reset()
        toks, rews = [], []
        for _t in range(max_steps):
            acts = env.actions(s)
            (i, j), _ = s
            gi, gj = env.goal
            best = min(acts, key=lambda a: abs(i + env.MOVES[a][0] - gi)
                       + abs(j + env.MOVES[a][1] - gj))
            a = best if rng.random() < 0.7 else rng.choice(acts)
            s, r, done = env.step(s, a)
            toks.append(a); rews.append(r)
            if done:
                break
        eps.append((toks, rews))
    return eps


def show_typed(store, tag="", buckets=None):
    for (b, pat), (w, k, c) in sorted(store.typed.items(), key=str):
        if k is not Kind.NEUTRAL and (buckets is None or b in buckets):
            print(f"  [{tag}] '{store.disp(pat)}' @bucket {b} {k.name} "
                  f"conf={c:.2f} w={w:+.2f}")


def main():
    # ---- 1. Alignment: renamed vocabulary, recovered by statistics ----
    print("═══ 1. TOKEN ALIGNMENT: {U,D,L,R} → {N,S,E,W} ═══")
    envA, envB = ComboGrid(n=6, seed=0), RenamedGrid(n=6, seed=1)
    TRUE_MAP = {'U': 'N', 'D': 'S', 'L': 'W', 'R': 'E'}
    storeA = train(envA, episodes=6, n_sims=70, seed=0, verbose=False)[0]
    src_eps = collect(envA, seed=2)
    tgt_eps = collect(envB, seed=3)
    aligner = VocabAligner()
    proposed, report = aligner.propose(src_eps, tgt_eps)
    print("  proposals (token, match, score, reward-sim, ctx-sim):")
    for s, t, sc, rw, cx in report[:8]:
        mark = "✔" if TRUE_MAP.get(s) == t else "✘"
        print(f"    {s} ↔ {t}  score={sc:.2f} (rew {rw:.2f}, ctx {cx:.2f}) {mark}")
    print(f"  curated map == truth: {proposed == TRUE_MAP}")
    storeB = PatternStore()
    n0 = storeB.port_from(storeA)                       # no map → everything drops
    n1 = storeB.port_from(storeA, token_map=proposed, scale=0.6)
    print(f"  port without map: {n0} entries (drops) | with curated map: {n1} entries")

    # ---- 2. Episode-level risk budgets on a heavy-tail grid ----
    print("\n═══ 2. RISK BUDGETS: noisy RRR + hazards → breach / suspend / recover ═══")
    env2 = ComboGrid(n=6, hazards=((2, 2), (2, 3)),
                     noisy={('R', 'R', 'R'): 8.0}, seed=4)
    train(env2, episodes=14, n_sims=90, seed=4, budget=-1.0, use_obs=False)

    # ---- 3. Compositional grammar domain ----
    print("\n═══ 3. GRAMMAR DOMAIN: S→NP VP, VP→verb (NP)? — per-slot buckets ═══")
    genv = GrammarGenEnv(seed=5)
    gplanner, gstore, *_ = train_env_gen(genv, episodes=8, n_sims=90, seed=5)
    print("  typed patterns per syntactic bucket:")
    show_typed(gstore, "gram")
    seq = generate(genv, gplanner, gstore)
    print(f"  generated sentence: {' '.join(seq)}")

    # ---- 4. Multi-source transfer with an adversarial source ----
    print("\n═══ 4. MULTI-SOURCE PORT: 2 honest sources + 1 sign-flipped ═══")
    envH1 = ComboGrid(n=6, seed=6)
    sH1 = train(envH1, episodes=5, n_sims=60, seed=6, verbose=False)[0]
    envH2 = ComboGrid(n=6, seed=8)
    sH2 = train(envH2, episodes=5, n_sims=60, seed=8, verbose=False)[0]
    envF = ComboGrid(n=6, seed=7)

    def flip(ep, env):
        env.set_sign(0, -1.0); env.set_sign(1, -1.0)     # RRR−5, DDD+5
    sF = train(envF, episodes=5, n_sims=60, seed=7, hook=flip, verbose=False)[0]
    tgt = PatternStore()
    n, rep = tgt.port_multi([(sH1, None, None), (sH2, None, None), (sF, None, None)],
                            base_scale=0.5, unanimous_boost=0.3, disagree_pen=0.4)
    print(f"  ported {n} patterns from 3 sources; consensus report:")
    for b, disp, nsr, ag, dg, boost in rep:
        if nsr >= 2:
            print(f"    '{disp}' bucket {b}: {nsr} sources, agree {ag}, "
                  f"disagree {dg} → boost ×{boost:.2f}")
    pool = ACPool(tgt); pool.rebuild()
    print("  resulting typing in target (DDD must be FAILURE despite flipped source):")
    show_typed(tgt, "multi", buckets={'G'})

    # ---- 5. Template-gen regression + serialization ----
    print("\n═══ 5. LOG-domain regression + serialization round-trip ═══")
    lenv = TemplateGenEnv(GEN_CFG['log'], seed=3)
    lplanner, lstore, lso, lao, lver = train_env_gen(
        lenv, episodes=6, n_sims=80, seed=3, bootstrap=GEN_CFG['log']['families'])
    print("  generated:", " ".join(generate(lenv, lplanner, lstore)))
    lstore.save("ac_patterns.json")
    PlannerState.save(lplanner, "ac_planner.json")
    store_b = PatternStore.load("ac_patterns.json")
    pool_b = ACPool(store_b); pool_b.rebuild()
    planner_b = ACMCTS(lenv, pool_b, ac_obs=lao, seed=99)
    planner_b.risk = RiskBudget()
    PlannerState.load(planner_b, "ac_planner.json")
    planner_b.refresh_options(store_b)
    print(f"  restored: verified S:{len(planner_b.verified_succ)} | "
          f"|R| {len(planner_b.R)} | risk buffers {len(planner_b.risk.buf)}")
    print("  loaded-agent generation:", " ".join(generate(lenv, planner_b, store_b)))


if __name__ == "__main__":
    main()