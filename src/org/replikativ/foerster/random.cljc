(ns org.replikativ.foerster.random
  "Where inference draws its randomness from.

  Every draw of a seeded inference run must depend only on WHAT is drawn,
  not on when: particles, chains and proposals run concurrently, and one
  generator shared in arrival order makes a seeded run reproducible only on
  a serial executor. So a draw made while deciding something in a world —
  a sample site, the site an MH move selects, its acceptance — reads a
  STREAM keyed by that world's seed and what is being decided:

    stream(world, key) = generator seeded by hash(seed(world), key)

  World seeds are derived per fork from the parent's seed, the site address
  and the fork index (`savepoint/fork`), so a replay's fresh proposal draws
  from a fresh stream and the same run draws the same numbers. Draws outside
  any world — the seeds of an inference's sessions, SMC's resampling at a
  barrier — come from the process generator in program order, which
  `set-seed!` seeds.

  The generator is xoshiro128** on 32-bit words, so a seed gives the same
  numbers on the JVM and in JavaScript."
  (:require [clojure.string :as str]
            [clojure.walk :as walk]
            [org.replikativ.spindel.effects.savepoint :as sp]
            [org.replikativ.spindel.engine.hash :as h]))

;; -----------------------------------------------------------------------------
;; xoshiro128** (Blackman & Vigna) on unsigned 32-bit words
;; -----------------------------------------------------------------------------

(defn- u32 [x]
  #?(:clj (bit-and (long x) 0xFFFFFFFF)
     :cljs (unsigned-bit-shift-right x 0)))

(defn- shl [x k]
  #?(:clj (bit-and (bit-shift-left (long x) k) 0xFFFFFFFF)
     :cljs (u32 (bit-shift-left x k))))

(defn- shr [x k] (unsigned-bit-shift-right x k))

(defn- bxor [a b] (u32 (bit-xor a b)))

(defn- mul [a b]
  #?(:clj (u32 (* (long a) (long b)))
     :cljs (u32 (js/Math.imul a b))))

(defn- rotl [x k] (u32 (bit-or (shl x k) (shr x (- 32 k)))))

(defn- step
  "[output next-state] of state [s0 s1 s2 s3]."
  [[s0 s1 s2 s3]]
  (let [out (mul (rotl (mul s1 5) 7) 9)
        t (shl s1 9)
        s2 (bxor s2 s0)
        s3 (bxor s3 s1)
        s1 (bxor s1 s2)
        s0 (bxor s0 s3)
        s2 (bxor s2 t)
        s3 (rotl s3 11)]
    [out [s0 s1 s2 s3]]))

(defn- parse-hex [s]
  #?(:clj (Long/parseLong s 16)
     :cljs (js/parseInt s 16)))

(defn- canonical
  "`x` with its numbers in one form: a JVM long 7 and a JavaScript 7 hash
  differently, so whole numbers become their decimal string."
  [x]
  (cond
    (number? x) (if (== x (Math/floor x))
                  (str #?(:clj (long x) :cljs x))
                  x)
    (vector? x) (mapv canonical x)
    (coll? x) (walk/walk canonical identity x)
    :else x))

(defn- state-of
  "A generator state from any value: its content hash, as four words."
  [x]
  (let [uuid (h/content-hash [::stream (canonical x)])
        words #?(:clj (let [hi (.getMostSignificantBits ^java.util.UUID uuid)
                            lo (.getLeastSignificantBits ^java.util.UUID uuid)]
                        [(u32 (unsigned-bit-shift-right hi 32)) (u32 hi)
                         (u32 (unsigned-bit-shift-right lo 32)) (u32 lo)])
                 :cljs (let [hex (str/replace (str uuid) "-" "")]
                         (mapv #(parse-hex (subs hex (* 8 %) (* 8 (inc %)))) (range 4))))]
    ;; the all-zero state is a fixed point
    (if (every? zero? words) [1 0 0 0] words)))

(defn generator
  "A generator seeded by `seed` (any value)."
  [seed]
  (atom (state-of seed)))

(defn next-u32!
  "The next unsigned 32-bit word of `g`."
  [g]
  (let [[before _] (swap-vals! g (comp second step))]
    ;; Only the output word is needed here; swap-vals! already advanced the
    ;; state. Computing step again allocated a second discarded state vector.
    (mul (rotl (mul (nth before 1) 5) 7) 9)))

(defn next-double!
  "The next double in [0, 1) of `g`: 53 random bits."
  [g]
  (let [a (shr (next-u32! g) 5)
        b (shr (next-u32! g) 6)]
    (/ (+ (* a 67108864.0) b) 9007199254740992.0)))

;; -----------------------------------------------------------------------------
;; Process generator and world streams
;; -----------------------------------------------------------------------------

(defonce ^:private process (generator 0))

(defn set-seed!
  "Seed the process generator: the draws made outside any world stream."
  [seed]
  (reset! process (state-of seed))
  nil)

(def ^:dynamic *stream*
  "The generator (or delayed generator) draws come from, or nil for the process generator."
  nil)

(defn current
  "The generator draws come from now."
  []
  (force (or *stream* process)))

(defn uniform01
  "A uniform draw in [0, 1) from the current generator."
  []
  (next-double! (current)))

(def ^:dynamic *stream-id*
  "`(volatile! [id n])`: the identity of the current keyed stream and the
  number of decisions drawn from it so far, or nil on the process generator."
  nil)

(defn with-stream*
  "Call `f` with draws coming from the stream keyed by `seed` and `key`. A nil
  seed (no session) leaves draws on the process generator. The keyed
  generator is seeded only when a draw actually needs it."
  [seed key f]
  (if (some? seed)
    (binding [*stream* (delay (generator [seed key]))
              *stream-id* (volatile! [[seed key] -1])]
      (f))
    (f)))

;; -----------------------------------------------------------------------------
;; Decisions
;; -----------------------------------------------------------------------------
;; A draw that DECIDES something among finitely many outcomes — a discrete
;; site's value, a resampled ancestor, the site a move selects, its
;; acceptance — goes through `decide` with the outcomes' probabilities. In a
;; run it draws exactly as before (`default`). An oracle installed with
;; `with-oracle` answers instead: given the decision's key (its stream and
;; position in it) and probabilities, it returns the outcome. That is how
;; `foerster.enumerate-sweeps` follows every path of an inference step with
;; its exact probability.

(defonce ^:private oracle (atom nil))
(defonce ^:private process-decisions (atom -1))

(defn- decision-key []
  (if-let [v *stream-id*]
    (let [[id n] (vswap! v (fn [[id n]] [id (inc n)]))] [id n])
    [::process (swap! process-decisions inc)]))

(defn decide
  "The index of the outcome drawn with probabilities `(probs)` (normalized),
  computed by `default` from the current generator, or by the installed
  oracle. `probs` is a thunk: a run never builds it."
  [probs default]
  (if-let [o @oracle]
    (o (decision-key) (vec (probs)))
    (default)))

(defn below?
  "True with probability `p`, drawn as `(< u p)`."
  [p]
  (= 1 (decide #(vector (- 1.0 p) p) #(if (< (uniform01) p) 1 0))))

(defn accept-log?
  "The Metropolis–Hastings accept test for a log ratio below zero, drawn as
  `(< (log u) log-ratio)`: true with probability exp(log-ratio)."
  [log-ratio]
  (= 1 (decide #(let [p (Math/exp (min 0.0 log-ratio))] [(- 1.0 p) p])
               #(if (< (Math/log (uniform01)) log-ratio) 1 0))))

(defn with-oracle
  "Call `f` with every decision answered by `(oracle key probs)`. For
  enumeration in tests: one oracle at a time, process-wide."
  [oracle-fn f]
  (when-not (compare-and-set! oracle nil oracle-fn)
    (throw (ex-info "An oracle is already installed" {:type ::oracle-installed})))
  (reset! process-decisions -1)
  (try (f) (finally (reset! oracle nil))))

(defn in-world-stream
  "Call `f` drawing from `world`'s stream for `key`."
  [world key f]
  (with-stream* (sp/seed world) key f))

(defn fresh-seed
  "A seed for a new session, drawn from the current generator: in a seeded
  run, the same seeds in the same program order."
  []
  (let [g (current)]
    [(next-u32! g) (next-u32! g)]))
