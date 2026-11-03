(ns org.replikativ.foerster.proposal
  "Proposals: where a latent site's value comes from when inference does not
  take it from the site's own law. One protocol serves every place that
  proposes a site's value: SMC and importance sampling (`:proposal`), a
  policy's `:draw`, MH moves and resample-move (`:propose`).

  A proposal is asked about a `site`, a map:

    :address  the site's address
    :dist     its law
    :options  its options (`:proposal`, `:init`, …)
    :old      its value in the trace being moved (absent for a fresh draw)
    :trace    a delay of the particle's choices so far, {address {:value …}}
              (deref it only if the proposal needs them: it is built on
              demand)

  `(propose p site)` draws from the current generator (`foerster.random`)
  and returns {:value v :log-q lq}, lq the log density of drawing v, or
  {:value v :symmetric? true} for a move whose reverse is as likely, or
  nil: the site is not this proposal's (the site's own law decides it).
  `(log-q p site v)` is that log density for a given v, nil if unknown;
  an MH move scores its reverse with it, so an asymmetric proposal must give
  it.

  The weight of a proposed value is log p(v) − log q(v): the target is
  unchanged, whatever the proposal. That holds only if `log-q` is the
  density `propose` draws from. The built-in proposals (`prior`,
  `distribution`, `random-walk`) are so by construction (`verified?`); a
  proposal made with `from-fn` is the caller's obligation, and results using
  it say so (`foerster.contract`: `:unverified` evidence)."
  (:require [org.replikativ.foerster.dist :as dist]))

(defprotocol PProposal
  (propose [p site]
    "A draw for `site`: {:value v :log-q lq}, {:value v :symmetric? true},
    or nil when the site is not this proposal's.")
  (log-q [p site v]
    "The log density of proposing `v` at `site`, or nil when unknown.")
  (verified? [p]
    "Whether `log-q` is, by construction, the density `propose` draws from."))

(defn proposal? [x] (satisfies? PProposal x))

(defrecord Prior []
  PProposal
  (propose [_ {d :dist}]
    (let [v (dist/draw d)] {:value v :log-q (dist/draw-logpdf d v)}))
  (log-q [_ {d :dist} v] (dist/draw-logpdf d v))
  (verified? [_] true))

(def prior
  "A fresh draw from the site's own law (a block's `:sample` density where
  it has one): importance sampling's and SMC's default."
  (->Prior))

(defn prior?
  "Whether `p` is the prior proposal."
  [p]
  (instance? Prior p))

(defrecord Distribution [q]
  PProposal
  (propose [_ _] (let [v (dist/draw q)] {:value v :log-q (dist/logpdf q v)}))
  (log-q [_ _ v] (dist/logpdf q v))
  (verified? [_] true))

(defn distribution
  "Draws from the distribution `q` at every site it is asked about, whatever
  the site's old value (an independence proposal). Restrict it to some
  sites with `only`."
  [q]
  (->Distribution q))

(defn- step-map
  "Every value of the map `old` stepped by N(0, `step-size`²), in the map's
  order."
  [old step-size]
  (into {} (map (fn [[a v]] [a (+ v (* step-size (dist/draw (dist/normal 0.0 1.0))))])) old))

(defrecord RandomWalk [step-size]
  PProposal
  (propose [_ {d :dist old :old :as site}]
    (cond
      (and (number? old) (dist/continuous? d))
      {:value (+ old (* step-size (dist/draw (dist/normal 0.0 1.0)))) :symmetric? true}
      ;; a parameter map with no law of its own (PMMH's θ)
      (and (nil? d) (map? old) (every? number? (vals old)))
      {:value (step-map old step-size) :symmetric? true}
      :else (propose prior site)))
  (log-q [_ {d :dist old :old :as site} v]
    (cond
      (and (number? old) (dist/continuous? d)) (dist/logpdf (dist/normal old step-size) v)
      (and (nil? d) (map? old) (every? number? (vals old)))
      (reduce + (map (fn [[a x]] (dist/logpdf (dist/normal (get old a) step-size) x)) v))
      :else (log-q prior site v)))
  (verified? [_] true))

(defn random-walk
  "A symmetric Gaussian step of `step-size` around a real-valued site's old
  value, or around every parameter of a map of them (PMMH's θ). A site that
  has no old value, or whose law is not continuous (a boolean, a count, a
  vector), gets the prior."
  [step-size]
  (->RandomWalk step-size))

(defrecord Only [selected? p]
  PProposal
  (propose [_ site] (when (selected? (:address site)) (propose p site)))
  (log-q [_ site v] (when (selected? (:address site)) (log-q p site v)))
  (verified? [_] (verified? p)))

(defn only
  "`p` at the sites whose address `selected?` holds (a set of addresses or a
  predicate); every other site keeps its own law."
  [selected? p]
  (->Only selected? p))

(defrecord FnProposal [propose-fn log-q-fn]
  PProposal
  (propose [_ site]
    (when-let [r (propose-fn site)]
      (when-not (or (:symmetric? r) (number? (:log-q r)))
        (throw (ex-info "A proposal returns :log-q or :symmetric? true"
                        {:type ::malformed-proposal :address (:address site) :proposal (dissoc r :value)})))
      r))
  (log-q [_ site v] (when log-q-fn (log-q-fn site v)))
  (verified? [_] false))

(defn from-fn
  "A proposal from `(propose-fn site)` -> {:value v :log-q lq},
  {:value v :symmetric? true} or nil, and optionally `(log-q-fn site v)`, the
  density an MH move scores its reverse with (without it an asymmetric
  proposal's reverse is scored under the prior). Its correctness is the
  caller's obligation."
  ([propose-fn] (from-fn propose-fn nil))
  ([propose-fn log-q-fn] (->FnProposal propose-fn log-q-fn)))
