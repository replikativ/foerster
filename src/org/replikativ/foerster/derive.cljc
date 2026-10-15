(ns org.replikativ.foerster.derive
  "Inference plans derived from a model's site graph (`foerster.analysis`).

  `plan` decides, from the graph alone, which finite latent sites are
  summed out where they are drawn (local enumeration): such a site z is
  forked once per value, each fork scores z and the observations that read
  it, and the particle continues with one value drawn in proportion to its
  fork's weight, weighted by their sum. Its weight is then the marginal
  over z, and z's value an exact draw from its conditional.

  A latent site z is enumerated when
  - its distribution has finite support;
  - every site that reads it, or runs depending on it (its children), is an
    observation or a factor — no latent child;
  - no later iteration of a loop reads it;
  - its group is contiguous: every site written between z and its last
    child is a child of z, so a run that leaves the group is done with z;
  - its location in the source is known, to match it at run time, and no
    other site is written there.

  `enumerate-site` does it at run time, for an engine's site handler
  (`foerster.smc/smc` with `:enumerate plan`). Correctness does not depend on
  the conditions: a fork that stops early scores fewer observations, and the
  draw of z is still a proposal the weight accounts for. They make the
  marginal exact."
  (:require [org.replikativ.foerster.dist :as dist]
            [org.replikativ.foerster.measure :as m]
            [org.replikativ.foerster.random :as random]
            [org.replikativ.foerster.trace :as itrace]
            [org.replikativ.spindel.effects.savepoint :as sp]
            [org.replikativ.spindel.engine.protocols :as rtp]
            [org.replikativ.spindel.trace :as trace]))

(def ^:private finite-ctors
  "Constructors of `foerster.dist` with finite support."
  '{bernoulli :finite flip :finite binomial :finite discrete :finite
    categorical :finite uniform-discrete :finite})

(defn- children [sites n]
  (into #{} (keep (fn [[m s]] (when (or (contains? (:reads s) n) (contains? (:control s) n)) m))) sites))

(defn- refusal
  "Why the latent site `n` of graph `g` is not enumerated, or nil."
  [{:keys [sites loops]} n]
  (let [s (sites n)
        kids (children sites n)]
    (cond
      (not= :finite (get finite-ctors (get-in s [:dist :ctor])))
      :infinite-support

      (empty? kids) :no-children

      (some #(contains? (:carried (loops %)) n) (:loops s)) :carried

      (some #(not (#{:observed :factor} (:kind (sites %)))) kids) :latent-child

      (not= (set (range (inc n) (inc (apply max kids)))) kids) :not-contiguous

      (some nil? (map (comp :loc sites) (cons n kids))) :no-location)))

(defn plan
  "The inference plan for the site graph `g`:
    {:enumerate {loc {:site n :address a :group #{loc …}}}
     :refused   {n reason}}
  keyed by the source location sites carry at run time (`:loc` of a
  `choose` payload). `:refused` lists the finite latent sites left alone."
  [{:keys [sites] :as g}]
  (let [finite (for [[n s] sites
                     :when (and (= :latent (:kind s))
                                (= :finite (get finite-ctors (get-in s [:dist :ctor]))))]
                 n)
        ;; a location two static sites share (code a macro generates) cannot
        ;; tell them apart at run time
        shared (into #{} (keep (fn [[loc k]] (when (and loc (< 1 k)) loc)))
                     (frequencies (map :loc (vals sites))))
        decided (for [n finite]
                  [n (or (refusal g n)
                         (when (some shared (map (comp :loc sites) (cons n (children sites n))))
                           :shared-location))])]
    {:enumerate (into {} (for [[n reason] decided :when (nil? reason)
                               :let [s (sites n)]]
                           [(:loc s) {:site n
                                      :address (:address s)
                                      :group (into #{} (map (comp :loc sites)) (children sites n))}]))
     :refused (into {} (filter second) decided)}))

;; -----------------------------------------------------------------------------
;; At run time
;; -----------------------------------------------------------------------------

(def max-support
  "Sites with more values than this are drawn as usual."
  64)

(defn enumerated
  "The plan's entry for the savepoint `s`, if it is a site the plan
  enumerates."
  [plan s]
  (when (= itrace/choose-site (:savepoint/site s))
    (get (:enumerate plan) (:loc (:savepoint/payload s)))))

(defn- weight-of [world] (or (rtp/get-state world [:inference :log-weight]) 0.0))

(defn- decide!
  "Decide `s` under `policy`, recording it in its world's trace."
  [policy s]
  (let [decision (policy s nil)]
    (trace/record! (:savepoint/world s) s decision nil)
    decision))

(defn- scores
  "CPS: Δ_k for each value k of the site `s` — the log weight of z = k and
  of the sites of its `group` that run after it, each scored in a fork of
  `s` under `policy`. A fork ends at the first site outside the group (or
  the result), or as soon as its weight is impossible."
  [s values group policy session]
  (fn [resolve reject]
    (let [address (:savepoint/address s)
          k (count values)
          out (object-array k)
          remaining (atom k)
          failed? (atom false)
          done! (fn [i score]
                  (aset out i score)
                  (when (zero? (swap! remaining dec)) (resolve (vec out))))
          fail! (fn [e] (when (compare-and-set! failed? false true) (reject e)))
          end! (fn [i x]
                 (let [w (:savepoint/world x)]
                   (when (:savepoint/address x)
                     (try (sp/abandon x) (catch #?(:clj Throwable :cljs :default) _ nil)))
                   (let [score (weight-of w)]
                     (sp/release-world! session w)
                     (done! i score))))
          table (fn [i]
                  {sp/any-site
                   (fn [x]
                     (try
                       (let [site (:savepoint/site x)]
                         (cond
                           ;; computed values pass through
                           (= itrace/deterministic-site site)
                           (sp/resume x (:value (decide! policy x)))

                           (contains? group (:loc (:savepoint/payload x)))
                           (let [{:keys [value]} (decide! policy x)]
                             (if (= ##-Inf (weight-of (:savepoint/world x)))
                               (end! i x)
                               (sp/resume x value)))

                           :else (end! i x)))
                       (catch #?(:clj Throwable :cljs :default) e (fail! e))))
                   sp/result-site
                   (fn [{w :savepoint/world}]
                     (let [score (weight-of w)]
                       (sp/release-world! session w)
                       (done! i score)))
                   sp/error-site (fn [{error :savepoint/payload}] (fail! error))
                   sp/abandoned-site (fn [_] nil)})]
      (dotimes [i k]
        ((sp/fork s {:handlers (table i)})
         (fn [child]
           (try
             (rtp/swap-state! (:savepoint/world child) [:inference :log-weight] (constantly 0.0))
             ;; a constraint adds log p(z = k)
             (let [{:keys [value]} (decide! (itrace/policy (assoc (itrace/policy-options policy)
                                                                  :constraints {address (nth values i)}))
                                            child)]
               (if (= ##-Inf (weight-of (:savepoint/world child)))
                 (end! i child)
                 (sp/resume child value)))
             (catch #?(:clj Throwable :cljs :default) e (fail! e))))
         fail!)))))

(defn enumerable?
  "Whether the site `s` can be enumerated under `policy` at run time: a
  `foerster.trace/policy` that does not decide `s` itself, and a finite
  support of at most `max-support` values."
  [s policy]
  (let [opts (itrace/policy-options policy)
        address (:savepoint/address s)
        values (dist/support (:dist (:savepoint/payload s)))]
    (and (contains? (meta policy) :org.replikativ.foerster.trace/options)
         (not (:draw opts))
         (not (contains? (:constraints opts) address))
         (not (contains? (:interventions opts) address))
         (not (contains? (:noise opts) address))
         (seq values)
         (<= (count values) max-support))))

(defn enumerate-site
  "CPS: sum out the site `s` (planned as `entry`) under `policy`. Resolves
  the policy that decides `s`: it draws z = k with probability ∝ exp Δ_k
  from the world's stream for `s`, with log-proposal Δ_k − log Σ exp Δ, so
  deciding it adds log Σ_k exp Δ_k minus what the group's sites will add
  after it. A site that is impossible for every value gets weight −∞."
  [s entry policy session]
  (fn [resolve reject]
    (let [address (:savepoint/address s)
          values (vec (dist/support (:dist (:savepoint/payload s))))
          opts (itrace/policy-options policy)]
      ((scores s values (:group entry) policy session)
       (fn [deltas]
         (try
           (let [total (m/log-sum-exp deltas)
                 [i lq] (if (= ##-Inf total)
                          [0 ##Inf]
                          (let [i (random/in-world-stream (:savepoint/world s) address
                                                          #(m/sample-categorical (m/normalize-log-weights deltas)))]
                            [i (- (nth deltas i) total)]))
                 v (nth values i)]
             (resolve (itrace/policy (assoc opts :draw (fn [x _]
                                                         (when (= address (:savepoint/address x))
                                                           {:value v :log-proposal lq}))))))
           (catch #?(:clj Throwable :cljs :default) e (reject e))))
       reject))))
