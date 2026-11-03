(ns org.replikativ.foerster.contract
  "What an inference result vouches for: its `:inference-contract`, built
  here and nowhere else, from a closed vocabulary. A contract states what
  the algorithm guarantees by construction; it is not a check that a run
  converged.

    :semantics        what the particles are (`semantics`)
    :normalizer       {:status s} how `measure/log-marginal` may be read
                      (`normalizer-statuses`), with the `:estimator` it
                      names when s is :estimate
    :weight-contract  obligations the weights rest on (`weight-contracts`)
    :adaptation       what the algorithm tuned from its own population
                      (`adaptations`)
    :approximation    what it approximated on purpose (`approximations`)

  Construct contracts with `weighted-population` and `markov-chain`; read
  them with `normalizer-status` and `unbiased-evidence?`. An unknown key or
  value is refused, so a new guarantee is named here first.")

(def semantics
  {:weighted-population "weighted particles for the target"
   :approximate-population "weighted particles for an approximation of the target"
   :markov-chain "autocorrelated draws of Markov chains; their weights combine samples"})

(def normalizer-statuses
  {:exact "the evidence itself (enumeration)"
   :unbiased "an unbiased estimate of the evidence, by construction of the algorithm"
   :estimate "an estimate by the named estimator, without an unbiasedness claim"
   :unverified "the population's log mass; what it estimates rests on obligations the run does not check"
   :unavailable "no evidence estimate (chain draws)"})

(def weight-contracts
  {:requires-operation-assumptions
   "the weights are correct only if user scores, proposals and moves meet their obligations"})

(def adaptations
  {:population "targets, schedules or proposals chosen from the current population"})

(def approximations
  {:coreset-rejuvenation "moves target a compressed core-set of the past likelihood factors"})

(def ^:private vocabulary
  {:semantics semantics :weight-contract weight-contracts
   :adaptation adaptations :approximation approximations})

(defn check
  "`contract` if every key and value is in the vocabulary, else an error
  naming the first that is not."
  [contract]
  (doseq [[k v] contract]
    (case k
      :normalizer (do (when-not (contains? normalizer-statuses (:status v))
                        (throw (ex-info (str "Unknown normalizer status " (:status v))
                                        {:type ::unknown-term :key k :value v})))
                      (when (not= (= :estimate (:status v)) (contains? v :estimator))
                        (throw (ex-info "A normalizer names its estimator exactly when its status is :estimate"
                                        {:type ::unknown-term :key k :value v}))))
      (when-not (contains? (get vocabulary k) v)
        (throw (ex-info (str "Unknown contract term " k " " v)
                        {:type ::unknown-term :key k :value v})))))
  contract)

(defn weighted-population
  "A population result's contract: `:semantics` :weighted-population unless
  given, the normalizer `:unverified` unless `:normalizer` names another
  status (with `:estimator` for :estimate), and the optional terms given."
  ([] (weighted-population {}))
  ([{:keys [semantics normalizer estimator weight-contract adaptation approximation]
     :or {semantics :weighted-population normalizer :unverified}}]
   (check (cond-> {:semantics semantics
                   :normalizer (cond-> {:status normalizer} estimator (assoc :estimator estimator))}
            weight-contract (assoc :weight-contract weight-contract)
            adaptation (assoc :adaptation adaptation)
            approximation (assoc :approximation approximation)))))

(defn markov-chain
  "Chain draws' contract: no evidence, unless the chain method computes a
  separate estimate by `estimator` (parallel tempering's stepping stones)."
  ([] (markov-chain {}))
  ([{:keys [estimator]}]
   (check {:semantics :markov-chain
           :normalizer (if estimator {:status :estimate :estimator estimator} {:status :unavailable})})))

(defn adapted
  "`contract` noting that the algorithm adapted `adaptation` from its own
  population."
  [contract adaptation]
  (check (assoc contract :adaptation adaptation)))

(defn normalizer-status
  "The normalizer status of a result (anything carrying `:inference-contract`)."
  [result]
  (get-in result [:inference-contract :normalizer :status]))

(defn unbiased-evidence?
  "Whether `result`'s `log-marginal` is the evidence or an unbiased estimate
  of it (on the natural scale): what a pseudo-marginal method may consume."
  [result]
  (contains? #{:exact :unbiased} (normalizer-status result)))
