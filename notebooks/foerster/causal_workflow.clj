;; # From Data to Counterfactual Repairs
;;
;; A declared causal graph, fitted mechanisms, checks on held-out data,
;; then an incident and candidate repairs. This synthetic system has a known
;; answer, so we can check the whole workflow. All fitting and evaluation
;; code runs natively in Clojure and ClojureScript.

(ns foerster.causal-workflow
  (:require [org.replikativ.foerster.causal-model :as cm]
            [org.replikativ.foerster.counterfactual :as cf]
            [org.replikativ.foerster.attribution :as attribution]
            [org.replikativ.foerster.dist :as dist]
            [org.replikativ.foerster.random :as random]
            [org.replikativ.spindel.core :as sp]))

;; ## Declare the system, then fit it
;;
;; Workload drives queue delay and latency; queue delay also drives latency.
;; Load and queue are centered operating measurements; latency is in ms.
;; The true equations are load = U₁, queue = load + U₂, and
;; latency = 100 + 20 queue + 30 load + 5 U₃, with independent standard
;; Gaussian noises. The graph and absence of hidden confounding are explicit
;; assumptions. A regression alone does not supply those assumptions.

(def graph
  {:directed [[:load :queue] [:load :latency] [:queue :latency]]})

(defn operating-data [n seed]
  (random/with-stream* seed ::operating-data
    #(mapv (fn [_]
             (let [load (dist/draw (dist/normal 0 1))
                   queue (+ load (dist/draw (dist/normal 0 1)))
                   latency (+ 100 (* 20 queue) (* 30 load) (dist/draw (dist/normal 0 5)))]
               {:load load :queue queue :latency latency})) (range n))))

(def training (operating-data 500 10))
(def validation (operating-data 250 20))
(def fitted (cm/fit graph training))

(into {} (map (fn [[v m]] [v (select-keys m [:intercept :coefficients :sigma])]) (:mechanisms fitted)))

;; ## Check held-out behavior
;;
;; Per-node conditional scores, 90% predictive coverage and residual checks.
;; Dependence tests look for noise related to parents or to other noise.
;; Their p-values use permutations and Holm multiple-testing correction.
;; This finite collection of checks can detect errors; passing does not
;; prove the graph, independence assumptions, or counterfactual coupling.

(def evaluation (cm/evaluate fitted validation {:seed 30 :permutations 399}))
(select-keys evaluation [:mechanisms :rejected-checks])

;; ## An effect, with sampling uncertainty
;;
;; Set queue delay one unit higher while leaving workload unchanged. The
;; true effect on latency is 20 ms. An unadjusted regression gives about
;; 35 ms: it mixes the queue effect with workload's direct effect.

(defn queue-effect [m]
  (let [u (zipmap (:order m) (repeat 0.0))]
    (- (:latency (cm/replay m u {:queue 1.0}))
       (:latency (cm/replay m u {:queue 0.0})))))

(def effect-uncertainty
  (cm/bootstrap fitted training queue-effect {:seed 40 :replicates 150}))

{:truth 20
 :adjusted (select-keys effect-uncertainty [:estimate :interval :level])
 :unadjusted (queue-effect (cm/fit {:directed [[:queue :latency]]} training))}

;; Bootstrap resamples IID rows and refits every mechanism. Its interval
;; measures fitting uncertainty within the chosen graph and mechanism
;; family. It does not account for a wrong graph, unobserved confounding,
;; dependent time-series observations, or an alternative noise coupling.

;; ## Reconstruct an incident and compare repairs
;;
;; For this incident we measured every node: load = 1, queue = 3,
;; latency = 200 ms. With fitted parameters fixed, abduction reconstructs
;; one standardized residual per mechanism. Replaying these residuals with
;; queue set to zero predicts roughly 140 ms, while preserving the incident's
;; workload and unexplained latency. There is no new outcome noise draw.

(def incident {:load 1.0 :queue 3.0 :latency 200.0})
(def abduction (cm/abduct fitted incident))

(defn resolve! [operation]
  (let [p (promise)]
    (operation #(deliver p [:ok %]) #(deliver p [:error %]))
    (let [[status value] (deref p 60000 [:timeout nil])]
      (if (= :ok status) value
          (throw (ex-info "Causal workflow operation failed" {:status status} value))))))

(def context (sp/create-execution-context))
(def executable (sp/with-context context (cm/model fitted)))

(def worlds
  (resolve! (cf/query executable
                      {:abductions [abduction]
                       :worlds [[:queue-normal {:queue {:do 0.0}}]
                                [:load-normal {:load {:do 0.0}}]]})))

(:worlds (first worlds))

(def repairs
  (resolve! (attribution/repairs executable [abduction]
                                 {:leave-as-is {}
                                  :queue-normal {:queue {:do 0.0}}
                                  :load-normal {:load {:do 0.0}}}
                                 #(< (:latency %) 145.0))))

repairs

;; Here success probabilities are 0 or 1 because every value is observed
;; and the fitted continuous mechanisms are invertible. Missing observations
;; or a posterior over parameters require multiple weighted abductions.
;; The bootstrap above is a different source of uncertainty.

;; ## Allocate the incident's excess latency
;;
;; Compared to a finite bank of normal operating observations, which
;; exogenous disturbances explain the increase? Shapley allocation shares
;; interactions rather than declaring a single winning 'root cause'. The
;; baseline population and latency score are part of the question.

(def baseline-rows (subvec training 0 16))
(def baseline-noise (mapv #(-> (cm/abduct fitted %) :noise) baseline-rows))

(def attribution-report
  (resolve! (attribution/attribute-noise executable abduction baseline-noise
                                         {:nodes (:order fitted) :score :latency})))

(select-keys attribution-report [:baseline :total :contributions :baseline-count])

;; The synchronous structural evaluator can also supply the same coalition
;; game to Shapley. This lets the ordinary row bootstrap refit and recompute
;; a contribution. Keep the incident and reference observations fixed;
;; reconstruct their noise under each refitted model.

(defn queue-contribution [m]
  (let [incident-noise (:noise (cm/abduct m incident))
        baselines (mapv #(:noise (cm/abduct m %)) baseline-rows)
        value (fn [coalition]
                (/ (reduce + (map (fn [baseline]
                                    (:latency (cm/replay m (merge baseline (select-keys incident-noise coalition)))))
                                  baselines))
                   (count baselines)))]
    (get-in (attribution/shapley (:order m) value) [:contributions :queue])))

(def attribution-uncertainty
  (cm/bootstrap fitted training queue-contribution {:seed 50 :replicates 100}))

(select-keys attribution-uncertainty [:estimate :interval :level])

;; These intervals condition on the particular reference bank, graph,
;; mechanism family and coupling. Noise attribution is not a probability
;; that a node is a cause, and the largest allocation need not name the
;; easiest intervention. Repairs act on endogenous values; the allocations
;; above concern exogenous disturbances.

;; ## A model that should fail a check
;;
;; Replace the queue mechanism by queue = 3 load² + small noise, then fit
;; a linear mechanism anyway. Linear correlation of residual with load can
;; be almost zero, while residual dependence on load² is strong. Held-out
;; checks include squared features specifically to expose this failure.

(defn nonlinear-data [seed]
  (random/with-stream* seed ::nonlinear-data
    #(mapv (fn [_]
             (let [load (dist/draw (dist/normal 0 1))]
               {:load load :queue (+ (* 3 load load) (dist/draw (dist/normal 0 0.2)))})) (range 180))))

(def misspecified (cm/fit {:directed [[:load :queue]]} (nonlinear-data 60)))
(def failure-report (cm/evaluate misspecified (nonlinear-data 70) {:seed 80 :permutations 399}))

(:rejected-checks failure-report)

;; A reversed linear Gaussian graph may pass the same observational checks
;; while giving a different intervention answer. Neither fit quality nor
;; bootstrap precision resolves that ambiguity. Graphical identification
;; (`causal/identify`) and explicit domain knowledge remain separate steps.
;;
;; See the [causal guide](https://github.com/replikativ/foerster/blob/main/doc/causal.md)
;; for custom fitting hooks, model contracts, and the boundaries of this workflow.
