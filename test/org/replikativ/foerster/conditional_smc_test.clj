(ns org.replikativ.foerster.conditional-smc-test
  "Particle Gibbs keeps the caller's policy for the retained particle."
  (:require [clojure.test :refer [deftest is]]
            [org.replikativ.foerster.benchmark-test :as b]
            [org.replikativ.foerster.core :as infer]
            [org.replikativ.foerster.diagnostics :as d]
            [org.replikativ.foerster.dist :as dist]
            [org.replikativ.foerster.effects :refer [observe sample]]
            [org.replikativ.foerster.measure :as m]
            [org.replikativ.foerster.trace :as itrace]
            [org.replikativ.spindel.spin.cps :refer [spin]]))

(defn- model []
  ;; x ~ Bernoulli(½), y ~ Flip(x ? 0.9 : 0.1); conditioned on y = true from
  ;; outside, P(x = 1) = 0.9
  (spin
   (let [x (sample (dist/bernoulli 0.5) :id :x)]
     (sample (dist/flip (if (= 1 x) 0.9 0.1)) :id :y)
     x)))

(deftest the-retained-particle-is-conditioned-like-the-others
  (let [measure (b/run-infer 21 #(infer/infer (model) {:method :pgibbs :particles 2 :iterations 4000
                                                       :policy (itrace/policy {:constraints {:y true}})}))
        {:keys [mean]} (d/summary measure identity)]
    (is (< (Math/abs (- mean 0.9)) 0.03) (str "P(x = 1) " mean))))

(defn- guided-branch []
  ;; k ~ Bernoulli(½); with k = 1 a site b ~ Flip(½) that a replay draws
  ;; from its guide Flip(0.9): the guide must not shift the posterior
  (spin
   (let [k (sample (dist/bernoulli 0.5) :id :k)]
     [k (when (= 1 k) (sample (dist/flip 0.5) :id :b :proposal (dist/flip 0.9)))])))

(deftest mh-scores-a-guided-site-it-drops-under-its-guide
  (let [measure (b/run-infer 22 #(infer/infer (guided-branch) {:method :mh :iterations 6000 :chains 2 :burn 500}))
        vs (map (comp m/get-value first) (m/get-particles measure))
        born (filter #(= 1 (first %)) vs)
        pk (/ (count born) (double (count vs)))
        pb (/ (count (filter second born)) (double (count born)))]
    (is (< (Math/abs (- pk 0.5)) 0.04) (str "P(k = 1) " pk))
    (is (< (Math/abs (- pb 0.5)) 0.04) (str "P(b | k = 1) " pb))))

(deftest particle-gibbs-with-moves-recovers-the-posterior
  ;; the retained particle moves too (`smc/retained-stages`); its exactness is
  ;; checked by enumeration in `sweep-enumeration-test`, this is the API path
  (let [measure (b/run-infer 23 #(infer/infer (model) {:method :pgibbs :particles 3 :iterations 3000
                                                       :anchors :all :rejuvenate {:moves 1}
                                                       :policy (itrace/policy {:constraints {:y true}})}))
        {:keys [mean]} (d/summary measure identity)]
    (is (< (Math/abs (- mean 0.9)) 0.03) (str "P(x = 1) " mean))))

(defn- chain-hmm
  "x1 ~ flip(½) → y1; x2 ~ flip(x1 ? 0.1 : 0.9) → y2; x3 ~ flip(x2 ? 0.1 :
  0.9) → y3, every y observed true."
  []
  (spin
   (let [x1 (sample (dist/flip 0.5) :id :x1)
         _ (observe (dist/flip (if x1 0.8 0.2)) true :id :y1)
         x2 (sample (dist/flip (if x1 0.1 0.9)) :id :x2)
         _ (observe (dist/flip (if x2 0.95 0.05)) true :id :y2)
         x3 (sample (dist/flip (if x2 0.1 0.9)) :id :x3)]
     (observe (dist/flip (if x3 0.95 0.05)) true :id :y3)
     [x1 x2 x3])))

(defn- law [measure]
  (let [ps (m/get-particles measure)]
    (reduce (fn [acc [[s _] w]] (update acc (m/get-value s) (fnil + 0.0) w))
            {} (map vector ps (m/normalize-log-weights (mapv second ps))))))

(deftest ancestor-weights-score-the-future-under-the-caller-s-policy
  ;; do(x2 = true) cuts x1 off from x2: P(x1 | y) = 0.8. Particles score 0
  ;; at x2; ancestor weights that scored the retained future under the bare
  ;; retained values added log p(x2 = true | x1) and drew x1 = false far too
  ;; often (P(x1) ≈ 0.5). Over six seeds the fixed chain gave 0.785–0.821 and
  ;; TV 0.013–0.078.
  (let [policy (itrace/policy {:interventions {:x2 true}})
        exact (law (b/run-infer 1 #(infer/infer (chain-hmm) {:method :enumerate :policy policy})))
        chain (law (b/run-infer 5 #(infer/pgas-infer (chain-hmm) 2 1000 {:policy policy :seed 5})))
        p-x1 (fn [law] (reduce + (map (fn [[xs w]] (if (first xs) w 0.0)) law)))
        tv (* 0.5 (reduce + (map #(Math/abs (- (get chain % 0.0) (get exact % 0.0)))
                                 (distinct (concat (keys chain) (keys exact))))))]
    (is (< (Math/abs (- (p-x1 exact) 0.8)) 1e-12) "the reference")
    (is (< (Math/abs (- (p-x1 chain) 0.8)) 0.06) (str "P(x1) " (p-x1 chain)))
    (is (< tv 0.12) (str "TV " tv))))
