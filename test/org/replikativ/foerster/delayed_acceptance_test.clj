(ns org.replikativ.foerster.delayed-acceptance-test
  "Two-stage delayed acceptance on a block: a cheap coarse density screens
  random-walk proposals and the program decides. The chain must target the
  program's exact posterior whatever the coarse density is; a poor one only
  lowers acceptance."
  (:require [clojure.test :refer [deftest is testing]]
            [org.replikativ.foerster.benchmark-test :as b]
            [org.replikativ.foerster.block :as block]
            [org.replikativ.foerster.core :as infer]
            [org.replikativ.foerster.diagnostics :as d]
            [org.replikativ.foerster.dist :as dist]
            [org.replikativ.foerster.effects :refer [observe sample]]
            [org.replikativ.foerster.kernel :as k]
            [org.replikativ.spindel.spin.cps :refer [spin]]))

;; y_i ~ N(a + b·x_i, σ = 0.5), (a, b) ~ N(0, I): a conjugate posterior
(def ^:private xs [-1.0 -0.6 -0.2 0.1 0.4 0.7 1.0 1.3])
(def ^:private ys [-1.9 -1.1 -0.6 0.3 0.6 1.4 1.9 2.6])
(def ^:private sigma 0.5)

(def ^:private exact
  ;; precision = I + XᵀX/σ², mean = precision⁻¹ Xᵀy/σ²
  (let [s2 (* sigma sigma)
        sxx (reduce + (map * xs xs)) sx (reduce + xs) n (count xs)
        p00 (+ 1.0 (/ n s2)) p01 (/ sx s2) p11 (+ 1.0 (/ sxx s2))
        r0 (/ (reduce + ys) s2) r1 (/ (reduce + (map * xs ys)) s2)
        det (- (* p00 p11) (* p01 p01))]
    {:a (/ (- (* p11 r0) (* p01 r1)) det)
     :b (/ (- (* p00 r1) (* p01 r0)) det)
     :a-sd (Math/sqrt (/ p11 det))
     :b-sd (Math/sqrt (/ p00 det))}))

(def ^:private prior-block
  ;; only the prior: the observations are the program's own sites, which a
  ;; delayed-acceptance move scores by replay
  (block/block {:block/id :line :block/target :complete-conditional
                :block/latents [{:name :a :shape []} {:name :b :shape []}]}
               {:log-density (fn [th _] (let [[a b] (vec th)] (* -0.5 (+ (* a a) (* b b)))))
                :value+grad (fn [th _] (let [[a b] (vec th)]
                                         [(* -0.5 (+ (* a a) (* b b))) (double-array [(- a) (- b)])]))}))

(defn- model []
  (spin (let [[a b] (sample (block/block-dist prior-block nil) :id :theta :init [0.0 0.0])]
          (loop [i 0]
            (when (< i (count xs))
              (observe (dist/normal (+ a (* b (nth xs i))) sigma) (nth ys i) :id [:y i])
              (recur (inc i))))
          {:a a :b b})))

(defn- gaussian-coarse
  "A diagonal Gaussian log density centred at `mu` with `sds`."
  [mu sds]
  (fn [th] (reduce + (map (fn [x m s] (let [z (/ (- x m) s)] (* -0.5 z z))) th mu sds))))

(defn- run [coarse]
  (b/run-infer 41 #(infer/kernel-infer (model)
                                       (k/delayed-acceptance-kernel 3000 {:theta coarse}
                                                                    {:scales 0.15 :samples :all :burn 300})
                                       4)))

(deftest the-chain-targets-the-program-not-the-coarse-density
  (testing "a coarse density that is off in place and width"
    (let [measure (run (gaussian-coarse [(+ (:a exact) 0.3) (- (:b exact) 0.2)]
                                        [(* 1.5 (:a-sd exact)) (* 0.7 (:b-sd exact))]))
          a (d/summary measure :a)
          bs (d/summary measure :b)]
      (is (< (:rhat a) 1.05) (str a))
      (is (< (Math/abs (- (:mean a) (:a exact))) (* 0.2 (:a-sd exact))) (str a " vs " exact))
      (is (< (Math/abs (- (:mean bs) (:b exact))) (* 0.2 (:b-sd exact))) (str bs " vs " exact))
      (is (< (Math/abs (- (:sd a) (:a-sd exact))) (* 0.15 (:a-sd exact))) "the program's spread, not the coarse one"))))

(def ^:private pair-block
  (block/block {:block/id :pair :block/target :complete-conditional
                :block/latents [{:name :u :shape []} {:name :v :shape []}]}
               {:log-density (fn [th _] (let [[u v] (vec th)] (* -0.5 (+ (* u u) (* v v)))))
                :value+grad (fn [th _] (let [[u v] (vec th)]
                                         [(* -0.5 (+ (* u u) (* v v))) (double-array [(- u) (- v)])]))
                ;; a replay that brings :y back draws it afresh
                :sample (fn [_] [(dist/draw (dist/normal 0.0 1.0)) (dist/draw (dist/normal 0.0 1.0))])
                :sample-log-density (fn [th _] (let [[u v] (vec th)] (* -0.5 (+ (* u u) (* v v)))))}))

(deftest a-block-a-move-removed-is-skipped
  ;; :y exists only while x's first coordinate is positive: a move of :x
  ;; that makes it negative takes :y out of the path
  (let [coarse (fn [th] (let [[u v] th] (* -0.5 (+ (* u u) (* v v)))))
        model #(spin (let [[x0] (sample (block/block-dist pair-block nil) :id :x :init [0.5 0.0])]
                       (when (pos? x0)
                         (sample (block/block-dist pair-block nil) :id :y :init [0.0 0.0]))
                       x0))
        measure (b/run-infer 5 #(infer/kernel-infer (model)
                                                    (k/delayed-acceptance-kernel 200 {:x coarse :y coarse}
                                                                                 {:scales 1.0 :samples :all})
                                                    2))]
    (is (some neg? (map (comp org.replikativ.foerster.measure/get-value first)
                        (org.replikativ.foerster.measure/get-particles measure)))
        "x crossed zero, so :y came and went")))

(deftest invalid-step-counts-are-refused
  (is (thrown? AssertionError (k/delayed-acceptance-kernel 10 {:theta identity} {:steps -1})))
  (is (thrown? AssertionError (k/delayed-acceptance-kernel 10 {:theta identity} {:steps 0.5}))))

(deftest a-block-an-earlier-move-creates-is-moved-in-its-turn
  ;; no observations, so the posterior is the prior: P(x₀ > 0) = ½ and :y,
  ;; when it exists, is N(0, I). Which blocks move must not depend on the
  ;; state before the iteration's first move (a fixed order, each block
  ;; moved if it exists in its turn); the bias a state-dependent schedule
  ;; causes here is below what this sample size can see, so this is a
  ;; sanity check of the conditional-block path, not a detector of it.
  (let [coarse (fn [th] (let [[u v] th] (* -0.5 (+ (* u u) (* v v)))))
        model #(spin (let [[x0] (sample (block/block-dist pair-block nil) :id :x :init [0.5 0.0])
                           y (when (pos? x0) (sample (block/block-dist pair-block nil) :id :y :init [0.0 0.0]))]
                       {:x0 x0 :y0 (first y)}))
        measure (b/run-infer 9 #(infer/kernel-infer (model)
                                                    (k/delayed-acceptance-kernel 20000 {:x coarse :y coarse}
                                                                                 {:scales 1.2 :samples :all :burn 500})
                                                    4))
        vs (map (comp org.replikativ.foerster.measure/get-value first)
                (org.replikativ.foerster.measure/get-particles measure))
        n (double (count vs))
        ys (keep :y0 vs)
        p (/ (count (filter #(pos? (:x0 %)) vs)) n)
        y2 (/ (reduce + (map #(* % %) ys)) (count ys))]
    (is (< (Math/abs (- p 0.5)) 0.02) (str "P(x0 > 0) " p))
    (is (< (Math/abs (- y2 1.0)) 0.05) (str "E[y0²] " y2))))

(deftest a-coarse-density-without-the-posterior-s-support-is-refused
  ;; a half-line surrogate at a state outside it would freeze the chain
  (let [half-line (fn [[a]] (if (pos? a) 0.0 ##-Inf))
        error (try (run half-line) nil (catch Exception e e))]
    (is (some #(= :org.replikativ.foerster.block-mh/coarse-support (:type (ex-data %)))
              (take-while some? (iterate ex-cause error)))
        (str error))))
