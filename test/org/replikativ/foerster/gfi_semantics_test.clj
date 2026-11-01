(ns org.replikativ.foerster.gfi-semantics-test
  "Regression tests for the trace-operation semantics a review found wrong:
  kept values that leave their support, the densities of sites a replay draws
  afresh or drops, constrained sites across replays, and constraints the run
  never reaches. Each is checked against an exact answer."
  (:require [clojure.test :refer [deftest is testing]]
            [org.replikativ.foerster.gfi :as gfi]
            [org.replikativ.foerster.involutive :as inv]
            [org.replikativ.foerster.trace :as itrace]
            [org.replikativ.foerster.effects :refer [sample observe]]
            [org.replikativ.spindel.engine.context :as context]
            [org.replikativ.spindel.engine.core :as ec]
            [org.replikativ.spindel.spin.cps :refer [spin]]
            [org.replikativ.foerster.dist :as dist]
            [org.replikativ.foerster.random :as random]))

(defn- await-cps [operation]
  (let [result (promise)]
    (operation #(deliver result [:ok %]) #(deliver result [:error %]))
    (let [outcome (deref result 60000 ::timeout)]
      (when (= ::timeout outcome)
        (throw (ex-info "CPS operation timed out" {})))
      (if (= :ok (first outcome)) (second outcome) (throw (second outcome))))))

(defmacro ^:private with-root [[root] & body]
  `(let [~root (context/create-execution-context)]
     (try
       ~@body
       (finally
         (context/close-context! ~root)))))

(defn- lp [d v] (dist/logpdf d v))

(defn- mean [xs] (/ (reduce + xs) (double (count xs))))

;; -----------------------------------------------------------------------------
;; Models

(defn- nested-support
  "x ~ Exp(1), z | x ~ U(0, x): no data, so the target is the prior,
  E[x] = 1 and E[z] = 1/2. Moving x below z leaves z outside its support."
  [root]
  (binding [ec/*execution-context* root]
    (spin (let [x (sample (dist/exponential 1.0) :id :x)
                z (sample (dist/uniform 0.0 x) :id :z)]
            [x z]))))

(defn- hierarchical
  "x ~ N(0,1), z | x ~ N(x,1), y = 2 | z ~ N(z,1)."
  [root]
  (binding [ec/*execution-context* root]
    (spin (let [x (sample (dist/normal 0.0 1.0) :id :x)
                z (sample (dist/normal x 1.0) :id :z)]
            (observe (dist/normal z 1.0) 2.0 :id :y)
            x))))

(defn- switch
  "Which latent exists depends on b."
  [root]
  (binding [ec/*execution-context* root]
    (spin (let [b (sample (dist/flip 0.5) :id :b)
                v (if b
                    (sample (dist/normal 0.0 1.0) :id :v0)
                    (sample (dist/normal 5.0 1.0) :id :v1))]
            (observe (dist/normal v 1.0) 5.0 :id :y)
            b))))

(def ^:private y-switch 2.0)

(defn- switch-with-proposal
  "b ~ flip(1/2); v0 ~ N(0,1) drawn from its site proposal N(3,1) when b,
  else v1 ~ N(5,1); y = 2 ~ N(v,1). P(b | y) = N(2;0,√2) / (N(2;0,√2) + N(2;5,√2))."
  [root]
  (binding [ec/*execution-context* root]
    (spin (let [b (sample (dist/flip 0.5) :id :b)
                v (if b
                    (sample (dist/normal 0.0 1.0) :id :v0 :proposal (dist/normal 3.0 1.0))
                    (sample (dist/normal 5.0 1.0) :id :v1))]
            (observe (dist/normal v 1.0) y-switch :id :y)
            b))))

(def ^:private exact-p-b
  (let [a (Math/exp (lp (dist/normal 0.0 (Math/sqrt 2.0)) y-switch))
        b (Math/exp (lp (dist/normal 5.0 (Math/sqrt 2.0)) y-switch))]
    (/ a (+ a b))))

(defn- v-site [t] (if (contains? (:trace/entries t) :v0) :v0 :v1))

(defn- walk-v
  "A random-walk MH step on whichever of v0, v1 the trace has."
  [t i]
  (let [a (v-site t)]
    (itrace/mh-step t {:iteration i
                       :select (fn [_ _] {:targets #{a} :log-selection (constantly 0.0)})
                       :propose (itrace/random-walk-proposal 0.7)})))

(defn- p-b-chain
  "Alternate `flip-step` with a random walk on v; the fraction of b = true
  after burn-in."
  [root flip-step n]
  (loop [t (await-cps (gfi/simulate (switch-with-proposal root))) i 0 bs []]
    (if (= i n)
      (do (await-cps (gfi/close! t)) (mean bs))
      (let [t1 (:trace (await-cps (flip-step t i)))
            t2 (:trace (await-cps (walk-v t1 (* 2 i))))]
        (recur t2 (inc i) (if (>= i 500) (conj bs (if (:trace/result t2) 1.0 0.0)) bs))))))

;; -----------------------------------------------------------------------------
;; A kept value that leaves its support

(deftest update-keeps-a-value-outside-its-new-support
  (with-root [root]
    (let [{t :trace} (await-cps (gfi/generate (nested-support root) {:x 2.0 :z 1.5}))
          {t' :trace w :weight d :discard} (await-cps (gfi/update t {:x 1.0}))]
      (testing "Gen's update keeps z: the new trace has density zero"
        (is (= 1.5 (get (itrace/choices t') :z)))
        (is (= ##-Inf w)))
      (testing "only the overwritten constraint is discarded"
        (is (= {:x 2.0} d)))
      (await-cps (gfi/close! t)))))

(deftest mh-proposal-samples-a-target-with-nested-supports
  ;; an independent Exp(1) proposal for x, alternated with regenerating z;
  ;; the target is the prior: E[x] = 1, E[z] = 1/2
  (random/set-seed! 5)
  (with-root [root]
    (let [proposal (fn [_choices]
                     (binding [ec/*execution-context* root]
                       (spin (sample (dist/exponential 1.0) :id :x))))
          samples (loop [t (await-cps (gfi/simulate (nested-support root))) i 0 acc []]
                    (if (= i 4000)
                      (do (await-cps (gfi/close! t)) acc)
                      (let [t1 (:trace (await-cps (gfi/mh-proposal t proposal)))
                            t2 (:trace (await-cps (gfi/mh t1 #{:z})))]
                        (recur t2 (inc i) (if (>= i 500) (conj acc (:trace/result t2)) acc)))))
          ex (mean (map first samples))
          ez (mean (map second samples))]
      ;; the review's probe measured E[x] ≈ 0.25 before the fix
      (is (< (Math/abs (- ex 1.0)) 0.15) (str "E[x] " ex))
      (is (< (Math/abs (- ez 0.5)) 0.1) (str "E[z] " ez)))))

;; -----------------------------------------------------------------------------
;; Densities of sites drawn afresh or dropped

(def ^:private flip-move
  {:propose (fn [_] {:aux nil :log-q 0.0})
   :log-q (fn [_ _] 0.0)
   :involution (fn [{b :b} _] {:choices {:b (not b)} :aux nil :log-jacobian 0.0})})

(deftest involutive-moves-score-fresh-sites-by-their-draw-density
  (random/set-seed! 11)
  (with-root [root]
    (let [p (p-b-chain root (fn [t _] (inv/step t flip-move)) 12000)]
      (is (< (Math/abs (- p exact-p-b)) 0.04)
          (str "P(b) " p " exact " exact-p-b)))))

(deftest mh-proposal-scores-regenerated-sites-by-their-draw-density
  ;; the proposal flips b only; the reverse move redraws the dropped v site
  ;; from its own draw law (v0's site proposal), not its prior
  (random/set-seed! 13)
  (with-root [root]
    (let [flip-b (fn [{b :b}]
                   (binding [ec/*execution-context* root]
                     (spin (sample (dist/flip (if b 0.0 1.0)) :id :b))))
          p (p-b-chain root (fn [t _] (gfi/mh-proposal t flip-b)) 12000)]
      (is (< (Math/abs (- p exact-p-b)) 0.04)
          (str "P(b) " p " exact " exact-p-b)))))

(deftest involutive-moves-can-check-the-involution
  (with-root [root]
    (let [t (await-cps (gfi/simulate (switch-with-proposal root)))
          not-an-involution {:propose (fn [_] {:aux 1.0 :log-q 0.0})
                             :log-q (fn [_ _] 0.0)
                             :involution (fn [{b :b} u] {:choices {:b (not b)} :aux (inc u) :log-jacobian 0.0})}]
      (is (thrown-with-msg? clojure.lang.ExceptionInfo #"not an involution"
                            (await-cps (inv/step t (assoc not-an-involution :check? true)))))
      (is (map? (await-cps (inv/step t (assoc flip-move :check? true)))))
      (await-cps (gfi/close! t)))))

;; -----------------------------------------------------------------------------
;; Constrained sites stay constrained

(deftest constrained-sites-stay-constrained-across-replays
  (with-root [root]
    (testing "update"
      (let [{t :trace} (await-cps (gfi/generate (hierarchical root) {:x 0.5 :z 1.5}))
            {t' :trace} (await-cps (gfi/update t {:x -0.5}))]
        (is (= [] (itrace/latent-addresses t')))
        (is (:constrained? (:note (get-in t' [:trace/entries :z]))))
        (await-cps (gfi/close! t))))
    (testing "an MH move upstream of a constrained site"
      (let [{t :trace} (await-cps (gfi/generate (hierarchical root) {:z 1.5}))
            {t' :trace} (await-cps (gfi/regenerate t #{:x}))]
        (is (= [:x] (itrace/latent-addresses t')))
        (is (= 1.5 (get (itrace/choices t') :z)))
        (await-cps (gfi/close! t))))))

;; -----------------------------------------------------------------------------
;; Constraints the run never reaches

(deftest constraints-must-be-visited
  (with-root [root]
    (testing "generate"
      (is (= :org.replikativ.foerster.gfi/unvisited-constraints
             (:type (ex-data (try (await-cps (gfi/generate (switch root) {:b true :v1 3.0}))
                                  (catch Exception e e))))))
      (let [{t :trace} (await-cps (gfi/generate (switch root) {:b true :v1 3.0}
                                                {:allow-unvisited? true}))]
        (is (not (contains? (itrace/choices t) :v1)))
        (await-cps (gfi/close! t))))
    (testing "update"
      (let [{t :trace} (await-cps (gfi/generate (switch root) {:b true :v0 0.3}))]
        (is (= :org.replikativ.foerster.gfi/unvisited-constraints
               (:type (ex-data (try (await-cps (gfi/update t {:v1 3.0}))
                                    (catch Exception e e))))))
        (is (map? (await-cps (gfi/update t {:v1 3.0} {:allow-unvisited? true}))))
        (is (= {:b false :v1 3.0}
               (select-keys (itrace/choices (:trace (await-cps (gfi/update t {:b false :v1 3.0}))))
                            [:b :v0 :v1]))
            "a constraint reached through the changed branch is fine")
        (await-cps (gfi/close! t))))))

;; -----------------------------------------------------------------------------
;; Second review: update's values are values, model errors, involution checks

(deftest updated-latent-sites-stay-latent
  (with-root [root]
    (let [t (await-cps (gfi/simulate (hierarchical root)))
          {t' :trace w :weight} (await-cps (gfi/update t {:x 1.0}))
          z (get (itrace/choices t) :z)]
      (is (= #{:x :z} (set (itrace/latent-addresses t'))) "a new value is not new conditioning")
      (let [x0 (get (itrace/choices t) :x)
            expected (- (+ (lp (dist/normal 0.0 1.0) 1.0) (lp (dist/normal 1.0 1.0) z))
                        (+ (lp (dist/normal 0.0 1.0) x0) (lp (dist/normal x0 1.0) z)))]
        (is (< (Math/abs (- w expected)) 1e-9) "nothing drawn afresh: the weight is the joint ratio"))
      (let [{t2 :trace} (await-cps (gfi/regenerate t' #{:x}))]
        (is (not= 1.0 (get (itrace/choices t2) :x)) "a later move changes it again"))
      (await-cps (gfi/close! t)))))

(deftest update-keeps-a-latent-value-outside-its-new-support
  (random/set-seed! 21)
  (with-root [root]
    (let [t (loop [i 0]
              (let [t (await-cps (gfi/simulate (nested-support root)))
                    [x z] (:trace/result t)]
                (if (or (> z 1.0) (> i 500)) t (do (await-cps (gfi/close! t)) (recur (inc i))))))
          z (get (itrace/choices t) :z)
          {t' :trace w :weight} (await-cps (gfi/update t {:x (/ z 2.0)}))]
      (is (> z 1.0))
      (is (= z (get (itrace/choices t') :z)) "z is kept")
      (is (= ##-Inf w))
      (is (= #{:x :z} (set (itrace/latent-addresses t'))))
      (await-cps (gfi/close! t)))))

(deftest update-reports-a-failing-replay
  (with-root [root]
    (let [model (binding [ec/*execution-context* root]
                  (spin (let [x (sample (dist/normal 0.0 1.0) :id :x)]
                          (when (> x 5.0) (throw (ex-info "too large" {})))
                          (sample (dist/normal x 1.0) :id :z))))
          t (await-cps (gfi/simulate model))
          r (await-cps (gfi/update t {:x 10.0 :z 1.0}))]
      (is (= ##-Inf (:weight r)))
      (is (= "too large" (ex-message (:error r))) "the program's error, not unvisited constraints")
      (await-cps (gfi/close! t)))))

(deftest involutive-moves-count-reverse-redraws
  ;; z ~ U(x, x + 1/2): shifting x by 2 redraws z from its new support; the
  ;; reverse redraws it back, so the ratio is the prior ratio of x alone
  (with-root [root]
    (let [model (binding [ec/*execution-context* root]
                  (spin (let [x (sample (dist/normal 0.0 1.0) :id :x)
                              z (sample (dist/uniform x (+ x 0.5)) :id :z)]
                          [x z])))
          shift {:propose (fn [_] {:aux 2.0 :log-q 0.0})
                 :log-q (fn [_ _] 0.0)
                 :involution (fn [{x :x} u] {:choices {:x (+ x u)} :aux (- u) :log-jacobian 0.0})}]
      (dotimes [_ 5]
        (let [t (await-cps (gfi/simulate model))
              x (get (itrace/choices t) :x)
              {r :log-ratio} (await-cps (inv/step t shift))]
          (is (< (Math/abs (- r (- (lp (dist/normal 0.0 1.0) (+ x 2.0)) (lp (dist/normal 0.0 1.0) x)))) 1e-9))
          (await-cps (gfi/close! t)))))))

(deftest the-involution-check-accepts-full-choice-maps
  (with-root [root]
    (let [t (await-cps (gfi/simulate (switch-with-proposal root)))
          full {:propose (fn [_] {:aux nil :log-q 0.0})
                :log-q (fn [_ _] 0.0)
                :involution (fn [x _] {:choices (update x :b not) :aux nil :log-jacobian 0.0})}]
      (is (map? (await-cps (inv/step t (assoc full :check? true)))))
      (await-cps (gfi/close! t)))))

(deftest involutions-cannot-change-observations
  (with-root [root]
    (let [t (await-cps (gfi/simulate (hierarchical root)))
          move {:propose (fn [_] {:aux nil :log-q 0.0})
                :log-q (fn [_ _] 0.0)
                :involution (fn [_ _] {:choices {:y 0.0} :aux nil :log-jacobian 0.0})}]
      (is (= ::inv/observed-site
             (:type (ex-data (try (await-cps (inv/step t move)) (catch Exception e e))))))
      (await-cps (gfi/close! t)))))
