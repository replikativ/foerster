(ns org.replikativ.foerster.resample-move-test
  "Resample-move SMC: after resampling, particles move by MH replays from
  their anchors up to the observation they are parked at. The posterior stays
  exact, degeneracy falls, a lag window bounds the live anchors, and seeded
  runs reproduce."
  (:require [clojure.test :refer [deftest is testing]]
            [org.replikativ.foerster.benchmark-test :as b]
            [org.replikativ.foerster.dist :as dist]
            [org.replikativ.foerster.effects :refer [observe sample]]
            [org.replikativ.foerster.measure :as m]
            [org.replikativ.foerster.smc :as smc]
            [org.replikativ.foerster.trace :as itrace]
            [org.replikativ.spindel.effects.await :as aw]
            [org.replikativ.spindel.effects.savepoint :as sp]
            [org.replikativ.spindel.world.scope :as world-scope]
            [org.replikativ.spindel.engine.context :as context]
            [org.replikativ.spindel.engine.core :as ec]
            [org.replikativ.spindel.engine.executor :as executor]
            [org.replikativ.spindel.spin.cps :refer [spin]]
            [org.replikativ.foerster.random :as random]))

;; μ ~ N(0,1), y_i ~ N(μ,1), 50 observations: μ | y ~ N(Σy/51, 1/51)
(def ^:private ys
  (let [r (java.util.Random. 3)] (vec (repeatedly 50 #(+ 0.7 (.nextGaussian r))))))

(def ^:private truth [(/ (reduce + ys) 51.0) (Math/sqrt (/ 1.0 51.0))])

(defn- static-model []
  (spin (let [mu (sample (dist/normal 0.0 1.0) :id :mu)]
          (loop [i 0]
            (when (< i (count ys))
              (observe (dist/normal mu 1.0) (nth ys i) :id [:y i])
              (recur (inc i))))
          mu)))

(deftest a-static-parameter-stays-diverse-and-exact
  (let [plain (b/weighted-values (b/run-infer 5 #(smc/smc (static-model) 200 {})))
        moved (b/run-infer 5 #(smc/smc (static-model) 200
                                       {:anchors #{:mu}
                                        :rejuvenate {:moves 2 :propose (itrace/random-walk-proposal 0.2)}}))
        wv (b/weighted-values moved)]
    (is (< (b/kl-normal (b/w-mean-sd identity wv) truth) 0.03)
        (str (b/w-mean-sd identity wv) " vs " truth))
    (is (> (count (distinct (map first wv))) (* 2 (count (distinct (map first plain)))))
        "moves undo the resampling's duplicates")
    (let [{:keys [moves accepted]} (:rejuvenation moved)]
      (is (pos? moves))
      (is (< 0 accepted moves)))))

(defn- hmm [] (apply b/hmm-model b/hmm-args))

(deftest moving-every-state-lowers-the-hmm-error
  ;; every state anchored: moves rejuvenate the degenerate past, which the
  ;; smoothing marginals the error measures depend on
  (let [plain (b/hmm-error (b/weighted-values (b/run-infer 11 #(smc/smc (hmm) 300 {:resample-threshold 1.0}))))
        moved (b/hmm-error (b/weighted-values
                            (b/run-infer 11 #(smc/smc (hmm) 300 {:resample-threshold 1.0
                                                                 :anchors :all
                                                                 :rejuvenate {:moves 2}}))))]
    (is (< moved 0.045) (str "rms " moved))
    (is (< moved plain) (str moved " vs plain " plain))))

(deftest a-lag-window-bounds-the-live-anchors
  (let [n 100
        measure (b/run-infer 12 #(smc/smc (hmm) n {:resample-threshold 1.0
                                                   :anchors {:lag 2}
                                                   :rejuvenate {:moves 1}}))
        {:keys [max-anchors moves]} (:rejuvenation measure)]
    ;; one latent state per step: the window holds at most lag + 1 per particle
    (is (<= max-anchors (* n 3)) (str max-anchors))
    (is (= (* n 16) moves) "one move per particle at each of the 16 resamplings")))

(deftest seeded-resample-move-reproduces
  (let [run (fn [] (let [exec (executor/thread-pool-executor {:threads 4})]
                     (try (b/weighted-values
                           (b/run-infer 13 #(smc/smc (static-model) 50
                                                     {:executor exec :anchors :all :rejuvenate {:moves 1}})))
                          (finally (.close ^java.lang.AutoCloseable exec)))))]
    (is (= (run) (run)))))

(deftest options-are-checked
  (let [root (context/create-execution-context)
        model (binding [ec/*execution-context* root] (static-model))
        refusal (fn [opts] (try ((smc/smc model 10 opts) (fn [_]) (fn [_])) nil
                                (catch clojure.lang.ExceptionInfo e (:type (ex-data e)))))]
    (is (= ::smc/rejuvenate-without-anchors (refusal {:rejuvenate {:moves 1}})))
    (is (= ::smc/opaque-policy (refusal {:anchors :all :rejuvenate {} :policy (fn [_ _] nil)})))
    (is (= ::smc/invalid-anchors (refusal {:anchors 3})))
    (is (= ::smc/rejuvenate-with-retained
           (refusal {:anchors :all :rejuvenate {} :retained {:mu 0.0}})))))

(deftest a-move-may-change-how-many-observations-there-are
  ;; b = true adds an observation: a move flipping b at the second barrier
  ;; would end the program before reaching it, and is rejected
  (let [measure (b/run-infer 14 #(smc/smc (b/varlen-model) 500
                                          {:resample-threshold 1.0 :anchors :all :rejuvenate {:moves 2}}))
        p (b/w-mean #(if % 1.0 0.0) (b/weighted-values measure))]
    (is (< (Math/abs (- p b/varlen-truth)) 0.06) (str p " vs " b/varlen-truth))
    (is (pos? (:accepted (:rejuvenation measure))))))

(defn- live-worlds [root]
  (count (world-scope/handles @(:scope (sp/session root)))))

(deftest moves-give-their-worlds-back
  ;; a displaced or rejected particle's world is released at once, not when
  ;; the session closes: the live worlds stay near N·(lag + 1)
  (random/set-seed! 15)
  (let [n 100
        root (context/create-execution-context)
        live (promise)]
    (binding [ec/*execution-context* root]
      ((smc/smc (hmm) n {:root root :resample-threshold 1.0
                         :anchors {:lag 2} :rejuvenate {:moves 3}})
       (fn [_] (deliver live (live-worlds root)))
       (fn [e] (deliver live e))))
    (let [worlds (deref live 120000 ::timeout)]
      (is (and (number? worlds) (<= worlds (* 5 n))) (str worlds " live worlds")))))

(deftest seeded-moves-reproduce-on-a-thread-pool
  ;; latents in awaited sub-spins run concurrently; siblings share anchors,
  ;; and every replay names its seed
  (let [ys (vec (take 12 ys))
        step (fn [t x] (spin (sample (dist/normal x 1.0) :id [:x t])))
        model (fn []
                (spin (loop [t 0 x 0.0]
                        (if (= t (count ys))
                          x
                          (let [x' (aw/await (step t x))]
                            (observe (dist/normal x' 1.0) (nth ys t) :id [:y t])
                            (recur (inc t) x'))))))
        run (fn [] (let [exec (executor/thread-pool-executor {:threads 4})]
                     (try (b/weighted-values
                           (b/run-infer 16 #(smc/smc (model) 60
                                                     {:executor exec :resample-threshold 1.0
                                                      :anchors :all :rejuvenate {:moves 3}})))
                          (finally (.close ^java.lang.AutoCloseable exec)))))]
    (is (apply = (repeatedly 3 run)))))

;; --- streaming ---------------------------------------------------------------

(defn- await-cps [operation]
  (let [result (promise)]
    (operation #(deliver result [:ok %]) #(deliver result [:error %]))
    (let [[k v] (deref result 120000 [:error (ex-info "timed out" {})])]
      (if (= :ok k) v (throw v)))))

(def ^:private stream-ys [0.8 1.9 1.2 3.0 2.5])

(defn- random-walk [root]
  (binding [ec/*execution-context* root]
    (spin
     (loop [t 0 x (sample (dist/normal 0.0 1.0) :id [:x 0])]
       (sample (dist/normal x 1.0) :id [:y t] :stream true)
       (if (= t (dec (count stream-ys)))
         x
         (recur (inc t) (sample (dist/normal x 1.0) :id [:x (inc t)])))))))

(deftest streaming-with-a-lag-tracks-the-kalman-filter
  ;; the stream data stay fixed in every move; the last state follows the
  ;; Kalman filter
  (random/set-seed! 6)
  (let [root (context/create-execution-context)
        step0 (await-cps (smc/stream (random-walk root) 2000
                                     {:resample-threshold 1.0 :anchors {:lag 1} :rejuvenate {:moves 1}
                                      :executor (executor/thread-pool-executor {:threads 1})}))
        final (reduce (fn [step y] (await-cps ((:push step) y))) step0 stream-ys)
        ;; Kalman filtering of the last state: the model's last x is x_4
        [kmean kvar] (loop [[y & more] stream-ys pm 0.0 pv 1.0 out nil]
                       (if-not y
                         out
                         (let [k (/ pv (+ pv 1.0)) mean (+ pm (* k (- y pm))) var (* (- 1.0 k) pv)]
                           (recur more mean (+ var 1.0) [mean var]))))
        ps (m/get-particles (:measure final))
        ws (m/normalize-log-weights (mapv second ps))
        xs (mapv (comp m/get-value first) ps)
        mean (reduce + (map * ws xs))
        var (reduce + (map (fn [w x] (* w (let [d (- x mean)] (* d d)))) ws xs))]
    (is (:done? final))
    (is (< (Math/abs (- mean kmean)) 0.1) (str mean " vs " kmean))
    (is (< (Math/abs (- var kvar)) 0.08) (str var " vs " kvar))
    (is (every? (fn [[s _]] (= stream-ys (mapv #(:value (get (m/get-trace s) [:y %])) (range 5)))) ps)
        "no move changed the data")))
