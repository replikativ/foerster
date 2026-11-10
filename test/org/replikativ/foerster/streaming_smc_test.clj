(ns org.replikativ.foerster.streaming-smc-test
  "Online SMC against the Kalman filter: a random walk x_t = x_{t-1} + N(0,1)
  observed as y_t ~ N(x_t, 1), its observations pushed one at a time. After
  each push the particle filter's x_t must match the Kalman filtering
  distribution, and its evidence the Kalman marginal likelihood."
  (:require [clojure.test :refer [deftest is testing]]
            [org.replikativ.foerster.inference.smc :as smc]
            [org.replikativ.foerster.inference.particle-mcmc :as pmcmc]
            [org.replikativ.foerster.runtime.trace :as itrace]
            [org.replikativ.foerster.measure :as m]
            [org.replikativ.foerster.weights :as weights]
            [org.replikativ.foerster.effects :refer [sample]]
            [org.replikativ.spindel.engine.context :as context]
            [org.replikativ.spindel.engine.core :as ec]
            [org.replikativ.spindel.engine.executor :as executor]
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

(def ^:private ys [0.8 1.9 1.2 3.0 2.5])

(defn- random-walk [root]
  (binding [ec/*execution-context* root]
    (spin
     (loop [t 0 x (sample (dist/normal 0.0 1.0) :id [:x 0])]
       (sample (dist/normal x 1.0) :id [:y t] :stream true)
       (if (= t (dec (count ys)))
         x
         (recur (inc t) (sample (dist/normal x 1.0) :id [:x (inc t)])))))))

(defn- kalman
  "[mean var log-likelihood] of the filtering distribution after each y."
  [ys]
  (loop [[y & more] ys, prior-mean 0.0, prior-var 1.0, ll 0.0, acc []]
    (if-not y
      acc
      (let [s (+ prior-var 1.0)
            k (/ prior-var s)
            mean (+ prior-mean (* k (- y prior-mean)))
            var (* (- 1.0 k) prior-var)
            ll (+ ll (dist/logpdf (dist/normal prior-mean (Math/sqrt s)) y))]
        (recur more mean (+ var 1.0) ll (conj acc [mean var ll]))))))

(defn- moments [measure t]
  (let [ps (m/get-particles measure)
        ws (weights/normalize-log-weights (mapv second ps))
        xs (mapv (fn [[s _]] (:value (get (m/get-trace s) [:x t]))) ps)
        mean (reduce + (map * ws xs))]
    [mean (reduce + (map (fn [w x] (* w (let [d (- x mean)] (* d d)))) ws xs))]))

(deftest online-smc-tracks-the-kalman-filter
  (random/set-seed! 5)
  (let [root (context/create-execution-context)
        truth (kalman ys)
        first-step (await-cps (smc/stream (random-walk root) 3000
                                          {:executor (executor/thread-pool-executor {:threads 1})}))]
    (is (not (:done? first-step)) "waiting for the first observation")
    (loop [step first-step t 0]
      (if (= t (count ys))
        (do (is (:done? step) "every particle returned after the last observation")
            (is (< (Math/abs (- (m/log-marginal (:measure step)) (nth (peek truth) 2))) 0.15)
                (str "evidence " (m/log-marginal (:measure step)) " vs " (nth (peek truth) 2))))
        (let [step' (await-cps ((:push step) (nth ys t)))
              [mean var] (moments (:measure step') t)
              [kmean kvar] (nth truth t)]
          (testing (str "after y_" t)
            (is (< (Math/abs (- mean kmean)) 0.08) (str "mean " mean " vs " kmean))
            (is (< (Math/abs (- var kvar)) 0.06) (str "var " var " vs " kvar)))
          (recur step' (inc t)))))))

(deftest batch-smc-refuses-stream-sites
  (let [root (context/create-execution-context)]
    (is (thrown-with-msg? clojure.lang.ExceptionInfo #"stream sites"
                          (await-cps (smc/smc (random-walk root) 10))))))

(deftest a-population-forks-into-independent-copies
  (random/set-seed! 7)
  (let [root (context/create-execution-context)
        truth (kalman ys)
        push (fn [step y] (await-cps ((:push step) y)))
        start (await-cps (smc/stream (random-walk root) 2000
                                     {:executor (executor/thread-pool-executor {:threads 1})}))
        shared (reduce push start (take 2 ys))
        copy (await-cps ((:fork shared)))
        original (reduce push shared (drop 2 ys))
        _ ((:close original))
        forked (reduce push copy (drop 2 ys))
        [_ _ ll] (peek truth)]
    (is (= (m/log-marginal (:measure shared)) (m/log-marginal (:measure copy)))
        "the copy starts where the population stood")
    (doseq [[label step] [["original" original] ["copy" forked]]]
      (testing label
        (is (:done? step))
        (is (< (Math/abs (- (m/log-marginal (:measure step)) ll)) 0.15))))
    (is (not= (m/log-marginal (:measure original)) (m/log-marginal (:measure forked)))
        "the copies resample independently")
    ((:close forked))))

(deftest a-population-with-anchors-offers-no-fork
  (let [root (context/create-execution-context)
        step (await-cps (smc/stream (random-walk root) 20 {:anchors :all}))]
    (is (nil? (:fork step)))
    ((:close step))))

(deftest a-pushed-value-is-scored-under-the-particles-policy
  ;; do(y_0 = 0): the stream site takes the intervened value and scores
  ;; nothing, as an observed site does under the same policy; scoring the
  ;; pushed 5.0 instead would give log N(5; 0, √2) ≈ −7.5
  (let [root (context/create-execution-context)
        step (await-cps (smc/stream (random-walk root) 50
                                    {:policy (itrace/policy {:interventions {[:y 0] {:do 0.0}}})}))
        step' (await-cps ((:push step) 5.0))]
    (is (= 0.0 (m/log-marginal (:measure step'))))
    ((:close step'))))

(deftest a-pushed-value-is-tempered-as-an-observation
  ;; under temperature 1/2 the pushed y = 5 counts N(5; x, 1)^(1/2):
  ;; log Z = −¼ log 2π + ½ log 4π + log N(5; 0, √3) ≈ −4.829
  (let [root (context/create-execution-context)
        exact (+ (* -0.25 (Math/log (* 2 Math/PI))) (* 0.5 (Math/log (* 4 Math/PI)))
                 (dist/logpdf (dist/normal 0.0 (Math/sqrt 3.0)) 5.0))
        step (await-cps (smc/stream (random-walk root) 4000 {:policy (itrace/policy {:temperature 0.5})}))
        step' (await-cps ((:push step) 5.0))]
    (is (< (Math/abs (- (m/log-marginal (:measure step')) exact)) 0.1)
        (str "log Z " (m/log-marginal (:measure step')) " exact " exact))
    ((:close step'))))

(defn- endless-walk [root]
  (binding [ec/*execution-context* root]
    (spin
     (loop [t 0 x (sample (dist/normal 0.0 1.0) :id [:x 0])]
       (sample (dist/normal x 1.0) :id [:y t] :stream true)
       (recur (inc t) (sample (dist/normal x 1.0) :id [:x (inc t)]))))))

(deftest a-push-does-not-convert-the-history
  ;; The step measure is built at every push. Converting every particle's
  ;; whole trace there made a push cost O(particles x steps so far): a
  ;; consumer reading the latest site must convert nothing.
  (let [root (context/create-execution-context)
        conversions (atom 0)
        convert itrace/legacy-trace
        pushes 30]
    (with-redefs [itrace/legacy-trace (fn [t] (swap! conversions inc) (convert t))]
      (loop [step (await-cps (smc/stream (endless-walk root) 20
                                         {:seed 3 :executor (executor/thread-pool-executor {:threads 1})}))
             t 0]
        (if (< t pushes)
          (let [step' (await-cps ((:push step) (* 0.1 t)))]
            (doseq [[s _] (m/get-particles (:measure step'))]
              (is (number? (:value (get (m/get-trace s) [:x (inc t)])))))
            (recur step' (inc t)))
          (do
            (is (zero? @conversions) "reading the latest sites converted no trace")
            ;; the whole trajectory is still there, converted on demand
            (let [tr (m/get-trace (ffirst (m/get-particles (:measure step))))]
              (is (= (into #{} (concat (map #(vector :x %) (range (inc pushes)))
                                       (map #(vector :y %) (range pushes))))
                     (set (keys tr))))
              (is (= (* 0.1 (dec pushes)) (:value (get tr [:y (dec pushes)])))))
            (is (pos? @conversions))
            ((:close step))))))))

(defn- run-stream
  "[x-means trace-keys] after each of `pushes` pushes: the weighted mean of
  x_t, and the keys of the first particle's trace."
  [opts pushes]
  (let [root (context/create-execution-context)]
    (loop [step (await-cps (smc/stream (endless-walk root) 20
                                       (merge {:seed 7 :executor (executor/thread-pool-executor {:threads 1})} opts)))
           t 0 acc []]
      (if (= t pushes)
        (do ((:close step)) acc)
        (let [step' (await-cps ((:push step) (Math/sin t)))
              ps (m/get-particles (:measure step'))
              ws (weights/normalize-log-weights (mapv second ps))
              mean (reduce + (map (fn [w [s _]] (* w (m/site-value s [:x t]))) ws ps))]
          (recur step' (inc t) (conj acc [mean (set (keys (m/get-trace (ffirst ps))))
                                          (count (:history (:measure step')))])))))))

(deftest stream-values-read-as-observed
  (let [[[_ ks]] (run-stream {} 1)
        root (context/create-execution-context)
        step (await-cps (smc/stream (endless-walk root) 5 {:seed 1}))
        step' (await-cps ((:push step) 0.5))
        tr (m/get-trace (ffirst (m/get-particles (:measure step'))))]
    ((:close step'))
    (is (contains? ks [:y 0]))
    (is (:observed? (get tr [:y 0])))
    (is (not (contains? (pmcmc/retained-choices tr) [:y 0])) "an observation is not a latent choice")))

(deftest retain-windows-the-traces-not-the-filtering-marginals
  (let [full (run-stream {} 12)
        filtering (run-stream {:retain :filtering} 12)
        last3 (run-stream {:retain {:last 3}} 12)]
    (is (= (map first full) (map first filtering) (map first last3))
        "same seed, same filtering means: windowing changes no draw")
    (let [[_ ks h] (peek filtering)]
      (is (= #{[:x 11] [:y 11] [:x 12]} ks) "the last step and the state drawn since")
      (is (= 1 h)))
    (let [[_ ks h] (peek last3)]
      (is (= #{[:x 9] [:y 9] [:x 10] [:y 10] [:x 11] [:y 11] [:x 12]} ks))
      (is (= 3 h)))
    (is (= 25 (count (second (peek full)))) ":all keeps the trajectory: x_0..x_12, y_0..y_11")))

(deftest retain-refuses-what-needs-the-trajectory
  (doseq [opts [{:retain :filtering :anchors true}
                {:retain {:last 2} :retained {}}]]
    (is (thrown-with-msg? clojure.lang.ExceptionInfo #":retain is not supported"
                          (await-cps (smc/stream (endless-walk (context/create-execution-context)) 4 opts)))))
  (is (thrown-with-msg? clojure.lang.ExceptionInfo #":retain is :all"
                        (await-cps (smc/stream (endless-walk (context/create-execution-context)) 4 {:retain {:last 0}})))))

(deftest a-windowed-population-forks
  (let [root (context/create-execution-context)
        step (await-cps (smc/stream (endless-walk root) 10 {:seed 2 :retain :filtering}))
        step (reduce (fn [s y] (await-cps ((:push s) y))) step [0.1 0.2 0.3])
        copy (await-cps ((:fork step)))
        copy' (await-cps ((:push copy) 0.4))
        step' (await-cps ((:push step) 0.4))]
    (is (= (mapv second (m/get-particles (:measure step')))
           (mapv second (m/get-particles (:measure copy')))))
    (is (= #{[:x 3] [:y 3] [:x 4]} (set (keys (m/get-trace (ffirst (m/get-particles (:measure copy'))))))))
    ((:close copy'))
    ((:close step'))))
