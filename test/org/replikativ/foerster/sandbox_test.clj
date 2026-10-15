(ns org.replikativ.foerster.sandbox-test
  "What a sandbox gets: posteriors as plain data with their chain structure,
  the host's limits and defaults, background jobs that stop when cancelled,
  and host-side log densities equal to the distributions'."
  (:require [clojure.test :refer [deftest is testing]]
            [clojure.walk :as walk]
            [org.replikativ.foerster.block :as block]
            [org.replikativ.foerster.diagnostics :as diag]
            [org.replikativ.foerster.dist :as dist]
            [org.replikativ.foerster.effects :refer [sample observe]]
            [org.replikativ.foerster.sandbox :as sb]
            [org.replikativ.spindel.effects.await :refer [await]]
            [org.replikativ.spindel.engine.context :as ctx]
            [org.replikativ.spindel.engine.core :as ec]
            [org.replikativ.spindel.spin.cps :refer [spin]]))

(def ^:private ns-maps (sb/namespaces {:world-policy :fresh :max-work 20000}))
(defn- f [ns sym] (get-in ns-maps [ns sym]))

(defn- in-context [thunk]
  (let [root (ctx/create-execution-context)]
    (try (binding [ec/*execution-context* root] (thunk))
         (finally (ctx/stop-context! root)))))

(defn- run!
  "Run the model `make-model` returns, in a fresh context."
  [make-model opts]
  (in-context #(deref (future @(spin (await ((f 'infer 'run) (make-model) opts)))) 120000 ::timeout)))

(defn- conjugate []
  ;; μ ~ N(0, 1), y ~ N(μ, 1) for y = 1, 2, 1.5: μ | y ~ N(1.125, 0.5²)
  (spin (let [mu (sample (dist/normal 0.0 1.0) :id :mu)]
          (loop [ys [1.0 2.0 1.5]]
            (when (seq ys) (observe (dist/normal mu 1.0) (first ys)) (recur (rest ys))))
          mu)))

(defn- plain-data? [x]
  (let [ok (atom true)]
    (walk/postwalk (fn [v] (when-not (or (nil? v) (number? v) (keyword? v) (string? v) (boolean? v)
                                         (map? v) (vector? v) (seq? v) (set? v) (map-entry? v))
                             (reset! ok false))
                     v)
                   x)
    @ok))

(deftest posteriors-are-plain-data
  (testing "particles"
    (let [post (run! conjugate {:method :smc :particles 1000 :seed 3})
          {:keys [mean sd ess]} (sb/summary post)]
      (is (plain-data? post))
      (is (nil? (:posterior/chain-lengths post)))
      (is (< (Math/abs (- mean 1.125)) 0.1))
      (is (< (Math/abs (- sd 0.5)) 0.1))
      (is (pos? ess))))
  (testing "chains: their structure survives, for R-hat and ESS"
    (let [post (run! conjugate {:method :mh :iterations 600 :chains 3 :burn 100 :seed 4})
          s (sb/summary post)]
      (is (plain-data? post))
      (is (= [500 500 500] (:posterior/chain-lengths post)))
      (is (= 3 (:chains s)))
      (is (= (diag/rhat (sb/chains post nil)) (:rhat s)))
      (is (< (:rhat s) 1.1))
      (is (< (Math/abs (- (:mean s) 1.125)) 0.15)))))

(deftest particle-mcmc-runs-here
  (let [post (run! conjugate {:method :pgibbs :particles 20 :iterations 40 :seed 3})
        mean (reduce + (map * (:posterior/values post) (:posterior/weights post)))]
    (is (< (Math/abs (- mean 1.125)) 0.15))))

(deftest the-host-world-policy-is-for-particle-methods
  (let [seen (atom [])]
    (with-redefs [org.replikativ.foerster.core/infer (fn [_ opts] (swap! seen conj opts) (spin nil))]
      (let [nss (sb/namespaces {:world-policy :fork})]
        (in-context #(do ((get-in nss ['infer 'run]) (conjugate) {:method :smc :particles 10})
                         ((get-in nss ['infer 'run]) (conjugate) {:method :nuts :iterations 10 :chains 1})))))
    (is (= [:fork nil] (map :world-policy @seen)))))

(deftest the-host-sets-defaults-and-limits
  (in-context
   (fn []
     (is (thrown-with-msg? clojure.lang.ExceptionInfo #"larger than this sandbox allows"
                           ((f 'infer 'run) (conjugate) {:method :smc :particles 50000})))
     (is (thrown-with-msg? clojure.lang.ExceptionInfo #"larger than this sandbox allows"
                           ((f 'infer 'run) (conjugate) {:method :mh :iterations 10000 :chains 4})))
     (is (thrown-with-msg? clojure.lang.ExceptionInfo #"not available here"
                           ((f 'infer 'run) (conjugate) {:method :tempered :particles 2})))
     (is (thrown-with-msg? clojure.lang.ExceptionInfo #"larger than this sandbox allows"
                           ((f 'infer 'run) (conjugate) {:method :ipmcmc :particles 1000 :iterations 10})))
     (is (thrown-with-msg? clojure.lang.ExceptionInfo #"must be an integer"
                           ((f 'infer 'run) (conjugate) {:method :mh :iterations 0.5 :chains 2})))
     (doseq [[opts why] [[{:method :smc :particles 10 :executor :x} "the host's executor"]
                         [{:method :smc :particles 10 :policy identity} "a policy callback sees live worlds"]
                         [{:method :smc :particles 10 :world-opts {}} "world options"]]]
       (is (thrown-with-msg? clojure.lang.ExceptionInfo #"not available here" ((f 'infer 'run) (conjugate) opts)) why))
     (is (thrown-with-msg? clojure.lang.ExceptionInfo #"not available here"
                           ((f 'infer 'run) (conjugate) {:method :parallel-tempering :chains 100000})))
     (is (thrown-with-msg? clojure.lang.ExceptionInfo #"Sizes missing"
                           ((f 'infer 'run) (conjugate) {:method :nuts})))))
  (testing "a seeded run reproduces"
    (is (= (run! conjugate {:method :mh :iterations 200 :chains 2 :seed 9})
           (run! conjugate {:method :mh :iterations 200 :chains 2 :seed 9})))))

(defn- busy-chain-threads []
  (->> (Thread/getAllStackTraces)
       (filter (fn [[t st]] (and (= Thread$State/RUNNABLE (.getState ^Thread t))
                                 (some #(re-find #"foerster\.(trace|nuts|kernel|core)" (str %)) st))))
       count))

(deftest jobs-run-in-the-background-and-stop-when-cancelled
  (let [started (atom [])
        nss (sb/namespaces {:world-policy :fresh :on-start #(swap! started conj %)})
        start (get-in nss ['infer 'start])
        result (get-in nss ['infer 'result])
        status (get-in nss ['infer 'status])
        cancel (get-in nss ['infer 'cancel])]
    (in-context
     (fn []
       (testing "a short job"
         (let [job (start (conjugate) {:method :smc :particles 500 :seed 1})
               post (result job 60000)]
           (is (= :done (:status (status job))))
           (is (= 500 (count (:posterior/values post))))
           (is (= #{:job/id} (set (keys job))) "an opaque handle")
           (is (= 1 (count @started)) "the host gets the job itself")))
       (testing "a handle from elsewhere"
         (is (thrown-with-msg? clojure.lang.ExceptionInfo #"Not a job" (status {:job/id (random-uuid)}))))
       (testing "a long job is pending, then cancelled"
         (let [job (start (conjugate) {:method :mh :iterations 1000000 :chains 2 :seed 2})]
           (is (= :pending (result job 500)))
           (is (= :running (:status (status job))))
           (cancel job)
           (Thread/sleep 3000)
           (is (= :cancelled (:status (status job))))
           (is (zero? (busy-chain-threads)))))))))

(deftest log-densities-equal-the-distributions
  (let [rng (java.util.Random. 2)
        xs (vec (repeatedly 50 (fn [] [(.nextGaussian rng) (.nextGaussian rng)])))
        ys (mapv (fn [[a b]] (+ 0.3 a (* -0.5 b) (.nextGaussian rng))) xs)
        bs (mapv #(if (pos? %) 1 0) ys)
        cs (mapv #(long (Math/floor (Math/abs %))) ys)
        eta (mapv (fn [[a b]] (+ 0.3 (* 0.8 a) (* -0.4 b))) xs)
        close? #(< (Math/abs (- %1 %2)) 1e-9)]
    (is (close? (sb/normal ys 0.3 1.7) (reduce + (map #(dist/logpdf (dist/normal 0.3 1.7) %) ys))))
    (is (close? (sb/normal-glm ys xs 0.3 [0.8 -0.4] 1.2)
                (reduce + (map #(dist/logpdf (dist/normal %2 1.2) %1) ys eta))))
    (is (close? (sb/bernoulli-logit-glm bs xs 0.3 [0.8 -0.4])
                (reduce + (map #(dist/logpdf (dist/bernoulli-logit %2) %1) bs eta))))
    (is (close? (sb/poisson-log-glm cs xs 0.3 [0.8 -0.4])
                (reduce + (map #(dist/logpdf (dist/poisson (Math/exp %2)) %1) cs eta))))
    (is (= ##-Inf (sb/normal ys 0.0 0.0)))
    (is (= ##-Inf (sb/bernoulli-logit-glm (assoc bs 0 2) xs 0.3 [0.8 -0.4])))
    (is (= ##-Inf (sb/poisson-log-glm (assoc cs 0 0.5) xs 0.3 [0.8 -0.4])))))

(deftest blocks-and-optimizers
  (let [b ((f 'model 'block) {:block/id :g :block/latents [{:name :theta :shape [2]}] :block/target :complete-conditional}
                             ((f 'model 'numeric-gradient)
                              {:log-density (fn [th _] (let [x (nth th 0) y (/ (nth th 1) 2.0) r 0.8]
                                                         (/ (+ (* x x) (* -2.0 r x y) (* y y)) (* -2.0 (- 1.0 (* r r))))))}))
        site ((f 'model 'site) b nil)
        lap ((f 'opt 'laplace) site {:init [0.5 0.5]})]
    (is (vector? (:theta lap)))
    (is (< (Math/abs (- 4.0 (get-in lap [:covariance 1 1]))) 1e-3))
    (testing "NUTS on the block site"
      (let [post (run! #(spin (sample site :id :theta :init [0.0 0.0])) {:method :nuts :iterations 600 :chains 2 :burn 300 :seed 5})
            s (sb/summary post 1)]
        (is (< (Math/abs (- 2.0 (:sd s))) 0.3))
        (is (< (:rhat s) 1.05))))))

(deftest only-plain-data-and-messages-cross
  (testing "a value that is not data"
    (is (thrown-with-msg? Exception #"plain data"
                          (run! #(spin (sample (dist/normal 0.0 1.0)) (Object.)) {:method :importance :particles 5}))))
  (testing "an error keeps its messages and type, not its data"
    (let [e (try (run! #(spin (sample (dist/normal 0.0 1.0)) (throw (ex-info "boom" {:handle (Object.)})))
                       {:method :importance :particles 5})
                 (catch Exception e e))
          e (loop [e e] (if (and (ex-cause e) (not (re-find #"boom" (str (ex-message e))))) (recur (ex-cause e)) e))]
      (is (re-find #"boom" (ex-message e)))
      (is (nil? (:handle (ex-data e))))))
  (testing "metadata does not cross"
    (let [post (run! #(spin (sample (dist/normal 0.0 1.0)) (with-meta {:a 1} {:handle (Object.)}))
                     {:method :importance :particles 3})]
      (is (every? nil? (map meta (:posterior/values post))))))
  (testing "no posterior when every particle has zero likelihood"
    (is (thrown-with-msg? Exception #"zero likelihood"
                          (run! #(spin (let [x (sample (dist/normal 0.0 1.0))] (observe (dist/uniform 10.0 11.0) 0.0) x))
                                {:method :importance :particles 5}))))
  (testing "nor from particle MCMC"
    (doseq [opts [{:method :pimh :particles 5 :iterations 2}
                  {:method :pgibbs :particles 5 :iterations 2}
                  {:method :pgas :particles 5 :iterations 2}
                  {:method :ipmcmc :particles 5 :iterations 1}]]
      (is (thrown-with-msg? Exception #"zero likelihood"
                            (run! #(spin (let [x (sample (dist/normal 0.0 1.0))] (observe (dist/uniform 10.0 11.0) x) x))
                                  opts))
          (str (:method opts))))))

(deftest a-finished-job-stays-finished
  (in-context
   (fn []
     (let [job ((f 'infer 'start) (conjugate) {:method :smc :particles 100 :seed 1})
           post ((f 'infer 'result) job 60000)]
       ((f 'infer 'cancel) job)
       (is (= :done (:status ((f 'infer 'status) job))))
       (is (= post ((f 'infer 'result) job 10)))
       ((f 'infer 'forget) job)
       (is (thrown-with-msg? clojure.lang.ExceptionInfo #"Not a job" ((f 'infer 'status) job)))))))

(deftest optimizers-return-data-from-seeded-streams
  (let [b ((f 'model 'block) {:block/id :n :block/latents [{:name :theta :shape [1]}] :block/target :complete-conditional}
                             ((f 'model 'numeric-gradient) {:log-density (fn [th _] (* -0.5 (nth th 0) (nth th 0)))}))
        site ((f 'model 'site) b nil)
        lap ((f 'opt 'laplace) site {:init [0.3] :draws 5 :seed 1})
        pf #((f 'opt 'pathfinder) site {:draws 50 :seed 2})]
    (is (sb/data? lap))
    (is (= 5 (count (:draws lap))))
    (is (nil? (:draw lap)))
    (is (= (pf) (pf)) "the same seed, the same Pathfinder")
    (is (= 50 (count (:draws (pf)))))
    (is (thrown-with-msg? clojure.lang.ExceptionInfo #"from 0 to"
                          ((f 'opt 'pathfinder) site {:paths 1000})))
    (is (thrown-with-msg? clojure.lang.ExceptionInfo #"Optimization larger"
                          ((get-in (sb/namespaces {:max-optimizer-work 10000}) ['opt 'pathfinder]) site {:paths 8})))
    (is (thrown-with-msg? clojure.lang.ExceptionInfo #"not available here"
                          ((f 'opt 'laplace) site {:callback identity})))))
