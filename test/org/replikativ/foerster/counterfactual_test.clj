(ns org.replikativ.foerster.counterfactual-test
  "Counterfactuals against closed forms: mechanisms round-trip their noise,
  a linear-Gaussian twin is exact, a confounded one is exact whatever the
  posterior over the confounder, and the probability of necessity of a binary
  model matches its analytic value."
  (:require [clojure.test :refer [deftest is testing]]
            [org.replikativ.foerster.counterfactual :as cf]
            [org.replikativ.foerster.mechanism :as mech]
            [org.replikativ.foerster.measure :as m]
            [org.replikativ.foerster.gfi :as gfi]
            [org.replikativ.foerster.attribution :as attr]
            [org.replikativ.foerster.effects :refer [sample observe deterministic]]
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

(def ^:private root (context/create-execution-context))

(defmacro ^:private model [& body]
  `(binding [ec/*execution-context* root] (spin ~@body)))

(defn- close? [a b] (< (Math/abs (- a b)) 1e-9))

(deftest mechanisms-round-trip-their-noise
  (random/set-seed! 1)
  (doseq [d [(dist/normal 2.0 3.0) (dist/uniform -1.0 4.0) (dist/exponential 1.5)]]
    (let [u (mech/noise d)]
      (is (close? u (mech/abduct d (mech/push d u))) (str (type d)))))
  (testing "discrete laws: abducted noise gives the value back"
    (doseq [d [(dist/flip 0.3) (dist/bernoulli 0.6) (dist/discrete [1.0 2.0 3.0])]]
      (dotimes [_ 50]
        (let [x (dist/draw d)]
          (is (= x (mech/push d (mech/abduct d x))) (str (type d) " " x)))))))

(deftest a-linear-gaussian-twin-is-exact
  ;; X = Ux, Y = 2X + Uy. Seen X = 1, Y = 3 (so Uy = 1); had X been 2, Y = 5.
  (let [[pair] (await-cps (cf/counterfactual
                           (model (let [x (sample (dist/normal 0.0 1.0) :id :x)
                                        y (sample (dist/normal (* 2.0 x) 1.0) :id :y)]
                                    {:x x :y y}))
                           {:evidence {:x 1.0 :y 3.0} :interventions {:x {:do 2.0}}}))]
    (is (= {:x 1.0 :y 3.0} (:factual pair)))
    (is (close? 5.0 (:y (:counterfactual pair))))
    (is (= [] (:unaligned pair)))))

(deftest a-confounded-twin-is-exact-under-any-posterior
  ;; Z ~ N(0,1), X ~ N(Z,1), Y ~ N(X+Z,1). Given X = 1, Y = 2.5 the posterior
  ;; over Z is uncertain, but Y - X = Z + Uy is known: Y' = 2 + 1.5 = 3.5.
  (random/set-seed! 2)
  (let [pairs (await-cps (cf/counterfactual
                          (model (let [z (sample (dist/normal 0.0 1.0) :id :z)
                                       x (sample (dist/normal z 1.0) :id :x)
                                       y (sample (dist/normal (+ x z) 1.0) :id :y)]
                                   {:z z :x x :y y}))
                          {:evidence {:x 1.0 :y 2.5} :interventions {:x {:do 2.0}} :particles 50}))]
    (is (< 1 (count (distinct (map (comp :z :factual) pairs)))) "Z is sampled, not fixed")
    (is (every? #(close? 3.5 (:y (:counterfactual %))) pairs))
    (is (every? #(= (:z (:factual %)) (:z (:counterfactual %))) pairs) "Z keeps its noise")))

(deftest the-probability-of-necessity-of-a-binary-cause
  ;; X ~ flip(.5); Y ~ flip(X ? 1 : .3). Seen X = Y = true: had X been false,
  ;; Y = [U < .3] with U | Y=true, X=true ~ U(0,1): P(Y' = false) = .7.
  (random/set-seed! 3)
  (let [pairs (await-cps (cf/counterfactual
                          (model (let [x (sample (dist/flip 0.5) :id :x)
                                       y (sample (dist/flip (if x 1.0 0.3)) :id :y)]
                                   {:x x :y y}))
                          {:evidence {:x true :y true} :interventions {:x {:do false}} :particles 1500}))
        ws (m/normalize-log-weights (mapv :weight pairs))
        pn (reduce + (map (fn [p w] (if (:y (:counterfactual p)) 0.0 w)) pairs ws))]
    (is (< (Math/abs (- pn 0.7)) 0.04) (str "PN " pn))))

(deftest a-site-only-the-counterfactual-reaches-is-reported
  (let [[pair] (await-cps (cf/counterfactual
                           (model (let [b (sample (dist/flip 0.5) :id :b)]
                                    (if b (sample (dist/normal 0.0 1.0) :id :u) :none)))
                           {:evidence {:b false} :interventions {:b {:do true}}}))]
    (is (= :none (:factual pair)))
    (is (number? (:counterfactual pair)))
    (is (= [:u] (:unaligned pair)))))

(deftest a-failing-world-fails-the-query
  ;; the program throws in the counterfactual world (x = 2): the query
  ;; rejects with the program's error instead of answering
  (let [boom (fn [x] (when (= 2.0 x) (throw (ex-info "boom" {:x x}))) x)
        outcome (try (await-cps (cf/counterfactual
                                 (model (let [x (sample (dist/normal 0.0 1.0) :id :x)]
                                          (boom x)))
                                 {:evidence {:x 1.0} :interventions {:x {:do 2.0}}}))
                     (catch Exception e e))]
    (is (instance? Exception outcome))
    (is (= "boom" (some #(when (= {:x 2.0} (ex-data %)) (ex-message %))
                        (take-while some? (iterate ex-cause outcome)))))))

(deftest reusable-abduction-and-nested-worlds
  ;; X -> M -> Y plus X -> Y. Both intermediate and outcome noise must
  ;; survive interventions; natural direct effect fixes M from world x=0.
  (let [program (model (let [x (sample (dist/normal 0 1) :id :x)
                             z (sample (dist/normal x 1) :id :z)
                             y (sample (dist/normal (+ x z) 1) :id :y)]
                         {:x x :z z :y y}))
        {t :trace} (await-cps (gfi/generate program {:x 1.0 :z 3.0 :y 7.0}))]
    (try
      (let [a (cf/abduct-trace t 0.0)
            r (await-cps (cf/worlds program a
                                    [[:untreated {:x {:do 0.0}}]
                                     [:direct (fn [worlds] {:x {:do 1.0}
                                                            :z {:do (get-in worlds [:untreated :z])}})]]))]
        (is (= :complete (:status r)))
        (is (close? 2.0 (get-in r [:worlds :untreated :z])))
        (is (close? 5.0 (get-in r [:worlds :untreated :y])))
        (is (close? 6.0 (get-in r [:worlds :direct :y])))
        (is (= [r] (cf/condition-worlds [r] #(> (get-in % [:direct :y]) 5.0))))
        (is (thrown? Exception (cf/condition-worlds [r] (constantly false)))))
      (finally (await-cps (gfi/close! t))))))

(deftest shifts-and-deterministic-interventions
  (let [program (model (let [x (sample (dist/normal 2 3) :id :x)
                             z (deterministic (* 2 x) :id :z)] [x z]))
        {t :trace} (await-cps (gfi/generate program {:x 8.0}))]
    (try
      (let [a (cf/abduct-trace t)
            shifted (await-cps (cf/predict program a {:x {:shift 5.0}}))
            fixed (await-cps (cf/predict program a {:z {:do 100.0}}))]
        (is (= [13.0 26.0] (:counterfactual shifted)))
        (is (= [8.0 100.0] (:counterfactual fixed))))
      (finally (await-cps (gfi/close! t))))))

(deftest incomplete-observed-coupling-is-refused-or-predicted
  (let [law (reify dist/Distribution (-draw [_] 7.0) (-logpdf [_ _] 0.0))
        program (model (observe law 99.0 :id :y))
        a {:factual 99.0 :noise {} :unsupported [:y] :weight 0.0}]
    (is (thrown? Exception (await-cps (cf/predict program a {}))))
    (let [r (await-cps (cf/predict program a {} {:strict? false}))]
      (is (= 7.0 (:counterfactual r)))
      (is (= :partial (:status r)))
      (is (= [:y] (:unaligned r))))
    ;; An explicit replacement with independent noise defines the new
    ;; mechanism, rather than accidentally clamping an observed site.
    (let [r (await-cps (cf/predict program a {:y {:dist law :coupling :fresh}}))]
      (is (= 7.0 (:counterfactual r)))
      (is (= :complete (:status r))))))

(deftest posterior-weights-survive-abduction
  (let [particle (fn [x] (m/sample-particle x {:x {:value x :distribution (dist/normal 0 1)}}))
        posterior (m/empirical [[(particle 1.0) (Math/log 0.25)]
                                [(particle 3.0) (Math/log 0.75)]])
        draws (cf/abduct-posterior posterior {:seed 7})]
    (is (close? 2.5 (cf/expectation draws :factual)))
    (is (= [1.0 3.0] (mapv #(get-in % [:noise :x]) draws)))))

(deftest unified-query-accepts-posteriors-and-cross-world-events
  (let [program (model (sample (dist/normal 0 1) :id :x))
        particle #(m/sample-particle % {:x {:value % :distribution (dist/normal 0 1)}})
        p (m/empirical [[(particle 1.0) (Math/log 0.25)] [(particle 3.0) (Math/log 0.75)]])
        r (await-cps (cf/query program {:posterior p :worlds [[:shifted {:x {:shift 2.0}}]]
                                        :event #(> (:shifted %) 4.0)}))]
    (is (= 1 (count r)))
    (is (= 5.0 (get-in r [0 :worlds :shifted])))
    (is (close? 3.0 (cf/expectation r #(get-in % [:worlds :factual])))))
  (is (thrown? Exception
               (await-cps (cf/query (model (sample (dist/flip 0.5) :id :x))
                                    {:evidence {:x true} :particles 0})))))

(deftest structural-laws-can-be-extended-and-categorical-abduction-is-supported
  (doseq [d [(dist/categorical [[:a 1] [:b 3]])
             (dist/log-normal 0 1) (dist/laplace 0 2) (dist/weibull 2 3)]]
    (let [x (dist/draw d) x' (mech/push d (mech/abduct d x))]
      (is (if (number? x) (< (Math/abs (- x x')) 1e-7) (= x x')))))
  (is (dist/continuous? (mech/shifted (dist/normal 0 1) 2)))
  (is (= [2 3] (dist/support (mech/shifted (dist/bernoulli 0.5) 2)))))

(deftest noise-attribution-and-joint-repairs
  (let [program (model (let [x (sample (dist/normal 0 1) :id :x)
                             y (sample (dist/normal 0 1) :id :y)] (+ x y)))
        a {:factual 5.0 :noise {:x 2.0 :y 3.0} :unsupported [] :weight 0.0}
        report (await-cps (attr/attribute-noise program a [{:x 0.0 :y 0.0}]
                                                {:nodes [:x :y] :score identity}))
        repairs (await-cps (attr/repairs program [a]
                                         {:x {:x {:do 0.0}} :both {:x {:do 0.0} :y {:do 0.0}}}
                                         #(<= % 1.0)))]
    (is (= {:x 2.0 :y 3.0} (:contributions report)))
    (is (= :both (:candidate (first repairs))))
    (is (= [1.0 0.0] (mapv :success-probability repairs)))))
