(ns org.replikativ.foerster.dist-test
  "Distributions agree with reference values (Apache Commons Math), sample
  their laws, and a seed draws the same numbers on the JVM and in JavaScript."
  (:require #?(:clj [clojure.test :refer [deftest is testing]]
               :cljs [cljs.test :refer-macros [deftest is testing]])
            [org.replikativ.foerster.dist :as d]
            [org.replikativ.foerster.mechanism :as mech]
            [org.replikativ.foerster.random :as random]))

(defn- close? [a b] (< (Math/abs (- a b)) (* 1e-12 (max 1.0 (Math/abs b)))))

(deftest densities-agree-with-reference-values
  (is (close? (d/logpdf (d/normal 1.0 2.0) -0.7) -1.9733357137646181))
  (is (close? (d/cdf (d/normal 1.0 2.0) -0.7) 0.19766254312269238))
  (is (close? (d/quantile (d/normal 1.0 2.0) 0.975) 4.919927969080108))
  (is (close? (d/logpdf (d/gamma 2.5 3.0) 4.2) -2.278586804209209))
  (is (close? (d/cdf (d/gamma 2.5 3.0) 4.2) 0.2692135134112415))
  (is (close? (d/logpdf (d/beta 2.0 5.0) 0.3) 0.7705248015812898))
  (is (close? (d/logpdf (d/poisson 12.0) 9) -2.4376676319894672))
  (is (close? (d/cdf (d/poisson 12.0) 9) 0.24239216167051233))
  (is (close? (d/logpdf (d/exponential 0.7) 1.3) -1.2666749439387324))
  (is (close? (d/logpdf (d/mvn [1.0 -2.0] [[2.0 0.6] [0.6 1.0]]) [0.5 -1.0]) -2.9541276263517897))
  (is (close? (d/logpdf (d/categorical {:a 1 :b 3}) :b) (Math/log 0.75)))
  (is (= ##-Inf (d/logpdf (d/categorical {:a 1 :b 3}) :c)))
  (is (close? (d/logpdf (d/student-t 5.0) 0.65)
              (- (d/logpdf (d/student-t 5.0 1.0 2.0) 2.3) (- (Math/log 2.0)))))
  (testing "a whole number is a count whatever its type"
    (is (= (d/logpdf (d/poisson 12.0) 9) (d/logpdf (d/poisson 12.0) 9.0))))
  (testing "outside the support"
    (is (= ##-Inf (d/logpdf (d/gamma 2.0 1.0) -1.0)))
    (is (= ##-Inf (d/logpdf (d/poisson 3.0) 1.5)))
    (is (= ##-Inf (d/logpdf (d/discrete [1 2 3]) 3)))))

(deftest the-normal-cdf-keeps-its-precision-in-the-tail
  ;; Φ(-8) = 6.22096057427178e-16
  (is (< (Math/abs (- (/ (d/normal-cdf -8.0) 6.22096057427178e-16) 1.0)) 1e-13))
  (is (close? (d/normal-quantile 1e-10) -6.361340902404056)))

(deftest the-generator-is-xoshiro128**
  ;; the reference output of state (1 2 3 4)
  (is (= [11520 0 5927040 70819200 2031721883 1637235492]
         (loop [s [1 2 3 4] n 6 out []]
           (if (zero? n)
             out
             (let [[o s'] (@#'random/step s)] (recur s' (dec n) (conj out o))))))))

(deftest a-seed-draws-the-same-numbers-on-every-platform
  (random/set-seed! 7)
  (is (= 279963896 (random/next-u32! (random/current))))
  (is (= 0.24152057094740398 (random/uniform01)))
  (is (close? (d/draw (d/normal 0 1)) 1.9775113839366343))
  (is (close? (d/draw (d/gamma 2.5 3.0)) 1.0685424805002073))
  (is (= 12 (d/draw (d/poisson 12.0)))))

(defn- moments [xs]
  (let [n (count xs) m (/ (reduce + xs) n)]
    [m (/ (reduce + (map #(let [r (- % m)] (* r r)) xs)) n)]))

(deftest draws-follow-their-laws
  (random/set-seed! 42)
  (let [n 20000]
    (doseq [dist [(d/normal 1.5 2.0) (d/uniform -1.0 3.0) (d/exponential 0.7)
                  (d/gamma 0.4 2.0) (d/gamma 3.3 1.5) (d/beta 2.0 5.0)
                  (d/poisson 3.2) (d/poisson 57.0) (d/bernoulli 0.3)
                  (d/student-t 5.0 1.0 2.0) (d/chi-squared 3.0)]]
      (let [[m v] (moments (vec (repeatedly n #(d/draw dist))))]
        (is (< (Math/abs (/ (- m (d/mean dist)) (Math/sqrt (/ (d/variance dist) n)))) 4.0)
            (str dist " mean"))
        (is (< (Math/abs (/ (- v (d/variance dist)) (d/variance dist))) 0.06)
            (str dist " variance"))))))

(deftest edge-cases
  (testing "a flat factor stays flat at the boundary"
    (is (close? (d/logpdf (d/beta 1.0 1.0) 0.0) 0.0))
    (is (close? (d/logpdf (d/beta 1.0 2.0) 0.0) (Math/log 2.0)))
    (is (= ##-Inf (d/logpdf (d/beta 2.0 2.0) 0.0)))
    (is (close? (d/logpdf (d/dirichlet [1.0 1.0 1.0]) [0.0 0.5 0.5]) (Math/log 2.0))))
  (testing "the Dirichlet lives on the simplex"
    (is (= ##-Inf (d/logpdf (d/dirichlet [2.0 2.0]) [0.3 0.3]))))
  (testing "an outcome listed twice carries both weights"
    (let [c (d/categorical [[:a 1.0] [:b 1.0] [:a 2.0]])]
      (is (close? (d/logpdf c :a) (Math/log 0.75)))
      (is (close? (d/logpdf c :b) (Math/log 0.25)))))
  (testing "Bernoulli outcomes as any number, and nothing else"
    (is (close? (d/logpdf (d/bernoulli 0.3) 1.0) (Math/log 0.3)))
    (is (= ##-Inf (d/logpdf (d/bernoulli 0.3) true))))
  (testing "a Bernoulli mechanism abducts 1.0 as 1"
    (is (< (mech/abduct (d/bernoulli 0.3) 1.0) 0.3)))
  (testing "quantiles take probabilities"
    (is (= ##Inf (d/quantile (d/poisson 3.0) 1.0)))
    (is (= 0 (d/quantile (d/poisson 3.0) 0.0)))
    (is (thrown? #?(:clj clojure.lang.ExceptionInfo :cljs ExceptionInfo)
                 (d/quantile (d/poisson 3.0) 1.5)))))

(deftest parameters-are-validated
  (doseq [make [#(d/normal 0.0 -1.0) #(d/normal 0.0 0.0) #(d/normal ##NaN 1.0)
                #(d/uniform 1.0 1.0) #(d/exponential 0.0) #(d/gamma -1.0 1.0)
                #(d/beta 1.0 0.0) #(d/poisson -2.0) #(d/bernoulli 1.5) #(d/flip -0.1)
                #(d/discrete []) #(d/discrete [1.0 -1.0]) #(d/discrete [0.0 0.0])
                #(d/dirichlet [1.0 0.0]) #(d/categorical {}) #(d/student-t 0.0)
                #(d/chi-squared -1.0) #(d/mvn [0.0 0.0] [[1.0]])
                #(d/bernoulli-logit ##Inf) #(d/binomial-logit 10 ##NaN) #(d/binomial-logit 2.5 0.0)]]
    (is (= ::d/invalid-parameters
           (try (make) nil
                (catch #?(:clj clojure.lang.ExceptionInfo :cljs ExceptionInfo) e
                  (:type (ex-data e))))))))

(deftest continuity-is-the-law's
  (is (every? d/continuous? [(d/normal 0 1) (d/gamma 2 1) (d/beta 2 2) (d/uniform 0 1)]))
  (is (not-any? d/continuous? [(d/poisson 3) (d/discrete [1 1]) (d/flip 0.5)
                               (d/bernoulli 0.5) (d/dirichlet [1 1])])))

(deftest logit-laws-keep-their-density-where-sigma-rounds
  (let [sigmoid (fn [t] (/ 1.0 (+ 1.0 (Math/exp (- t)))))]
    (testing "they agree with the probability parameterization"
      (doseq [eta [-3.0 -0.4 0.0 0.7 5.0] k [0 3 10]]
        (is (close? (d/logpdf (d/binomial-logit 10 eta) k) (d/logpdf (d/binomial 10 (sigmoid eta)) k))))
      (doseq [eta [-2.0 0.0 1.5] x [0 1]]
        (is (close? (d/logpdf (d/bernoulli-logit eta) x) (d/logpdf (d/bernoulli (sigmoid eta)) x))))
      (is (close? (d/mean (d/binomial-logit 10 0.7)) (* 10 (sigmoid 0.7))))
      (is (= (range 11) (d/support (d/binomial-logit 10 0.7)))))
    (testing "σ(40) rounds to 1.0, so the binomial says a miss is impossible; the logit law does not"
      (is (= 1.0 (sigmoid 40.0)))
      (is (= ##-Inf (d/logpdf (d/binomial 10 (sigmoid 40.0)) 9)))
      (is (close? (d/logpdf (d/binomial-logit 10 40.0) 9) (- (Math/log 10.0) 40.0)))
      (is (close? (d/logpdf (d/bernoulli-logit -40.0) 1) -40.0))
      (is (close? (d/logpdf (d/bernoulli-logit 800.0) 0) -800.0)))))
