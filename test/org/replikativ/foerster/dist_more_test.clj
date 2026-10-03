(ns org.replikativ.foerster.dist-more-test
  "The laws added for regression and marketing models, pinned to scipy."
  (:require [clojure.test :refer [deftest is testing]]
            [org.replikativ.foerster.dist :as dist]
            [org.replikativ.foerster.random :as random]))

(defn- close? [a b] (< (Math/abs (- a b)) (* 1e-9 (max 1.0 (Math/abs b)))))

(deftest densities-agree-with-scipy
  (doseq [[label d x expected]
          [["weibull" (dist/weibull 1.5 2.0) 0.7 -1.0196559271096064]
           ["weibull" (dist/weibull 1.5 2.0) 3.1 -1.9982890446744568]
           ["laplace" (dist/laplace 0.5 1.3) -1.0 -2.10935759887359]
           ["laplace" (dist/laplace 0.5 1.3) 2.0 -2.10935759887359]
           ["inverse-gamma" (dist/inverse-gamma 3.0 2.0) 0.4 0.051457288616510444]
           ["inverse-gamma" (dist/inverse-gamma 3.0 2.0) 1.7 -1.912689231364085]
           ["gumbel" (dist/gumbel 1.0 2.0) -0.5 -2.06014719717262]
           ["gumbel" (dist/gumbel 1.0 2.0) 4.0 -2.4162773407083753]
           ["half-student-t" (dist/half-student-t 3.0 1.5) 0.4 -0.760061042962009]
           ["half-student-t" (dist/half-student-t 3.0 1.5) 5.0 -3.8099072180802533]
           ["beta-binomial" (dist/beta-binomial 10 2.0 3.0) 0 -2.719100037288795]
           ["beta-binomial" (dist/beta-binomial 10 2.0 3.0) 4 -1.967112356705917]
           ["beta-binomial" (dist/beta-binomial 10 2.0 3.0) 10 -4.51085950651685]
           ["truncated normal" (dist/truncated (dist/normal 1.0 2.0) 0.0 3.0) 0.5 -1.0137400812117545]
           ["truncated poisson" (dist/truncated (dist/poisson 3.0) 2 6) 2 -1.2311014717141884]
           ["censored at lo" (dist/censored (dist/normal 0.0 1.0) -1.0 1.0) -1.0 -1.8410216450092634]
           ["censored at hi" (dist/censored (dist/normal 0.0 1.0) -1.0 1.0) 1.0 -1.8410216450092634]
           ["censored inside" (dist/censored (dist/normal 0.0 1.0) -1.0 1.0) 0.3 -0.9639385332046727]
           ["zero-inflated at 0" (dist/zero-inflated 0.3 (dist/poisson 2.5)) 0 -1.028733212673555]
           ["zero-inflated at 3" (dist/zero-inflated 0.3 (dist/poisson 2.5)) 3 -1.899562217544322]
           ["hurdle at 0" (dist/hurdle 0.3 (dist/poisson 2.5)) 0 (Math/log 0.3)]
           ["hurdle at 2" (dist/hurdle 0.3 (dist/poisson 2.5)) 2 -1.6315901770083294]
           ["zero-sum-normal" (dist/zero-sum-normal 2.0 3) [1.0 -0.5 -0.5] -3.411671427529236]
           ["gamma by mean and sd" (dist/gamma-mean-sd 3.0 1.5) 2.0 -1.2282563044077621]]]
    (testing label (is (close? (dist/logpdf d x) expected) (str (dist/logpdf d x)))))
  (testing "cdfs"
    (is (close? (dist/cdf (dist/weibull 1.5 2.0) 1.2) 0.37171286869100184))
    (is (close? (dist/cdf (dist/laplace 0.5 1.3) 0.1) 0.36757074029584225))
    (is (close? (dist/cdf (dist/inverse-gamma 3.0 2.0) 0.8) 0.5438131158833297))
    (is (close? (dist/cdf (dist/gumbel 1.0 2.0) 2.0) 0.545239211892605))
    (is (close? (dist/cdf (dist/truncated (dist/normal 1.0 2.0) 0.0 3.0) 2.0) 0.7186932107354801)))
  (testing "outside the support"
    (is (= ##-Inf (dist/logpdf (dist/truncated (dist/normal 1.0 2.0) 0.0 3.0) 3.5)))
    (is (= ##-Inf (dist/logpdf (dist/zero-sum-normal 1.0 3) [1.0 0.0 0.0])))
    (is (= ##-Inf (dist/logpdf (dist/censored (dist/normal 0.0 1.0) -1.0 1.0) 1.5)))))

(defn- moments [d n]
  (let [xs (repeatedly n #(dist/draw d))
        m (/ (reduce + xs) n)]
    [m (/ (reduce + (map #(let [e (- % m)] (* e e)) xs)) n)]))

(deftest draws-match-moments
  (random/set-seed! 11)
  (doseq [[label d m tol] [["weibull" (dist/weibull 1.5 2.0) (dist/mean (dist/weibull 1.5 2.0)) 0.03]
                           ["laplace" (dist/laplace 0.5 1.3) 0.5 0.04]
                           ["inverse-gamma" (dist/inverse-gamma 5.0 2.0) 0.5 0.01]
                           ["gumbel" (dist/gumbel 1.0 2.0) (+ 1.0 (* 2.0 0.5772156649015329)) 0.05]
                           ["half-student-t" (dist/half-student-t 3.0 1.5) 1.6539866862653758 0.06]
                           ["beta-binomial" (dist/beta-binomial 10 2.0 3.0) 4.0 0.06]
                           ["zero-inflated" (dist/zero-inflated 0.3 (dist/poisson 2.5)) 1.75 0.04]
                           ["gamma by mean and sd" (dist/gamma-mean-sd 3.0 1.5) 3.0 0.04]]]
    (testing label
      (is (< (Math/abs (- (first (moments d 20000)) m)) tol) (str (first (moments d 20000))))))
  (testing "variances"
    (is (< (Math/abs (- (second (moments (dist/gamma-mean-sd 3.0 1.5) 20000)) 2.25)) 0.15))
    (is (< (Math/abs (- (second (moments (dist/beta-binomial 10 2.0 3.0) 20000))
                        (dist/variance (dist/beta-binomial 10 2.0 3.0)))) 0.25)))
  (testing "truncated and censored draws stay in bounds"
    (is (every? #(<= 0.0 % 3.0) (repeatedly 2000 #(dist/draw (dist/truncated (dist/normal 1.0 2.0) 0.0 3.0)))))
    (is (every? #(<= 5.0 % 6.0) (repeatedly 500 #(dist/draw (dist/truncated (dist/normal 0.0 1.0) 5.0 6.0)))))
    (is (every? #(<= -1.0 % 1.0) (repeatedly 2000 #(dist/draw (dist/censored (dist/normal 0.0 1.0) -1.0 1.0))))))
  (testing "hurdle draws a positive part that is never zero"
    (let [xs (repeatedly 20000 #(dist/draw (dist/hurdle 0.3 (dist/poisson 0.5))))]
      (is (< (Math/abs (- (/ (count (filter zero? xs)) 20000.0) 0.3)) 0.015))))
  (testing "zero-sum draws sum to zero"
    (is (every? #(< (Math/abs (reduce + %)) 1e-12) (repeatedly 100 #(dist/draw (dist/zero-sum-normal 2.0 5)))))))

(deftest finite-supports-enumerate
  (is (= (range 11) (dist/support (dist/beta-binomial 10 2.0 3.0))))
  (is (= [2 3 4] (dist/support (dist/truncated (dist/binomial 6 0.5) 2 4))))
  (is (= [0 1 2] (dist/support (dist/zero-inflated 0.2 (dist/binomial 2 0.5)))))
  (let [d (dist/truncated (dist/binomial 6 0.5) 2 4)]
    (is (close? 1.0 (reduce + (map #(Math/exp (dist/logpdf d %)) (dist/support d)))))))

(deftest edge-cases-from-the-review
  (testing "cdfs at infinity"
    (is (= 1.0 (dist/cdf (dist/normal 0.0 1.0) ##Inf)))
    (is (= 0.0 (dist/cdf (dist/normal 0.0 1.0) ##-Inf)))
    (is (= 1.0 (dist/cdf (dist/gamma 2.0 1.0) ##Inf))))
  (testing "small concentrations draw without 0/0"
    (random/set-seed! 1)
    (is (every? #(<= 0.0 % 1.0) (repeatedly 200 #(dist/draw (dist/beta 0.001 0.001)))))
    (is (every? #(< (Math/abs (- 1.0 (reduce + %))) 1e-9)
                (repeatedly 100 #(dist/draw (dist/dirichlet [0.001 0.001 0.001]))))))
  (testing "huge finite weights"
    (is (close? (dist/logpdf (dist/discrete [1e308 1e308]) 0) (Math/log 0.5)))
    (is (close? (dist/logpdf (dist/categorical {:a 1e308 :b 1e308}) :a) (Math/log 0.5))))
  (testing "uniform-discrete quantiles invert the cdf"
    (is (= [0 0 1 3 3] (mapv #(dist/quantile (dist/uniform-discrete 0 4) %) [0.0 0.25 0.3 0.99 1.0]))))
  (testing "mvn refuses an asymmetric covariance and a value of the wrong dimension"
    (is (thrown? clojure.lang.ExceptionInfo (dist/mvn [0.0 0.0] [[1.0 100.0] [0.0 1.0]])))
    (is (= ##-Inf (dist/logpdf (dist/mvn [0.0 0.0] [[1.0 0.0] [0.0 1.0]]) [0.0 0.0 999.0])))))
