(ns org.replikativ.foerster.conjugate-test
  "Conjugate parameters carried as posteriors: the evidence of a stream of
  observations is the closed-form marginal likelihood (no parameter is
  sampled, so SMC's estimate is exact), and the parameter drawn from the
  final posterior has the closed-form posterior mean."
  (:require #?(:clj [clojure.test :refer [deftest is testing]]
               :cljs [cljs.test :refer-macros [deftest is testing]])
            [org.replikativ.foerster.conjugate :as conj]
            [org.replikativ.foerster.dist :as dist]))

(defn- close? [a b tol] (< (Math/abs (- a b)) tol))

(defn- chain
  "Σ log predictive of `ys` and the final posterior: what a program that
  observes under the predictive and updates computes."
  [c ys]
  (reduce (fn [[ll c] y] [(+ ll (dist/logpdf (conj/predictive c) y)) (conj/update c y)])
          [0.0 c] ys))

(def ^:private lgamma dist/lgamma)

(deftest the-predictives-chain-to-the-marginal-likelihood
  (testing "beta-bernoulli: the beta-binomial"
    (let [ys [1 0 1 1 1 0 1 1]
          k (count (filter #{1} ys)) n (count ys)
          lbeta (fn [a b] (- (+ (lgamma a) (lgamma b)) (lgamma (+ a b))))
          [ll c] (chain (conj/beta-bernoulli 2 3) ys)]
      (is (close? ll (- (lbeta (+ 2 k) (+ 3 (- n k))) (lbeta 2 3)) 1e-10))
      (is (close? (dist/mean (conj/posterior c)) (/ (+ 2 k) (+ 5 n)) 1e-12))))
  (testing "gamma-poisson: a negative binomial chain"
    (let [ys [3 0 2 5 1 4] a 2.0 th 1.5
          s (reduce + ys) n (count ys)
          exact (+ (- (lgamma (+ a s)) (lgamma a) (reduce + (map #(lgamma (inc %)) ys)))
                   (* s (Math/log th)) (- (* (+ a s) (Math/log (+ 1 (* n th))))))
          [ll c] (chain (conj/gamma-poisson a th) ys)]
      (is (close? ll exact 1e-10))
      (is (close? (dist/mean (conj/posterior c)) (* (+ a s) (/ th (+ 1 (* n th)))) 1e-12))))
  (testing "normal-mean"
    (let [ys [0.9 1.4 0.2 1.1] m0 0.0 s0 2.0 s 1.0
          n (count ys)
          ;; the marginal is Normal(m0·1, s²I + s0²·11ᵀ): by the chain rule
          prec (+ (/ 1 (* s0 s0)) (/ n (* s s)))
          post-mean (/ (/ (reduce + ys) (* s s)) prec)
          [ll c] (chain (conj/normal-mean m0 s0 s) ys)
          ;; exact log marginal in closed form
          ss (reduce + (map #(* % %) ys))
          exact (- (* -0.5 n (Math/log (* 2 Math/PI s s)))
                   (* 0.5 (Math/log (+ 1 (/ (* n s0 s0) (* s s))))))
          exact (- exact (* 0.5 (/ ss (* s s))) (* -0.5 (* post-mean post-mean prec)))]
      (is (close? ll exact 1e-10))
      (is (close? (dist/mean (conj/posterior c)) post-mean 1e-12))))
  (testing "dirichlet-discrete: the Dirichlet-multinomial"
    (let [ys [0 2 2 1 2 0] alpha [1.0 2.0 0.5]
          counts (mapv (fn [k] (count (filter #{k} ys))) (range 3))
          a (reduce + alpha) n (count ys)
          exact (+ (- (lgamma a) (lgamma (+ a n)))
                   (reduce + (map (fn [al c] (- (lgamma (+ al c)) (lgamma al))) alpha counts)))
          [ll c] (chain (conj/dirichlet-discrete alpha) ys)]
      (is (close? ll exact 1e-10))
      (is (close? (first (dist/mean (conj/posterior c))) (/ (+ 1.0 2) (+ a n)) 1e-12)))))

(deftest the-negative-binomial
  (let [d (dist/negative-binomial 2.5 0.4)]
    (is (close? (reduce + (map #(Math/exp (dist/logpdf d %)) (range 400))) 1.0 1e-9))
    (is (close? (reduce + (map #(* % (Math/exp (dist/logpdf d %))) (range 400))) (dist/mean d) 1e-6))))
