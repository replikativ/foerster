(ns org.replikativ.foerster.proposal-mh-test
  "MH with a joint proposal program against an exact Gaussian posterior:
  x, y ~ N(0, 1), z = 2 ~ N(x + y, 0.5). The posterior is N([8/9 8/9],
  [[5 −4] [−4 5]]/9): x and y strongly anti-correlated, so a proposal that
  moves them together along x + y = const mixes where single sites crawl."
  (:require [clojure.test :refer [deftest is]]
            [org.replikativ.foerster.diagnostics :as d]
            [org.replikativ.foerster.dist :as dist]
            [org.replikativ.foerster.effects :refer [observe sample]]
            [org.replikativ.foerster.gfi :as gfi]
            [org.replikativ.foerster.random :as random]
            [org.replikativ.spindel.engine.context :as ctx]
            [org.replikativ.spindel.engine.core :as rtc]
            [org.replikativ.spindel.spin.cps :refer [spin]]))

(defn- model []
  (spin
   (let [x (sample (dist/normal 0.0 1.0) :id :x)
         y (sample (dist/normal 0.0 1.0) :id :y)]
     (observe (dist/normal (+ x y) 0.5) 2.0 :id :z)
     [x y])))

(defn- along-the-ridge
  "x' ~ N(x, 0.6), then y' ~ N(y − (x' − x), 0.15): a joint move."
  [{:keys [x y]}]
  (spin
   (let [x' (sample (dist/normal x 0.6) :id :x)]
     (sample (dist/normal (- y (- x' x)) 0.15) :id :y))))

(defn- await* [op]
  (let [p (promise)] (op #(deliver p %) #(deliver p %))
       (let [v (deref p 60000 ::timeout)] (if (instance? Throwable v) (throw v) v))))

(defn- run-chain [seed n burn]
  (random/set-seed! seed)
  (let [root (ctx/create-execution-context)]
    (try
      (binding [rtc/*execution-context* root]
        (let [t0 (:trace (await* (gfi/generate (model) {:x 0.0 :y 0.0})))]
          (loop [i 0 t t0 out [] acc 0]
            (if (= i n)
              {:draws out :acceptance (/ acc (double n))}
              (let [{t' :trace a :accepted?} (await* (gfi/mh-proposal t along-the-ridge))]
                (recur (inc i) t' (if (>= i burn) (conj out (:trace/result t')) out) (if a (inc acc) acc)))))))
      (finally (ctx/stop-context! root)))))

(deftest a-joint-proposal-reaches-the-exact-posterior
  (let [runs (mapv #(run-chain % 4000 500) [1 2 3 4])
        xs (mapv (fn [r] (mapv first (:draws r))) runs)
        draws (mapcat :draws runs)
        mean (fn [f] (/ (reduce + (map f draws)) (count draws)))
        mx (mean first) my (mean second)
        vx (mean #(let [e (- (first %) mx)] (* e e)))
        cxy (mean #(* (- (first %) mx) (- (second %) my)))]
    (is (every? #(< 0.2 (:acceptance %)) runs) (str (mapv :acceptance runs)))
    (is (< (d/rhat xs) 1.01) "the chains agree")
    (is (< (Math/abs (- mx (/ 8.0 9))) (* 4 (d/mcse xs))) (str "E[x] " mx))
    (is (< (Math/abs (- my (/ 8.0 9))) 0.06) (str "E[y] " my))
    (is (< (Math/abs (- vx (/ 5.0 9))) 0.06) (str "Var[x] " vx))
    (is (< (Math/abs (- cxy (/ -4.0 9))) 0.06) (str "Cov[x,y] " cxy))))
