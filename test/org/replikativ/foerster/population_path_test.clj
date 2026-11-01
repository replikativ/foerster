(ns org.replikativ.foerster.population-path-test
  (:require [clojure.test :refer [deftest is testing]]
            [is.simm.partial-cps.async :as pcps]
            [is.simm.partial-cps.runtime :as cps-runtime]
            [org.replikativ.foerster.execution :as execution]
            [org.replikativ.foerster.population :as pop]
            [org.replikativ.foerster.population.path :as path]
            [org.replikativ.foerster.weights :as weights]))

(deftest retention-conserves-parent-mass
  (let [population {:particles [:a :b] :mass (weights/uniform 2)}
        result (path/retain population [[:a1] [:b1 :b2 :b3]] false)]
    (is (= [:a1 :b1 :b2 :b3] (:particles result)))
    (is (= [0.5 0.16666666666666669 0.16666666666666669 0.16666666666666669]
           (mapv #(Math/exp %) (get-in result [:mass :log-weights]))))
    (is (= 0.0 (get-in result [:mass :log-mass]))))
  (testing "Waste-free retention keeps initial states and canonical uniform mass"
    (let [population {:particles [:a :b] :mass (weights/uniform 2)}
          result (path/retain population [[:a1] [:b1]] true)]
      (is (= [:a :a1 :b :b1] (:particles result)))
      (is (= (weights/uniform 4) (:mass result))))))

(defn- synthetic-run [stages waste-free]
  (path/run {:particles [0 0 0 0] :mass (weights/uniform 4) :position 0}
            {:done? #(= stages %) :advance (fn [ps _ p] {:pos (inc p) :log-incr (vec (repeat (count ps) 0.25))})
             :exhausted #(ex-info "exhausted" {:position %})}
            {:move (fn [starts _ _ moves]
                     (pcps/async {:chains (mapv #(vec (take moves (iterate inc (inc %)))) starts)}))}
            {:n 4 :waste-free waste-free :seed 42 :keys {:resample ::selection} :max-steps (inc stages)}))

(deftest shared-path-synchronous-and-nested-trampoline
  (doseq [waste-free [nil 2]]
    (let [result (path/run-sync (synthetic-run 2000 waste-free))]
      (is (= 4 (count (:particles result))))
      (is (= 2000 (:steps result)))
      (is (< (Math/abs (- 500.0 (get-in result [:mass :log-mass]))) 1e-10))))
  (is (= 3 (:steps (path/run-sync (pcps/async (path/run-sync (synthetic-run 3 nil))))))))

(deftest asynchronous-backend-is-explicit
  (is (thrown-with-msg? clojure.lang.ExceptionInfo #"suspended"
                        (path/run-sync (fn [_ _] nil)))))

(deftest parallel-fails-fast-and-preserves-order
  (let [started (atom []) error (ex-info "start failed" {})]
    (is (identical? error
                    (try (path/run-sync (execution/parallel nil 4
                                                            (fn [i resolve _]
                                                              (swap! started conj i)
                                                              (if (= 1 i) (throw error) (resolve i)))))
                         (catch Throwable e e))))
    (is (= [0 1] @started)))
  (is (= [0 1 2] (path/run-sync (execution/parallel nil 3
                                                    (fn [i resolve reject]
                                                      ((pcps/async i) resolve reject)))))))

(deftest suspended-phases-resume-the-same-path
  (let [pending (atom nil)
        result (atom nil)
        error (atom nil)
        drive (fn [operation]
                (loop [t (operation)]
                  (when (cps-runtime/thunk? t) (recur (cps-runtime/force-thunk t)))))
        operation (path/run
                   {:particles [0 1] :mass (weights/uniform 2) :position 0}
                   {:done? #(= 2 %) :advance (fn [_ _ p] {:pos (inc p) :log-incr [0.5 0.5]})
                    :exhausted #(ex-info "exhausted" {:position %})}
                   {:move (fn [starts _ _ _]
                            (fn [resolve _]
                              (reset! pending #(resolve {:chains (mapv (fn [x] [(inc x)]) starts)}))))}
                   {:seed 7 :keys {:resample ::selection}})]
    (drive #(operation (partial reset! result) (partial reset! error)))
    (is (nil? @result))
    (drive @pending)
    (is (nil? @result))
    (drive @pending)
    (is (nil? @error))
    (is (= 2 (:steps @result)))
    (is (= 1.0 (get-in @result [:mass :log-mass])))))

(deftest a-chain-carries-its-density-from-walk-to-walk
  ;; A 2-d Gaussian path in four steps, waste-free with P = 5: eight chains of
  ;; four walks of two random-walk steps per SMC step. A chain evaluates the
  ;; density at its start state once, then carries it, so each step costs
  ;; 8·(3 + 3·2) evaluations instead of 8·4·3; the draws are unchanged (the
  ;; population and evidence recorded on origin/main 24d9c1a).
  (let [evaluations (atom 0)
        path {:start 0.0 :done? #(>= % 1.0)
              :advance (fn [qs _ beta]
                         (let [beta' (min 1.0 (+ beta 0.25))]
                           {:pos beta' :log-incr (mapv (fn [[x y]] (* (- beta' beta) -0.5 (+ (* (- x 1) (- x 1)) (* 4 (- y 2) (- y 2))))) qs)}))
              :density (fn [beta] (fn [[x y]]
                                    (swap! evaluations inc)
                                    (+ (* -0.5 (+ (* x x) (* y y)))
                                       (* beta -0.5 (+ (* (- x 1) (- x 1)) (* 4 (- y 2) (- y 2)))))))
              :exhausted (fn [_] (ex-info "exhausted" {}))}
        run (fn [waste-free]
              (reset! evaluations 0)
              (pop/theta-smc (mapv (fn [i] [(Math/sin i) (Math/cos (* 3 i))]) (range 40)) path
                             {:n 40 :waste-free waste-free :scale 2.38 :seed [77] :max-steps 10
                              :stats (atom {:moves 0 :accepted 0})
                              :keys {:resample :ws5.baseline/r :move :ws5.baseline/m}}))]
    (let [{:keys [qs log-z]} (run 5)]
      (is (= 288 @evaluations))
      (is (= 2146367930 (hash qs)))
      (is (= -3.154981289962682 log-z)))
    (testing "one walk per chain evaluates its start state, as before"
      (let [{:keys [qs log-z]} (run nil)]
        (is (= (* 4 40 3) @evaluations))
        (is (= -2007510409 (hash qs)))
        (is (= -3.2636502794737403 log-z))))))
