;; W1 in raster: the same bootstrap particle filter (propagate, reweight,
;; systematic resampling when ESS < N/2), as one compiled deftm. Run from the
;; foerster-raster checkout (which depends on raster):
;;   clojure -J-Xmx2g -M -i ../foerster-compare/experiments/comparison/raster/pf.clj
(ns comparison.raster-pf
  (:require [raster.core :refer [deftm]]
            [clojure.string :as str])
  (:import [java.util SplittableRandom]
           [java.util.random RandomGenerator]))

(deftm pf
  "log evidence of a bootstrap particle filter over ys; v, lw, w, out are
  scratch arrays of length n."
  [ys :- (Array double), n :- Long, rng :- RandomGenerator,
   v :- (Array double), lw :- (Array double), w :- (Array double), out :- (Array double)] :- Double
  (let [T (alength ys)
        c (* 0.5 (Math/log (* 2.0 Math/PI)))]
    (dotimes [i n] (aset v i (.nextGaussian rng)) (aset lw i 0.0))
    (loop [t 0 lz 0.0]
      (if (= t T)
        (let [m (loop [i 0 m Double/NEGATIVE_INFINITY] (if (= i n) m (recur (inc i) (Math/max m (aget lw i)))))
              s (loop [i 0 s 0.0] (if (= i n) s (recur (inc i) (+ s (Math/exp (- (aget lw i) m))))))]
          (+ lz m (Math/log (/ s n))))
        (do
          (when (> t 0)
            (dotimes [i n] (aset v i (+ (aget v i) (* 0.1 (.nextGaussian rng))))))
          (let [yt (aget ys t)]
            (dotimes [i n]
              (let [d (- yt (aget v i))] (aset lw i (- (aget lw i) (* 0.5 d d) c)))))
          (let [m (loop [i 0 m Double/NEGATIVE_INFINITY] (if (= i n) m (recur (inc i) (Math/max m (aget lw i)))))
                s (loop [i 0 s 0.0] (if (= i n) s
                                         (let [e (Math/exp (- (aget lw i) m))] (aset w i e) (recur (inc i) (+ s e)))))
                s2 (loop [i 0 s2 0.0] (if (= i n) s2 (let [x (/ (aget w i) s)] (recur (inc i) (+ s2 (* x x))))))]
            (if (< (/ 1.0 s2) (* 0.5 n))
              (let [u0 (.nextDouble rng)]
                ;; systematic resampling into out, then copy back
                (loop [i 0 j 0 cum (/ (aget w 0) s)]
                  (when (< i n)
                    (let [u (/ (+ u0 i) n)]
                      (if (and (< cum u) (< j (dec n)))
                        (recur i (inc j) (+ cum (/ (aget w (inc j)) s)))
                        (do (aset out i (aget v j)) (recur (inc i) j cum))))))
                (dotimes [i n] (aset v i (aget out i)) (aset lw i 0.0))
                (recur (inc t) (+ lz m (Math/log (/ s n)))))
              (recur (inc t) lz))))))))

(def ys
  (let [txt (slurp "../foerster-compare/experiments/comparison/data/workloads.edn")]
    (double-array (:y (:rw (read-string txt))))))
(def exact (:log-evidence (:rw (read-string (slurp "../foerster-compare/experiments/comparison/data/workloads.edn")))))

(defn bench [n reps]
  (let [run (fn [seed] (pf ys n (SplittableRandom. seed) (double-array n) (double-array n) (double-array n) (double-array n)))]
    (dotimes [s 20] (run s))                              ; JIT warm-up
    (let [rs (vec (for [s (range reps)]
                    (let [t0 (System/nanoTime) lz (run (+ 100 s))] [(/ (- (System/nanoTime) t0) 1e9) (- lz exact)])))
          ts (sort (map first rs)) med (nth ts (quot reps 2))
          errs (map second rs) mu (/ (reduce + errs) reps)]
      {:median_s med :steps_per_s (/ (* n (alength ys)) med) :logZ_err_mean mu
       :logZ_err_sd (Math/sqrt (/ (reduce + (map #(let [e (- % mu)] (* e e)) errs)) reps))})))

(let [r {:W1_rw_pf {:system "raster deftm bootstrap PF" :N100 (bench 100 20) :N1000 (bench 1000 20)}}]
  (spit "../foerster-compare/experiments/comparison/results/raster.edn" (pr-str r))
  (prn r))
(shutdown-agents)
(System/exit 0)
