;; foerster side of the comparison (W1-W4). Run from the foerster root:
;;   clojure -J-Xmx3g -M -i experiments/comparison/clj/bench.clj
;; Each workload runs twice; the warm (second) run is reported, the first
;; run's time too. Results: experiments/comparison/results/foerster.edn
(require '[clojure.string :as str]
         '[org.replikativ.foerster.block :as block]
         '[org.replikativ.foerster.core :as infer]
         '[org.replikativ.foerster.diagnostics :as d]
         '[org.replikativ.foerster.dist :as dist]
         '[org.replikativ.foerster.effects :refer [sample observe]]
         '[org.replikativ.foerster.measure :as m]
         '[org.replikativ.foerster.random :as random]
         '[org.replikativ.spindel.core :as sp]
         '[org.replikativ.spindel.spin.cps :refer [spin]])

(def dir "experiments/comparison/")
(def W (read-string (slurp (str dir "data/workloads.edn"))))
(def world (sp/create-execution-context))

(defn run* [seed f]
  (random/set-seed! seed)
  (let [t0 (System/nanoTime) r (sp/with-context world @(f))]
    [r (/ (- (System/nanoTime) t0) 1e9)]))

(defn twice [seed f]
  (let [[_ t1] (run* seed f) [r t2] (run* seed f)] [r t1 t2]))

(defn summ [measure f]
  (select-keys (d/summary measure f) [:mean :sd :ess-bulk :rhat]))

;; W1 random-walk particle filter
(def ys (:y (:rw W)))
(defn rw []
  (spin (loop [t 0 v (sample (dist/normal 0.0 1.0) :id [:v 0])]
          (observe (dist/normal v 1.0) (nth ys t) :id [:y t])
          (if (= t (dec (count ys))) v (recur (inc t) (sample (dist/normal v 0.1) :id [:v (inc t)]))))))

(defn w1 []
  (into {}
        (for [n [100 1000]]
          (let [_ (run* 99 #(infer/infer (rw) {:method :smc :particles n})) ; warm-up
                runs (vec (for [s (range (if (= n 100) 10 5))]
                            (let [[r t] (run* s #(infer/infer (rw) {:method :smc :particles n}))]
                              [t (- (m/log-marginal r) (:log-evidence (:rw W)))])))
                ts (sort (map first runs)) med (nth ts (quot (count ts) 2))
                errs (map second runs) mu (/ (reduce + errs) (count errs))]
            [(keyword (str "N" n))
             {:median_s med :steps_per_s (/ (* n (count ys)) med)
              :logZ_err_mean mu
              :logZ_err_sd (Math/sqrt (/ (reduce + (map #(let [e (- % mu)] (* e e)) errs)) (count errs)))}]))))

;; golf
(def golf (:golf W))
(defn sigmoid [z] (/ 1.0 (+ 1.0 (Math/exp (- z)))))
(defn logistic []
  (spin (let [a (sample (dist/normal 0.0 1.0) :id :a) b (sample (dist/normal 0.0 1.0) :id :b)]
          (loop [i 0] (when (< i 19)
                        (observe (dist/binomial (nth (:n golf) i) (sigmoid (+ a (* b (nth (:x golf) i)))))
                                 (nth (:y golf) i) :id [:putt i])
                        (recur (inc i))))
          {:a a :b b})))

(defn w3 []
  (let [[r t1 t2] (twice 1 #(infer/infer (logistic) {:method :rmh :iterations 22000 :burn 2000 :chains 4 :step-size 0.02}))]
    {:first_s t1 :warm_s t2 :a (summ r :a) :b (summ r :b)}))

(defn w4 []
  (let [[r t1 t2] (twice 1 #(infer/infer (logistic) {:method :tempered :particles 2000}))]
    {:first_s t1 :warm_s t2 :log_evidence (m/log-marginal r) :a (summ r :a) :b (summ r :b)}))

;; W2 radon: the notebook's block (hand-written gradient), NUTS 4 x (1000 + 1000)
(def radon
  (let [[_ & lines] (str/split-lines (slurp (str dir "data/radon.csv")))
        rows (mapv #(let [[c f y] (str/split % #",")] [(Long/parseLong c) (Long/parseLong f) (Double/parseDouble y)]) lines)]
    {:county (int-array (map first rows)) :floor (double-array (map second rows))
     :y (double-array (map #(nth % 2) rows)) :counties 85}))

(defn radon-value+grad [^doubles th {:keys [^ints county ^doubles floor ^doubles y counties]}]
  (let [mu-a (aget th 0) s-a (aget th 1) beta (aget th 2) s-y (aget th 3)
        n (alength y) g (double-array (alength th))
        lp (volatile! (+ (* -0.5 (/ (* mu-a mu-a) 100.0)) (- s-a) (* -0.5 (/ (* beta beta) 100.0)) (- s-y)))]
    (aset g 0 (/ (- mu-a) 100.0)) (aset g 1 -1.0) (aset g 2 (/ (- beta) 100.0)) (aset g 3 -1.0)
    (dotimes [c counties]
      (let [a (aget th (+ 4 c)) z (/ (- a mu-a) s-a)]
        (vswap! lp + (- (* -0.5 z z) (Math/log s-a)))
        (aset g (+ 4 c) (- (aget g (+ 4 c)) (/ z s-a)))
        (aset g 0 (+ (aget g 0) (/ z s-a)))
        (aset g 1 (+ (aget g 1) (/ (- (* z z) 1.0) s-a)))))
    (dotimes [i n]
      (let [c (aget county i) r (/ (- (aget y i) (aget th (+ 4 c)) (* beta (aget floor i))) s-y)]
        (vswap! lp + (- (* -0.5 r r) (Math/log s-y)))
        (aset g (+ 4 c) (+ (aget g (+ 4 c)) (/ r s-y)))
        (aset g 2 (+ (aget g 2) (/ (* r (aget floor i)) s-y)))
        (aset g 3 (+ (aget g 3) (/ (- (* r r) 1.0) s-y)))))
    [@lp g]))

(def radon-block
  (block/block {:block/id :radon :block/coordinates :constrained
                :block/latents [{:name :mu-a :shape []} {:name :sigma-a :shape [] :support :positive}
                                {:name :beta :shape []} {:name :sigma-y :shape [] :support :positive}
                                {:name :alpha :shape [85]}]
                :block/target :complete-conditional}
               {:log-density (fn [th inputs] (first (radon-value+grad th inputs))) :value+grad radon-value+grad}))

(defn w2 []
  (let [[r t1 t2] (twice 3 #(infer/infer (spin (sample (block/block-dist radon-block radon) :id :theta :init (vec (repeat 89 0.0))))
                                         {:method :nuts :iterations 2000 :burn 1000 :chains 4}))
        nat (fn [i] #(nth (block/constrain radon-block %) i))]
    {:first_s t1 :warm_s t2 :beta (summ r (nat 2)) :sigma_a (summ r (nat 1)) :sigma_y (summ r (nat 3))}))

(def results
  {:W1_rw_pf (assoc (w1) :system "foerster smc")
   :W2_radon_nuts (assoc (w2) :system "foerster NUTS (block, hand gradient)")
   :W3_golf_mh (assoc (w3) :system "foerster rmh")
   :W4_golf_smc (assoc (w4) :system "foerster tempered SMC")})

(spit (str dir "results/foerster.edn") (with-out-str (clojure.pprint/pprint results)))
(clojure.pprint/pprint results)
(shutdown-agents)
(System/exit 0)
