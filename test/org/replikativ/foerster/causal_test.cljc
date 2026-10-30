(ns org.replikativ.foerster.causal-test
  (:require [clojure.test :refer [deftest is testing]]
            [org.replikativ.foerster.causal :as c]))

(defn- assignments [nodes]
  (if-let [[v & more] (seq nodes)]
    (for [a (assignments more) x [0 1]] (assoc a v x)) [{}]))

(defn- joint-table [order laws interventions]
  (mapv (fn [a]
          [a (reduce * (map (fn [v]
                              (if (contains? interventions v)
                                (if (= (a v) (interventions v)) 1 0)
                                (let [p ((laws v) a)] (if (= 1 (a v)) p (- 1 p))))) order))])
        (assignments order)))

(defn- mass [table event]
  (reduce + (for [[a w] table :when (every? (fn [[v x]] (= x (a v))) event)] w)))

(defn- oracle [table]
  (fn [event given] (/ (mass table (merge event given)) (mass table given))))

(deftest graph-validation-and-separation
  (is (thrown? #?(:clj Exception :cljs js/Error)
               (c/graph {:directed [[:x :y] [:y :x]]})))
  (let [g (c/graph {:directed [[:x :z] [:y :z]]})]
    (is (c/m-separated? g #{:x} #{:y} #{}))
    (is (not (c/m-separated? g #{:x} #{:y} #{:z}))))
  (is (not (c/m-separated? (c/graph {:bidirected [[:x :y]]}) #{:x} #{:y} #{}))))

(deftest observational-functionals-match-intervention-ground-truth
  ;; Compute observational and interventional laws independently by finite
  ;; SCM enumeration. In the front-door example H is unobserved.
  (doseq [{:keys [order laws graph]}
          [{:order [:z :x :y]
            :laws {:z (constantly (/ 2 5)) :x #(+ (/ 1 5) (* (/ 3 5) (:z %)))
                   :y #(+ (/ 1 10) (* (/ 2 5) (:x %)) (* (/ 3 10) (:z %)))}
            :graph {:directed [[:z :x] [:z :y] [:x :y]]}}
           {:order [:h :x :z :y]
            :laws {:h (constantly (/ 2 5)) :x #(+ (/ 1 5) (* (/ 3 5) (:h %)))
                   :z #(+ (/ 1 10) (* (/ 7 10) (:x %)))
                   :y #(+ (/ 1 10) (* (/ 2 5) (:z %)) (* (/ 3 10) (:h %)))}
            :graph {:directed [[:x :z] [:z :y]] :bidirected [[:x :y]]}}]]
    (let [g (c/graph graph) observational (joint-table order laws {})
          identified (c/identify g {:outcomes #{:y} :treatments #{:x}})]
      (is (= :identified (:status identified)))
      (doseq [x [0 1] y [0 1]]
        (is (< (Math/abs (double (- (mass (joint-table order laws {:x x}) {:y y})
                                    (c/evaluate (:expression identified) (zipmap order (repeat [0 1]))
                                                (oracle observational) {:x x :y y})))) 1e-12))))))

(deftest nonidentification-is-distinct-from-conditional-identification
  (let [bow (c/graph {:directed [[:x :z]] :bidirected [[:x :z]]})
        result (c/identify bow {:outcomes #{:z} :treatments #{:x}})]
    (is (= :not-identified (:status result)))
    (is (map? (:witness result))))
  ;; P_x(z,y) is not identified, but P_x(y|z) is. IDC must not just
  ;; normalize a failed joint identification.
  (let [g (c/graph {:directed [[:x :z] [:z :y]] :bidirected [[:x :z]]})
        result (c/identify g {:outcomes #{:y} :treatments #{:x} :given #{:z}})]
    (is (= :identified (:status result)))
    (let [order [:h :x :z :y]
          laws {:h (constantly (/ 2 5)) :x #(+ (/ 1 5) (* (/ 3 5) (:h %)))
                :z #(+ (/ 1 10) (* (/ 2 5) (:x %)) (* (/ 3 10) (:h %)))
                :y #(+ (/ 1 5) (* (/ 3 5) (:z %)))}
          p (oracle (joint-table order laws {}))]
      (is (< (Math/abs (double (- (/ 4 5) (c/evaluate (:expression result) (zipmap order (repeat [0 1]))
                                                      p {:x 0 :z 1 :y 1})))) 1e-12)))))

(deftest disconnected-and-isolated-nodes
  (let [g (c/graph {:nodes #{:x :y} :directed []})]
    (is (= {:op :probability :vars [:y] :given []}
           (:expression (c/identify g {:outcomes #{:y} :treatments #{:x}}))))))

(deftest identified-functionals-agree-across-a-family-of-latent-scms
  ;; All eight forward DAGs on X,Z,Y, with all eight possible subsets
  ;; affected by an unobserved root H. Every returned ID/IDC functional is
  ;; evaluated against an independently intervened joint distribution.
  (let [powers (fn [xs] (reduce (fn [ss x] (into ss (map #(conj % x) ss))) [[]] xs))
        positive (atom 0) negative (atom 0)]
    (doseq [edges (powers [[:x :z] [:x :y] [:z :y]]) hidden (powers [:x :z :y])]
      (let [g (c/graph {:nodes #{:x :z :y} :directed edges
                        :bidirected (for [a hidden b hidden :when (neg? (compare a b))] [a b])})
            all-edges (concat edges (map #(vector :h %) hidden))
            laws (into {:h (constantly 0.4)}
                       (for [node [:x :z :y]
                             :let [parents (map first (filter #(= node (second %)) all-edges))]]
                         [node (fn [a] (+ 0.1 (* 0.2 (reduce + (map a parents)))))]))
            observational (joint-table [:h :x :z :y] laws {})]
        (doseq [given [#{} #{:z}]
                :let [r (c/identify g {:outcomes #{:y} :treatments #{:x} :given given})]]
          (if (= :not-identified (:status r))
            (swap! negative inc)
            (do
              (swap! positive inc)
              (doseq [x [0 1] y [0 1] z (if (seq given) [0 1] [nil])
                      :let [intervened (joint-table [:h :x :z :y] laws {:x x})
                            observed (if (seq given) {:z z} {})
                            truth (/ (mass intervened (assoc observed :y y)) (mass intervened observed))
                            estimate (c/evaluate (:expression r) (zipmap [:x :z :y] (repeat [0 1]))
                                                 (oracle observational) (merge {:x x :y y} observed))]]
                (is (< (Math/abs (- truth estimate)) 1e-10)
                    (pr-str {:graph g :given given :expression (:expression r)}))))))))
    (is (pos? @positive))
    (is (pos? @negative))))
