(ns org.replikativ.foerster.block-test
  "Constraint transforms of block latents: supports checked at definition,
  round trips, and log-Jacobians and gradients against finite differences."
  (:require #?(:clj [clojure.test :refer [deftest is testing]]
               :cljs [cljs.test :refer-macros [deftest is testing]])
            [org.replikativ.foerster.block :as block]))

(defn- unknown-support [description]
  (try (block/block description {:log-density (fn [_ _] 0.0) :value+grad (fn [_ _] [0.0 []])})
       nil
       (catch #?(:clj clojure.lang.ExceptionInfo :cljs :default) e
         (:type (ex-data e)))))

(deftest supports-are-checked-when-the-block-is-defined
  (doseq [[latent why] [[{:name :p :shape [] :support :unit-interval} "an unknown keyword"]
                        [{:name :p :shape [] :support [:interval 1.0 0.0]} "an empty interval"]
                        [{:name :p :shape [] :support [:interval 0.0]} "an interval without its upper bound"]
                        [{:name :p :shape [] :support [:lower]} "a lower bound without its value"]
                        [{:name :p :shape [] :support [:upper ##Inf]} "an infinite bound"]
                        [{:name :p :shape [] :support :simplex} "a scalar simplex"]
                        [{:name :p :shape [1] :support :simplex} "a one-point simplex"]
                        [{:name :p :shape [2 2] :support :ordered} "a matrix ordered"]]]
    (is (= ::block/unknown-support
           (unknown-support {:block/id :b :block/coordinates :constrained :block/latents [latent]}))
        why))
  (is (nil? (unknown-support {:block/id :b :block/coordinates :constrained
                              :block/latents [{:name :a :shape [] :support [:lower -1.0]}
                                              {:name :b :shape [2] :support [:upper 3]}
                                              {:name :c :shape [3] :support :ordered}
                                              {:name :d :shape [4] :support :simplex}]}))))

;; One latent of every support, with a smooth density in natural
;; coordinates, f(x) = Σ cᵢxᵢ − ½ Σ xᵢ²/10, and its gradient.

(def ^:private latents
  [{:name :r :shape [] :support :real}
   {:name :pos :shape [] :support :positive}
   {:name :lo :shape [] :support [:lower 3.0]}
   {:name :up :shape [] :support [:upper -1.0]}
   {:name :iv :shape [] :support [:interval -2.0 5.0]}
   {:name :ord :shape [3] :support :ordered}
   {:name :p :shape [4] :support :simplex}])

(def ^:private n-x 12)
(def ^:private n-theta 11)
(def ^:private cs (mapv #(- (* 0.3 %) 1.1) (range n-x)))

(defn- f [xs] (reduce + (map (fn [c x] (- (* c x) (* 0.05 x x))) cs xs)))
(defn- f-grad [xs] (mapv (fn [c x] (- c (* 0.1 x))) cs xs))

(def ^:private every-support
  (block/block {:block/id :every :block/coordinates :constrained :block/latents latents}
               {:log-density (fn [x _] (f (vec x)))
                :value+grad (fn [x _] [(f (vec x)) (f-grad (vec x))])}))

(def ^:private thetas
  ;; deterministic points, mild and far into the tails
  [(vec (repeat n-theta 0.0))
   (mapv #(* 0.37 (- % 5)) (range n-theta))
   (mapv #(Math/sin (* 1.7 %)) (range n-theta))
   (mapv #(* 3.0 (Math/cos (* 0.9 %))) (range n-theta))])

(defn- close? [tol a b] (<= (Math/abs (- (double a) (double b))) (* tol (max 1.0 (Math/abs (double a)) (Math/abs (double b))))))

(deftest constrain-and-unconstrain-are-inverse
  (is (= n-theta (block/dimension every-support)))
  (doseq [theta thetas]
    (let [x (block/constrain every-support theta)
          [r pos lo up iv o1 o2 o3 & p] x]
      (is (= n-x (count x)))
      (is (pos? pos))
      (is (> lo 3.0))
      (is (< up -1.0))
      (is (< -2.0 iv 5.0))
      (is (< o1 o2 o3))
      (is (every? pos? p))
      (is (close? 1e-12 1.0 (reduce + p)))
      (is (= r (first theta)))
      (is (every? true? (map (partial close? 1e-9) theta (block/unconstrain every-support x)))
          (str theta))
      (is (= p (block/latent every-support x :p)))
      (is (= [o1 o2 o3] (block/latent every-support x :ord))))))

(defn- determinant
  "det of the square matrix `m` (rows) by Gaussian elimination."
  [m]
  (let [n (count m)]
    (loop [m (mapv vec m) k 0 det 1.0]
      (if (= k n)
        det
        (let [pivot (apply max-key #(Math/abs (double (get-in m [% k]))) (range k n))
              m (if (= pivot k) m (assoc m k (m pivot) pivot (m k)))
              det (if (= pivot k) det (- det))
              p (get-in m [k k])]
          (if (zero? p)
            0.0
            (recur (reduce (fn [m i]
                             (let [factor (/ (get-in m [i k]) p)]
                               (assoc m i (mapv - (m i) (map #(* factor %) (m k))))))
                           m (range (inc k) n))
                   (inc k) (* det p))))))))

(defn- numeric-log-jacobian
  "log |det ∂x/∂θ| by central differences, x the first `n-theta` latent
  coordinates (a simplex's last coordinate is determined by the others)."
  [theta]
  (let [h 1e-6
        x-of #(subvec (block/constrain every-support %) 0 n-theta)
        cols (for [j (range n-theta)]
               (mapv #(/ (- %1 %2) (* 2.0 h))
                     (x-of (update theta j + h)) (x-of (update theta j - h))))]
    ;; the latents are ordered so that the dropped coordinate is the
    ;; simplex's last one: the remaining square block is ∂x/∂θ
    (Math/log (Math/abs (double (determinant (apply mapv vector cols)))))))

(deftest log-jacobians-match-finite-differences
  (doseq [theta thetas]
    (let [[lj _] (block/jacobian every-support theta)]
      (is (close? 1e-6 (numeric-log-jacobian theta) lj) (str theta " " lj)))))

(deftest gradients-match-finite-differences
  ;; value+grad in θ (the transforms' chain rule and log-Jacobian gradient)
  ;; against central differences of log-density in θ; jacobian's gradient
  ;; against central differences of its value
  (let [vg (block/capability every-support :value+grad)
        ld (block/capability every-support :log-density)
        h 1e-6]
    (doseq [theta thetas]
      (let [[lp g] (vg (double-array theta) nil)
            [_ gj] (block/jacobian every-support theta)]
        (is (close? 1e-12 lp (ld (double-array theta) nil)))
        (doseq [j (range n-theta)]
          (let [at #(double-array (update theta j + %))
                numeric (/ (- (ld (at h) nil) (ld (at (- h)) nil)) (* 2.0 h))
                numeric-j (/ (- (first (block/jacobian every-support (vec (at h))))
                                (first (block/jacobian every-support (vec (at (- h))))))
                             (* 2.0 h))]
            (is (close? 1e-6 numeric (aget ^doubles g j)) (str "∇ " j " at " theta))
            (is (close? 1e-6 numeric-j (nth gj j)) (str "∇log|J| " j " at " theta))))))))

(deftest a-simplex-is-stans-stick-breaking
  ;; θ = 0 is the uniform point (Stan's offsets log(K − 1 − j))
  (let [b (block/block {:block/id :s :block/coordinates :constrained
                        :block/latents [{:name :p :shape [5] :support :simplex}]}
                       {:log-density (fn [_ _] 0.0) :value+grad (fn [_ _] [0.0 [0.0 0.0 0.0 0.0 0.0]])})]
    (is (every? #(close? 1e-12 0.2 %) (block/constrain b [0.0 0.0 0.0 0.0])))
    (testing "and far into the tails it stays on the simplex"
      (let [x (block/constrain b [40.0 -40.0 30.0 -30.0])]
        (is (every? #(<= 0.0 %) x))
        (is (close? 1e-12 1.0 (reduce + x)))))))
