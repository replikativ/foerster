(ns org.replikativ.foerster.incremental-api-test
  (:refer-clojure :exclude [await])
  (:require #?(:clj [clojure.test :refer [deftest is]]
               :cljs [cljs.test :refer-macros [deftest is async]])
            #?(:clj [org.replikativ.foerster.benchmark-test :as b])
            [org.replikativ.foerster.core :as infer]
            [org.replikativ.foerster.dist :as d]
            [org.replikativ.foerster.effects :refer [sample observe]]
            #?(:clj [org.replikativ.foerster.runtime.gfi :as gfi])
            [org.replikativ.foerster.moves.kernel :as k]
            [org.replikativ.foerster.measure :as m]
            [org.replikativ.foerster.program :as ip]
            #?(:clj [org.replikativ.foerster.runtime.trace :as it])
            [org.replikativ.foerster.moves.transition :as transition]
            [org.replikativ.foerster.runtime.async :as async :refer [await-op]]
            [org.replikativ.spindel.v2.core :as v2 :refer [task]]
            [org.replikativ.spindel.spin.keyed :refer [keyed]])
  #?(:cljs (:require-macros [org.replikativ.spindel.v2.core :as v2 :refer [task]])))

(defn- coin [i]
  (task (let [x (sample (d/flip 0.3) :id [:coin i])]
          (observe (d/flip (if x 0.9 0.2)) true :id [:observed i])
          x)))

(defn- keyed-model []
  (task (await-op (keyed :coins [0 1] coin))))

#?(:clj
   (do
     (defn- await-cps [operation]
       (let [result (promise)]
         ((async/op-of operation) #(deliver result [:ok %]) #(deliver result [:error %]))
         (let [[status value] (deref result 60000 [:timeout nil])]
           (case status :ok value :error (throw value)
                 (throw (ex-info "Incremental inference timed out" {}))))))

     (deftest ordinary-public-mh-infers-keyed-only-models
  ;; Neither latent has a replay anchor. Public MH must discover keyed
  ;; ownership rather than requiring callers to select another driver.
       (doseq [make [(fn [] (infer/infer (fn [] (keyed-model)) {:method :mh :iterations 2400 :chains 2
                                                        :burn 400 :seed 401}))
                     (fn [] (infer/kernel-infer (fn [] (keyed-model))
                                                (k/single-site-mh-kernel 2400 {:samples :all :burn 400})
                                                2 {:seed 402}))]]
         (let [measure (b/run-infer 400 make)
               samples (b/weighted-values measure)]
           (is (= 4000 (count samples)))
           (is (< (Math/abs (- (b/w-mean #(if (first %) 1.0 0.0) samples) (/ 0.27 0.41))) 0.1)))))

     (deftest finite-gibbs-composes-with-the-public-mh-driver
       (let [kernel (k/cycle 500 [(k/finite-gibbs-kernel 1 [:coin 0])
                                  (k/single-site-mh-kernel 1)] {:samples :all :burn 50})
             measure (b/run-infer 403 #(infer/infer (fn [] (keyed-model)) {:method :kernel :kernel kernel
                                                                   :chains 2 :seed 403}))
             samples (b/weighted-values measure)]
         (is (k/markov-kernel? (k/finite-gibbs-kernel 1 [:coin 0])))
         (is (= :finite-gibbs (k/transition-id (k/finite-gibbs-kernel 1 [:coin 0]))))
         (is (true? (get-in (transition/prepare kernel) [:capabilities :tempering?])))
         (is (= 900 (count samples)))
         ;; seed-to-seed SD of each coin's estimate over 12 seeds: 0.043
         (doseq [i [0 1]]
           (is (< (Math/abs (- (b/w-mean #(if (nth % i) 1.0 0.0) samples) (/ 0.27 0.41))) 0.16)))))

     (deftest keyed-populations-can-advance-copy-fork-and-move-without-mutating-the-source
       (let [before (atom nil) after (atom nil) masses (atom nil)
             result
             (b/run-infer
              404
              #(ip/run {:seed 404}
                       (fn [scope]
                         (task
                          (let [source (await-op (ip/initialize scope (ip/trace-target (fn [] (keyed-model))) 8))
                                _ (reset! before (mapv it/choices (:particles source)))
                                child (await-op (ip/fork scope source))
                                child (ip/advance scope child (ip/temperature (:target child) 0.5))
                                child (await-op (ip/resample scope child))
                                initial-mass (get-in child [:mass :log-mass])
                                child (await-op (ip/move scope child
                                                      (k/cycle 3 [(k/finite-gibbs-kernel 1 [:coin 0])
                                                                  (k/single-site-mh-kernel 1)])))
                                _ (reset! after (mapv it/choices (:particles source)))
                                _ (reset! masses [initial-mass (get-in child [:mass :log-mass])])
                            ;; Source worlds remain usable after the copied child
                            ;; moved and reclaimed its own intermediate states.
                                source (await-op (ip/move scope source (k/finite-gibbs-kernel 1 [:coin 1])))
                                child-result (ip/finish scope child)]
                            (ip/finish scope source)
                            child-result)))))]
         (is (= @before @after))
         (is (apply = @masses) "invariant moves preserve population mass")
         (is (= 8 (count (m/get-particles result))))
         (is (= [:advance :resample :move] (mapv :operation (:history result))))))

     (deftest gibbs-with-source-retention-leaves-the-source-available
       (let [root (v2/world)]
         (try
           (binding [v2/*world* root]
             (let [source (:trace (await-cps (gfi/generate (fn [] (keyed-model)) {} {:seed 405})))
                   original (it/choices source)]
               (try
                 (await-cps (it/update-gibbs source [:coin 0] {:iteration 0 :keep-old? true}))
                 (is (= original (it/choices source)))
            ;; A second independent move from the retained source must still
            ;; resolve, even if the first candidate displaced its chosen state.
                 (let [again (await-cps (it/update-gibbs source [:coin 1]
                                                         {:iteration 1 :keep-old? true}))]
                   (is (= #{[:coin 0] [:coin 1]} (set (keys (it/choices (:trace again)))))))
                 (finally (await-cps (gfi/close! source))))))
           (finally (v2/release-world! root)))))

     (defn- nested-coin [calls group i]
       (task (swap! calls update [group i] (fnil inc 0))
             (let [x (sample (d/flip 0.3) :id [:nested group i])]
               (observe (d/flip (if x 0.9 0.2)) true :id [:nested-observed group i])
               x)))

     (defn- nested-group [calls group]
       (task (await-op (keyed :items [0 1] (fn [i] (nested-coin calls group i))))))

     (defn- nested-model [calls]
       (task (await-op (keyed :groups [0] (fn [group] (nested-group calls group))))))

     (deftest finite-gibbs-reruns-only-the-owning-nested-key
       (let [root (v2/world) calls (atom {})]
         (try
           (binding [v2/*world* root]
             (let [source (:trace (await-cps (gfi/generate (fn [] (nested-model calls)) {} {:seed 406})))
                   original (it/choices source)
                   before @calls]
               (try
                 (let [prepared (transition/prepare (k/finite-gibbs-kernel 1 [:nested 0 1]))
                       result (await-cps (transition/step prepared source {:keep-old? true :temperature 1.0}))
                       chosen (:trace result)]
                   (is (= 1 (:moves result)))
                   (is (= 1 (:accepted result)))
                   (is (= {[0 1] 1} (into {} (keep (fn [[site n]]
                                                     (let [delta (- n (get before site 0))]
                                                       (when (pos? delta) [site delta])))) @calls)))
                   (is (= (dissoc original [:nested 0 1])
                          (dissoc (it/choices chosen) [:nested 0 1])))
                   (is (= original (it/choices source)))
              ;; Both inner and outer keyed aggregations must see the new slot.
                   (is (= (mapv (fn [group] (mapv #(get (it/choices chosen) [:nested group %]) [0 1])) [0])
                          (:trace/result chosen))))
                 (finally (await-cps (gfi/close! source))))))
           (finally (v2/release-world! root)))))))

#?(:clj
   (do
     (defn- conditional-collections []
       (task
        (let [prefix (sample (d/flip 0.5) :id :prefix)
              switches (await-op (keyed :switches [0]
                                     (fn [_] (task (sample (d/flip 0.5) :id :switch)))))]
          (when (first switches)
            (await-op (keyed :later [0]
                          (fn [_] (task (observe (d/flip 0.8) true :id :later-observation))))))
          [prefix (first switches)])))

     (deftest changing-an-earlier-collection-replays-the-prefix-before-dropping-a-later-one
       (let [root (v2/world)]
         (try
           (binding [v2/*world* root]
             (let [draw-switch (fn [value]
                                 (fn [site _]
                                   (when (= :switch (:savepoint/address site))
                                     {:value value :symmetric? true})))
                   source (await-cps (gfi/run-policy (fn [] (conditional-collections))
                                                     (it/policy {:draw (draw-switch true)}) {:seed 409}))]
               (try
                 (is (contains? (:trace/entries source) :later-observation))
                 (let [next (await-cps (it/reexecute source #{:switch}
                                                     (it/policy {:keep? true :draw (draw-switch false)}) {:seed 410}))]
                   (is (= :prefix (:org.replikativ.foerster.runtime.trace/restart next)))
                   (is (false? (get (it/choices next) :switch)))
                   (is (not (contains? (:trace/entries next) :later-observation)))
                   (is (= [(get (it/choices source) :prefix) false] (:trace/result next))))
                 (finally (await-cps (gfi/close! source))))))
           (finally (v2/release-world! root)))))))

(defn- run-portable [make check done]
  (let [root (v2/world)
        finish (fn [result]
                 (try (check result)
                      (finally (v2/release-world! root) (done))))]
    (binding [v2/*world* root]
      ((make) finish finish))))

(deftest finite-gibbs-and-mh-are-portable-public-kernels
  (let [task (fn []
               (infer/infer (fn [] (keyed-model))
                            {:method :kernel :chains 1 :seed 407
                             :kernel (k/cycle 4 [(k/finite-gibbs-kernel 1 [:coin 0])
                                                 (k/single-site-mh-kernel 1)])}))
        check (fn [result]
                (is (satisfies? m/PMeasure result))
                (when (satisfies? m/PMeasure result)
                  (is (= 1 (count (m/get-particles result)))))
                (is (true? (get-in (transition/prepare (k/finite-gibbs-kernel 1 [:coin 0]))
                                   [:capabilities :tempering?]))))]
    #?(:clj (let [outcome (promise)]
              (run-portable task #(deliver outcome %) (fn []))
              (check (deref outcome 30000 ::timeout)))
       :cljs (async done (run-portable task check done)))))

(deftest keyed-population-copy-and-gibbs-are-portable
  (let [task (fn []
               (ip/run {:seed 408}
                       (fn [scope]
                         (task
                          (let [source (await-op (ip/initialize scope (ip/trace-target (fn [] (keyed-model))) 4))
                                child (await-op (ip/fork scope source))
                                child (ip/advance scope child (ip/temperature (:target child) 0.5))
                                child (await-op (ip/resample scope child))
                                child (await-op (ip/move scope child (k/finite-gibbs-kernel 1 [:coin 0])))
                                result (ip/finish scope child)]
                            (ip/finish scope source)
                            result)))))
        check (fn [result]
                (is (satisfies? m/PMeasure result))
                (when (satisfies? m/PMeasure result)
                  (is (= 4 (count (m/get-particles result))))
                  (is (= [:advance :resample :move] (mapv :operation (:history result))))))]
    #?(:clj (let [outcome (promise)]
              (run-portable task #(deliver outcome %) (fn []))
              (check (deref outcome 30000 ::timeout)))
       :cljs (async done (run-portable task check done)))))
