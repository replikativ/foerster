(ns org.replikativ.foerster.attribution-test
  (:require [clojure.test :refer [deftest is]]
            [org.replikativ.foerster.attribution :as a]))

(deftest interaction-is-shared-and-dummy-player-gets-zero
  (let [game #(if (and (% :a) (% :b)) 10.0 0.0)]
    (is (= {:a 5.0 :b 5.0 :dummy 0.0}
           (:contributions (a/shapley [:a :b :dummy] game))))
    (let [r (a/shapley [:a :b :dummy] game {:method :permutation :permutations 100 :seed 8})]
      (is (= 10.0 (reduce + (vals (:contributions r)))))
      (is (= 0.0 (get-in r [:contributions :dummy])))
      (is (= r (a/shapley [:a :b :dummy] game {:method :permutation :permutations 100 :seed 8}))))))

(deftest budgets-do-not-produce-a-completed-answer
  (is (thrown? #?(:clj Exception :cljs js/Error)
               (a/shapley [:a :b] count {:max-evaluations 1}))))

(deftest mechanism-change-attribution
  (is (= {:supplier 2.0 :warehouse 3.0}
         (:contributions (a/attribute-change {:supplier 0 :warehouse 0}
                                             {:supplier 2 :warehouse 3}
                                             #(reduce + (vals %)))))))

(deftest preemption-needs-a-witness-rather-than-a-single-but-for-test
  (let [evaluate (fn [interventions]
                   (let [s (get interventions :s true) b (get interventions :b true)
                         hit-s (get interventions :hit-s s)
                         hit-b (get interventions :hit-b (and b (not hit-s)))
                         broken (get interventions :broken (or hit-s hit-b))]
                     {:s s :b b :hit-s hit-s :hit-b hit-b :broken broken}))
        domains {:s [false true] :b [false true]}]
    (is (:broken (evaluate {:s false})))
    (is (= :actual-cause (:status (a/actual-cause evaluate :broken #{:s} domains))))
    (is (= :not-cause (:status (a/actual-cause evaluate :broken #{:b} domains))))
    (is (= :not-minimal (:status (a/actual-cause evaluate :broken #{:s :b} domains))))
    (is (= :unresolved (:status (a/actual-cause evaluate :broken #{:s} domains {:max-evaluations 1}))))))
