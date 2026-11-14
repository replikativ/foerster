(ns org.replikativ.foerster.site-path-test
  "Readable site paths: `:id` names — hierarchical ones as vectors — give
  savepoint sites Gen-style addresses a caller can name from outside;
  `v2/with-path` prefixes the names of the sites inside it."
  (:require [org.replikativ.foerster.runtime.async :as async]
            [clojure.test :refer [deftest is testing]]
            [org.replikativ.spindel.v2.savepoint :as sp]
            [org.replikativ.spindel.v2.trace :as trace]
            [org.replikativ.foerster.runtime.trace :as itrace]
            [org.replikativ.foerster.effects :refer [sample]]
            [org.replikativ.spindel.v2.core :as v2 :refer [task with-path]]
            [org.replikativ.foerster.dist :as dist]))

(defn- await-cps [operation]
  (let [result (promise)]
    ((async/op-of operation) #(deliver result [:ok %]) #(deliver result [:error %]))
    (let [outcome (deref result 20000 ::timeout)]
      (when (= ::timeout outcome)
        (throw (ex-info "CPS operation timed out" {})))
      (if (= :ok (first outcome)) (second outcome) (throw (second outcome))))))

(defn- run
  "Run the model `make` (a function creating a task) under the scoring
  policy with `constraints`, in a root world and session of its own;
  returns the trace."
  ([make] (run make {}))
  ([make constraints]
   (let [root (v2/world)
         session (sp/open! root {:fork-opts {:systems :none} :retain-released? false})
         t (await-cps (trace/run session make
                                 (itrace/policy {:constraints constraints})))]
     (await-cps (sp/close! session))
     t)))

(defn- steps []
  (task
   (loop [i 0 xs []]
     (if (= i 3)
       xs
       (recur (inc i)
              (conj xs (sample (dist/normal 0.0 1.0) :id [:step i :x])))))))

(deftest a-hierarchical-name-is-the-address
  (let [t (run steps {[:step 1 :x] 0.5})]
    (is (= [[:step 0 :x] [:step 1 :x] [:step 2 :x]] (:trace/order t)))
    (is (= 0.5 (second (:trace/result t))) "constrained from outside, by path")))

(deftest a-scalar-name-is-its-own-address
  (let [t (run #(task (sample (dist/normal 0.0 1.0) :id :top)))]
    (is (= [:top] (:trace/order t)))))

(defn- branchy []
  (task
   (let [b (sample (dist/flip 0.5) :id :b)
         _ (when b (sample (dist/normal 0.0 1.0)))
         v (sample (dist/normal 0.0 1.0))]
     [b v])))

(deftest an-unnamed-site-keeps-its-address-across-an-upstream-branch
  (let [a1 (peek (:trace/order (run branchy {:b true})))
        a2 (peek (:trace/order (run branchy {:b false})))]
    (is (= a1 a2))
    (testing "its address is [frame key]: the creating frames and the site's key"
      (is (= [] (first a1)) "a task the host created")
      (is (integer? (second a1))))))

(deftest a-name-reused-in-one-run-is-an-error
  (is (thrown-with-msg? clojure.lang.ExceptionInfo #"Duplicate site address"
                        (run #(task (loop [i 0]
                                      (when (< i 2)
                                        (sample (dist/normal 0.0 1.0) :id :x)
                                        (recur (inc i)))))))))

(deftest a-path-prefixes-names-and-nests
  (let [t (run #(task
                 [(loop [i 0 xs []]
                    (if (= i 2)
                      xs
                      (recur (inc i) (conj xs (with-path [:step i]
                                                (sample (dist/normal 0.0 1.0) :id :x))))))
                  (with-path [:a]
                    (with-path [:b 2]
                      (sample (dist/normal 0.0 1.0) :id :y)))
                  (sample (dist/normal 0.0 1.0) :id :top)])
               {[:step 1 :x] 0.5})]
    (is (= [[:step 0 :x] [:step 1 :x] [:a :b 2 :y] :top] (:trace/order t))
        "the path does not leak past its form, across suspensions")
    (is (= 0.5 (second (first (:trace/result t)))) "constrained from outside by the full name")))

(deftest unnamed-sites-in-different-scopes-are-different-sites
  (let [t (run #(task [(with-path [:a] (sample (dist/normal 0.0 1.0)))
                       (with-path [:b] (sample (dist/normal 0.0 1.0)))]))
        [pa pb] (:trace/order t)]
    (is (= 2 (count (distinct (:trace/order t)))))
    (is (= :a (first pa)))
    (is (= :b (first pb)))))
