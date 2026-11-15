(ns org.replikativ.foerster.cancellation-test
  "Cancelling a public inference task stops its work: every world the
  inference created is released, no model run starts once the driver has
  noticed, and the driver settles (`foerster.runtime.lifetime`)."
  (:require [clojure.test :refer [deftest is testing]]
            [org.replikativ.foerster.core :as inference]
            [org.replikativ.foerster.dist :as dist]
            [org.replikativ.foerster.effects :refer [observe sample]]
            [org.replikativ.foerster.moves.kernel :as kernel]
            [org.replikativ.foerster.runtime.async :refer [await-op]]
            [org.replikativ.spindel.v2.core :as v2 :refer [task]]))

(defn- released? [w]
  (try (v2/get-in-world w [::probe]) false
       (catch clojure.lang.ExceptionInfo e (boolean (:spindel/released (ex-data e))))))

(defn- wait-until [ms f]
  (let [end (+ (System/currentTimeMillis) ms)]
    (loop [] (cond (f) true (> (System/currentTimeMillis) end) false :else (do (Thread/sleep 10) (recur))))))

(defmacro ^:private capturing-worlds
  "Run `body` with every world and host fork made meanwhile in `worlds`, held
  weakly (a collected world was released)."
  [[worlds] & body]
  `(let [~worlds (atom [])
         world# v2/world
         fork# v2/fork!
         keep# (fn [w#] (swap! ~worlds conj (java.lang.ref.WeakReference. w#)) w#)]
     (with-redefs [v2/world (fn [& args#] (keep# (apply world# args#)))
                   v2/fork! (fn [& args#] (keep# (apply fork# args#)))]
       ~@body)))

(defn- live [refs] (keep #(.get ^java.lang.ref.WeakReference %) refs))

(defn- model
  "Twenty sites, one observation each, a little wait at each; counts the
  steps every run of it takes, in whatever world, in `steps`."
  [steps]
  (fn []
    (task
     (loop [i 0 x 0.0]
       (if (= i 20)
         x
         (let [x (sample (dist/normal x 1.0) :id [:x i])]
           (v2/sleep 2)
           (swap! steps inc)
           (observe (dist/normal x 1.0) 0.0 :id [:y i])
           (recur (inc i) x)))))))

(def ^:private particle-methods
  ;; sizes no test waits for: inference is cancelled long before it ends
  {:smc #(inference/smc-infer %1 100 %2)
   :importance #(inference/importance-sampling %1 2000 %2)
   :tempered #(inference/tempered-infer %1 100 %2)
   :mh #(inference/kernel-infer %1 (kernel/single-site-mh-kernel 100000 {:samples :final}) 2 %2)
   :pimh #(inference/pimh-infer %1 10 10000 %2)
   :pgibbs #(inference/pgibbs-infer %1 10 10000 %2)
   :pgas #(inference/pgas-infer %1 10 10000 %2)
   :ipmcmc #(inference/ipmcmc-infer %1 10 10000 (assoc %2 :num-nodes 2 :num-csmc-nodes 1))
   :bbvi #(inference/bbvi-infer %1 10 10000 %2)})

(defn- cancel-mid-run
  "Starts `run` in a caller world, cancels its task once its model ran a
  few times, and reports what happened after."
  [run policy]
  (let [caller (v2/world)
        starts (atom 0)]
    (try
      (capturing-worlds [worlds]
        (let [t (binding [v2/*world* caller] (task (await-op (run (model starts) {:world-policy policy}))))
              started? (wait-until 20000 #(>= @starts 50))]
          (v2/cancel-task! t)
          (let [outcome (try (v2/value t 5000 ::timed-out) (catch Exception e (if (v2/cancelled? e) ::cancelled e)))
                ;; the driver notices at its next await or loop head
                _ (Thread/sleep 300)
                at-rest @starts
                _ (Thread/sleep 700)
                made @worlds]
            {:started? started?
             :outcome outcome
             :quiet? (= at-rest @starts)
             :released? (wait-until 5000 #(every? released? (remove #{caller} (live made))))
             :worlds (count made)})))
      (finally (v2/release-world! caller)))))

(deftest cancelling-stops-every-method
  (doseq [[method run] (sort-by key particle-methods)
          policy [:fresh :fork]]
    (testing (str method " " policy)
      (let [{:keys [started? outcome quiet? released? worlds]} (cancel-mid-run run policy)]
        (is started? "the model ran before the cancellation")
        (is (= ::cancelled outcome))
        (is quiet? "no model step is taken after the driver noticed")
        (is (pos? worlds))
        (is released? "every world the inference made is released")))))

(deftest a-nested-inference-is-cancelled-with-its-outer-one
  ;; an inference awaited inside a model body runs under that body's task:
  ;; releasing the outer particle's world cancels it
  (let [caller (v2/world)
        inner-starts (atom 0)]
    (try
      (let [t (binding [v2/*world* caller]
                (task (await-op (inference/smc-infer
                                 (fn [] (task (await-op (inference/smc-infer (model inner-starts) 20))))
                                 2))))]
        (is (wait-until 20000 #(>= @inner-starts 50)))
        (v2/cancel-task! t)
        (Thread/sleep 300)
        (let [at-rest @inner-starts]
          (Thread/sleep 700)
          (is (= at-rest @inner-starts) "the inner inference stopped too")))
      (finally (v2/release-world! caller)))))
