(ns org.replikativ.foerster.transition-context-test
  (:require [clojure.test :refer [deftest is]]
            [org.replikativ.foerster.benchmark-test :as b]
            [org.replikativ.foerster.block :as block]
            [org.replikativ.foerster.core :as infer]
            [org.replikativ.foerster.effects :refer [sample]]
            [org.replikativ.foerster.kernel :as k]
            [org.replikativ.foerster.nuts :as nuts]
            [org.replikativ.foerster.measure :as m]
            [org.replikativ.foerster.random :as random]
            [org.replikativ.foerster.trace :as it]
            [org.replikativ.foerster.transition :as transition]
            [org.replikativ.foerster.transition-context :as tc]
            [org.replikativ.spindel.trace :as trace]
            [org.replikativ.spindel.engine.protocols :as rtp]
            [org.replikativ.spindel.spin.cps :refer [spin]]))

(defn- await-cps [op]
  (let [p (promise)]
    (op #(deliver p [:ok %]) #(deliver p [:err %]))
    (let [[kind value] (deref p 30000 [:err (ex-info "timeout" {})])]
      (if (= :ok kind) value (throw value)))))

(deftest repeated-population-moves-keep-local-tuning-and-independent-rng-indices
  (let [seen (atom []) made (atom 0)
        context {:temperature 1.0 :constraints {:y 1} :policy-options {:interventions {:z 0}}
                 :until :frontier :keep-old? true :shared-anchors? true}
        visits (atom [])
        current {:trace/world (atom 0)}]
    (with-redefs [rtp/swap-state! (fn [world _ f] (swap! world f))
                  it/mh-step (fn [t _] (fn [resolve _] (resolve {:trace t :accepted? false})))
                  nuts/within-gibbs (fn [_]
                                      (swap! made inc)
                                      (fn [t opts]
                                        (swap! seen conj opts)
                                        (fn [resolve _] (resolve {:trace t :accepted? true}))))]
      (let [prepared (transition/prepare
                      (k/cycle 1 [(k/single-site-mh-kernel 100)
                                  (k/nuts-kernel 1 {:burn 10})]))]
        (dotimes [_ 10]
          (let [result (await-cps (transition/step prepared current (assoc context :on-visit #(swap! visits conj %))))]
            (is (= 101 (:moves result)))
            (is (= 1 (:accepted result)))))
        (is (transition/adaptation? prepared))
        (is (= 1 @made))
        (is (= (range 10) (map :adaptation-iteration @seen)))
        (is (= (range 100 1010 101) (map :iteration @seen)))
        (is (every? #(= context (select-keys % (keys context))) @seen))
        (is (= 1010 (count @visits)))))))

(deftest nested-mixtures-preserve-component-counters-and-context
  (let [seen (atom []) current {:trace/world (atom 0)}
        context {:temperature 1.0 :until :frontier :keep-old? true}]
    (with-redefs [rtp/swap-state! (fn [world _ f] (swap! world f))
                  it/mh-step (fn [t _] (fn [resolve _] (resolve {:trace t :accepted? false})))
                  random/in-world-stream (fn [_ _ f] (f))
                  m/sample-categorical (constantly 0)
                  nuts/within-gibbs (fn [_]
                                      (fn [t opts]
                                        (swap! seen conj opts)
                                        (fn [resolve _] (resolve {:trace t :accepted? true}))))]
      (let [prepared (transition/prepare
                      (k/cycle 1 [(k/mixture 2 [[1 (k/nuts-kernel 1)]
                                                [1 (k/single-site-mh-kernel 1)]])
                                  (k/single-site-mh-kernel 3)]))]
        (dotimes [_ 2] (await-cps (transition/step prepared current context)))
        (is (= [0 1 2 3] (mapv :adaptation-iteration @seen)))
        (is (= [0 1 5 6] (mapv :iteration @seen)))
        (is (every? #(= context (select-keys % (keys context))) @seen))))))

(deftest distinct-prepared-descriptors-share-the-worlds-rng-reservation
  (let [seen (atom []) current {:trace/world (atom 0)}]
    (with-redefs [rtp/swap-state! (fn [world _ f] (swap! world f))
                  it/mh-step (fn [t opts]
                               (swap! seen conj (:iteration opts))
                               (fn [resolve _] (resolve {:trace t :accepted? false})))]
      (let [a (transition/prepare (k/single-site-mh-kernel 2))
            b (transition/prepare (k/single-site-mh-kernel 1))]
        (await-cps (transition/step a current {}))
        (await-cps (transition/step b current {}))
        (await-cps (transition/step a current {}))
        (is (= [0 1 2 3 4] @seen))))))

(deftest chain-driver-reserves-composed-move-spans-once
  (let [seen (atom []) current {:trace/world (atom 0)}]
    (with-redefs [rtp/swap-state! (fn [world _ f] (swap! world f))
                  it/mh-step (fn [t opts]
                               (swap! seen conj (:iteration opts))
                               (fn [resolve _] (resolve {:trace t :accepted? false})))]
      (let [opts (transition/options (k/cycle 3 [(k/single-site-mh-kernel 2)
                                                 (k/single-site-mh-kernel 1)]))]
        (is (= 3 (:iteration-stride opts)))
        (await-cps (it/mh-chain current (:iterations opts) opts))
        (is (= (range 9) @seen))
        (is (= 9 @(:trace/world current)))))))

(deftest numerical-tempering-fails-before-any-component-moves
  (let [moves (atom 0)]
    (with-redefs [it/mh-step (fn [_ _] (swap! moves inc))]
      (let [prepared (transition/prepare (k/cycle 1 [(k/single-site-mh-kernel 2)
                                                     (k/nuts-kernel 1)]))]
        (is (thrown-with-msg? clojure.lang.ExceptionInfo #"temperature one"
                              (transition/step prepared {} {:temperature 0.5})))
        (is (zero? @moves))))))

(deftest ownership-policy-retains-only-displaced-states
  (let [released (atom []) old {:id :old} candidate {:id :new}]
    (with-redefs [trace/release! (fn [& args] (swap! released conj args))]
      (tc/settle! {:keep-old? true} old candidate true)
      (is (empty? @released))
      (tc/settle! {:keep-old? true} old candidate false)
      (is (= [[candidate old]] @released)))))

(def ^:private gaussian
  (block/block {:block/id :local-warmup :block/target :complete-conditional
                :block/latents [{:name :x :shape [1] :support :real}]}
               {:log-density (fn [x _] (* -0.5 (aget ^doubles x 0) (aget ^doubles x 0)))
                :value+grad (fn [x _]
                              [(* -0.5 (aget ^doubles x 0) (aget ^doubles x 0))
                               (double-array [(- (aget ^doubles x 0))])])}))

(deftest actual-nuts-warmup-uses-component-visits
  ;; MH's 100 visits must not consume NUTS's ten warm-up iterations.
  (let [seen (atom []) adapt @#'nuts/adapt]
    (with-redefs-fn {#'it/mh-step (fn [t _] (fn [resolve _] (resolve {:trace t :accepted? false})))
                     #'nuts/adapt (fn [state i & args]
                                    (swap! seen conj i)
                                    (apply adapt state i args))}
      #(b/run-infer 893 (fn []
                          (infer/kernel-infer
                           (spin (sample (block/block-dist gaussian nil) :id :x :init [0.2]))
                           (k/cycle 10 [(k/single-site-mh-kernel 100)
                                        (k/nuts-kernel 1 {:burn 10})]) 1))))
    (is (= (range 10) @seen))))
