(ns org.replikativ.foerster.workflow.posterior-test
  "The posteriordb environment end to end with a scripted candidate, no model:
  `clojure -M:workflow -e \"(require 'org.replikativ.foerster.workflow.posterior-test) (clojure.test/run-tests 'org.replikativ.foerster.workflow.posterior-test)\"`."
  (:require [clojure.test :refer [deftest is testing]]
            [dvergr.agent.evaluation :as evaluation]
            [dvergr.agent.roster :as roster]
            [dvergr.chat.agent :as chat-agent]
            [dvergr.discourse :as d]
            [dvergr.model.chat :as model-chat]
            [dvergr.model.providers :as providers]
            [dvergr.room.store :as store]
            [dvergr.room.store.memory :as memory]
            [dvergr.substrate.geschichte :as g]
            [muschel.fs :as fs]
            [muschel.fs.geschichte :as gfs]
            [org.replikativ.foerster.posteriordb :as pdb]
            [org.replikativ.foerster.workflow.posterior :as wp]
            [org.replikativ.spindel.engine.core :as ec]
            [org.replikativ.spindel.yggdrasil :as ygg]
            [yggdrasil.adapters.geschichte :as gy]))

(def ^:private posterior "eight_schools-eight_schools_noncentered")

(defn- resampled
  "2000 joint draws resampled from the reference draws, each parameter's
  spread around its mean scaled by `sd-scale`."
  [reference-draws sd-scale seed]
  (let [rng (java.util.Random. seed)
        n (count (val (first reference-draws)))
        idx (vec (repeatedly 2000 #(.nextInt rng n)))]
    (into (sorted-map)
          (for [[k xs] reference-draws
                :let [mu (/ (reduce + xs) (count xs))]]
            [k (mapv #(+ mu (* sd-scale (- (xs %) mu))) idx)]))))

(deftest draws-are-checked-against-the-reference
  (let [ref-draws (pdb/reference-draws posterior)
        reference (wp/describe ref-draws)
        good (resampled ref-draws 1.0 1)
        checks #(:checks (wp/check-draws reference %))]
    (is (every? true? (vals (checks good))))
    (testing "too narrow"
      (let [c (checks (resampled ref-draws 0.5 1))]
        (is (:means? c))
        (is (false? (:sds? c)))))
    (testing "a shifted parameter"
      (is (false? (:means? (checks (update good "mu" #(mapv (fn [x] (+ x 5.0)) %)))))))
    (testing "columns shuffled independently: the moments hold, the dependence does not"
      (let [rng (java.util.Random. 3)
            c (checks (update-vals good #(let [l (java.util.ArrayList. ^java.util.Collection %)]
                                           (java.util.Collections/shuffle l rng)
                                           (vec l))))]
        (is (:means? c))
        (is (:sds? c))
        (is (false? (:correlations? c)))))
    (testing "two points with the right mean and sd"
      (let [two (into (sorted-map)
                      (for [[k {:keys [mean sd]}] (:stats reference)]
                        [k (vec (take 2000 (cycle [(- mean sd) (+ mean sd)])))]))]
        (is (false? (:quantiles? (checks two))))))
    (testing "a missing parameter, too few draws, unequal lengths, no draws"
      (is (false? (:all-params? (checks (dissoc good "mu")))))
      (is (false? (:joint-draws? (checks (update-vals good #(subvec % 0 10))))))
      (is (false? (:joint-draws? (checks (update good "mu" pop))))))
    (is (false? (:draws? (checks nil))))))

(deftest only-one-flat-edn-map-is-read
  (is (= {"a" [1.0 2.0]} (wp/parse-draws "{\"a\" [1 2.0]}")))
  (is (nil? (wp/parse-draws "{\"a\" [1 2]} {\"b\" [3]}")) "a second form")
  (is (nil? (wp/parse-draws "{\"a\" [1 2]} )")) "trailing garbage")
  (is (nil? (wp/parse-draws "{\"a\" [[1] 2]}")) "nested")
  (is (nil? (wp/parse-draws (str "{\"a\" " (apply str (repeat 100000 "[")) "}"))) "deeply nested")
  (is (nil? (wp/parse-draws "{\"a\" #inst \"2020-01-01\"}")) "a tag")
  (is (nil? (wp/parse-draws "{\"a\" (1 2)}")) "a list")
  (is (nil? (wp/parse-draws "{\"a\" [1 \"2\"]}")) "a string draw")
  (is (nil? (wp/parse-draws "{\"a\\")) "an escape at the end")
  (is (nil? (wp/parse-draws "{\"a\" [1]} :org.replikativ.foerster.workflow.posterior/eof")) "a keyword like a sentinel")
  (is (= {"a" [1.0 2.0]} (wp/parse-draws "{\"a\" [1 ; [ a comment\n 2]}")) "brackets in a comment"))

(defn- run-scripted
  "One evaluation of `posterior` whose candidate writes `draws` with one
  clojure_eval call; the attempt receipt and the persisted evidence."
  [draws]
  (let [{:keys [conn close!]} (gfs/memory-repository! {:name "workflow-run"})
        room (d/make-room {:id :workflow-posterior-run :store (memory/make)})
        team (roster/make-agent (roster/make-roster)
                                {:id :analyst :tools #{:clojure_eval}
                                 :model-policy {:provider :test :model "stub"}
                                 :program {:kind :llm :max-model-steps 3 :auto-compact? false}})
        calls (atom 0)
        seen (atom nil)
        code (str "(let [stan (slurp " (pr-str wp/model-path) ") data (slurp " (pr-str wp/data-path) ")] "
                  "(spit " (pr-str wp/draws-path) " " (pr-str (pr-str draws)) ") "
                  "[(count stan) (count data)])")]
    (try
      (binding [ec/*execution-context* (:ctx room)]
        (ygg/register! (gy/create conn {:system-name "room-repo-workflow-test"}))
        (with-redefs [providers/ensure-initialized! (constantly nil)
                      chat-agent/messages->api-format (fn [messages _ _] messages)
                      model-chat/chat
                      (fn [messages _]
                        (if (= 1 (swap! calls inc))
                          {:content "" :tool-calls [{:id "infer" :name "clojure_eval" :input {:code code}}]
                           :usage {:input-tokens 0 :output-tokens 0} :stop-reason :tool-use}
                          (do (reset! seen (:message/content (last (filter #(= :tool-result (:message/role %)) messages))))
                              {:content "Wrote the draws." :tool-calls nil
                               :usage {:input-tokens 0 :output-tokens 0} :stop-reason :end-turn})))]
          (let [done (promise)
                computation (evaluation/evaluate room team :analyst (wp/definition posterior) (wp/evaluator posterior)
                                                 {:world-setup (wp/world-setup posterior)})
                _ (computation #(deliver done {:result %}) #(deliver done {:error %}))
                outcome (deref done 120000 ::timeout)]
            (when-let [error (:error outcome)] (throw error))
            {:outcome outcome
             :tool-result @seen
             :receipt (get-in outcome [:result :attempt-receipt])
             :persisted (first (store/-list-attempts (:store room) (:id room) {}))
             :draws-left? (some? (fs/stat (g/filesystem) wp/draws-path))})))
      (finally
        (evaluation/await-cleanups! room)
        (d/close-room! room)
        (close!)))))

(deftest a-scripted-candidate-through-the-world-lifecycle
  (let [ref-draws (pdb/reference-draws posterior)]
    (doseq [[sd-scale reward] [[1.0 1.0] [0.5 0.0]]]
      (let [{:keys [outcome tool-result receipt persisted draws-left?]} (run-scripted (resampled ref-draws sd-scale 2))]
        (is (not= ::timeout outcome))
        (is (= :completed (:attempt/status receipt)))
        (is (= reward (:attempt/reward receipt)) (str tool-result))
        (is (= :ok (get-in persisted [:attempt/evidence :draws :status])))
        (testing "the run's world is discarded"
          (is (false? draws-left?)))))))
