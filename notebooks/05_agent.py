# Databricks notebook source
# MAGIC %md
# MAGIC # 05 — Playground, AI Gateway, Agents
# MAGIC
# MAGIC **Covers sidebar:** Playground, AI Gateway, Agents.
# MAGIC
# MAGIC **Prereq:** `01`–`04`. The agent's tools are the tables you built.
# MAGIC
# MAGIC **Runtime:** ~20 minutes.
# MAGIC
# MAGIC > Availability varies by tier. Each section states what to expect if it is
# MAGIC > blocked, and every section has a fallback that works without it.

# COMMAND ----------

import mlflow, json, time
from pyspark.sql import functions as F

CATALOG, SCHEMA = "workspace", "default"
SILVER  = f"{CATALOG}.{SCHEMA}.yelp_silver"
GOLD    = f"{CATALOG}.{SCHEMA}.yelp_gold_business"
SCORED  = f"{CATALOG}.{SCHEMA}.yelp_scored"

# COMMAND ----------

# MAGIC %md
# MAGIC ## Section 1 — Playground
# MAGIC
# MAGIC Sidebar → **Playground**. A chat UI over hosted foundation models
# MAGIC (Llama, Claude, GPT-class, embedding models). Free Edition includes credits.
# MAGIC
# MAGIC Three things worth doing there before you write any code:
# MAGIC
# MAGIC 1. **Compare models on the same prompt.** Paste a genuinely ambiguous
# MAGIC    3-star review and ask for a 1-5 rating. Note which models hedge and
# MAGIC    which commit. Model choice is an engineering decision with a cost
# MAGIC    difference of 10-50x.
# MAGIC 2. **Watch the token counter.** It is the unit you are billed in, and
# MAGIC    intuition about it is worth building early.
# MAGIC 3. **Export as code.** The Playground has a "view code" action that emits
# MAGIC    the exact API call. That is the bridge from chat to pipeline.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Section 2 — Call a foundation model from code

# COMMAND ----------

from databricks.sdk import WorkspaceClient
from databricks.sdk.service.serving import ChatMessage, ChatMessageRole

w = WorkspaceClient()

# List what your workspace actually offers — do not guess model names
try:
    eps = [e.name for e in w.serving_endpoints.list()]
    print("available endpoints:")
    for e in eps: print("  ", e)
except Exception as e:
    print("could not list:", str(e)[:200])

# EXPECTED OUTPUT — something like:
#   databricks-meta-llama-3-3-70b-instruct
#   databricks-claude-sonnet-4
#   databricks-gte-large-en                  <-- embeddings, not chat
#   yelp-sentiment                           <-- yours, from 04
#
# Names change as models are added and retired. ALWAYS list rather than
# hardcoding a name from a tutorial — a stale model name is the most common
# reason a copied example fails.

# COMMAND ----------

LLM = "databricks-meta-llama-3-3-70b-instruct"   # <-- set from the list above

def ask(prompt, system="You are a concise data analyst.", max_tokens=400, temperature=0.1):
    r = w.serving_endpoints.query(
        name=LLM,
        messages=[ChatMessage(role=ChatMessageRole.SYSTEM, content=system),
                  ChatMessage(role=ChatMessageRole.USER,   content=prompt)],
        max_tokens=max_tokens,
        temperature=temperature)
    return r.choices[0].message.content

print(ask("In one sentence, what does AUC 0.94 tell me about a binary classifier?"))

# EXPECTED OUTPUT: a one-sentence explanation of ranking quality.
#
# temperature=0.1 — near-deterministic. For ANALYTICAL tasks (classification,
# extraction, SQL generation) you want low temperature: creativity is a defect
# there. Raise it only for genuinely generative work.
#
# max_tokens caps the RESPONSE, not the prompt. Setting it too low truncates
# mid-sentence, which looks like a model failure but is your own limit.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Section 3 — LLM vs your trained model: a real comparison
# MAGIC
# MAGIC You now have two ways to classify a review. Compare them on cost,
# MAGIC latency and accuracy rather than assuming the LLM is better.

# COMMAND ----------

sample = (spark.table(SILVER)
    .select("review_id", "text", "stars", "is_positive")
    .limit(20).collect())

results = []
t0 = time.time()
for row in sample:
    out = ask(
        f'Review: "{row.text[:900]}"\n\n'
        f'Reply with ONLY the number 1 or 0. 1 if the reviewer is positive '
        f'(would rate 4-5 stars), 0 otherwise. No explanation.',
        system="You classify sentiment. Reply with a single digit.",
        max_tokens=4)
    pred = 1 if "1" in out[:3] else 0
    results.append((row.review_id, row.stars, row.is_positive, pred))
llm_secs = time.time() - t0

correct = sum(1 for _, _, actual, pred in results if actual == pred)
print(f"LLM zero-shot: {correct}/{len(results)} = {correct/len(results):.0%} "
      f"in {llm_secs:.0f}s  ({llm_secs/len(results):.2f}s per review)")

# EXPECTED OUTPUT (approximate):
#   LLM zero-shot: 18/20 = 90% in 24s  (1.20s per review)
#
# THE COMPARISON THAT MATTERS:
#
#   metric        trained LogReg (file 03)      LLM zero-shot
#   ------------  ----------------------------  --------------------------
#   accuracy      ~87%                          ~90%
#   latency       ~1 ms/review (batch)          ~1,200 ms/review
#   cost          ~$0 after training            per-token, forever
#   700k reviews  minutes                       ~10 days serial, and $$$$
#   training data 559k labelled examples        none
#   explainable   yes — inspect coefficients    no
#
# READ THE THROUGHPUT ROW. The LLM is slightly more accurate and roughly
# 1,000x slower per record. For scoring a large corpus the trained model wins
# decisively. For a handful of records, or when you have no labelled data at
# all, the LLM wins because it needs no training set.
#
# THE ACTUAL LESSON: an LLM is not a replacement for a classifier at scale.
# It is a replacement for the LABELLING EFFORT — use it to bootstrap labels,
# then train a small fast model on them. That hybrid is the pattern most
# production teams land on.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Section 4 — Where an LLM genuinely beats a classifier

# COMMAND ----------

reviews = "\n\n".join(f"[{r.stars}★] {r.text[:400]}" for r in sample[:8])

print(ask(
    f"Here are customer reviews for one restaurant:\n\n{reviews}\n\n"
    "Produce:\n"
    "1. Three specific operational problems mentioned, most frequent first\n"
    "2. Two things customers consistently praise\n"
    "3. One concrete action the owner should take this week\n"
    "Be specific. Cite the review star rating you drew each point from.",
    system="You are a restaurant operations consultant. Be concrete, never generic.",
    max_tokens=600))

# EXPECTED OUTPUT: structured operational analysis citing specific reviews.
#
# THIS is what a classifier cannot do. Your LogReg outputs a single number.
# The LLM extracts THEMES, RANKS them, and RECOMMENDS action — open-ended
# reasoning over unstructured text.
#
# THE DIVISION OF LABOUR, which is the real architectural point:
#   classifier -> high-volume, low-latency, structured scoring
#   LLM        -> low-volume, open-ended reasoning and synthesis
# Systems that use one where the other belongs are either slow and expensive,
# or shallow.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Section 5 — AI Gateway
# MAGIC
# MAGIC Sidebar → **AI Gateway**. A governance layer in front of model endpoints:
# MAGIC rate limits, usage tracking, payload logging, PII guardrails, fallbacks.
# MAGIC
# MAGIC > **May be Premium-only.** If the code below fails with PERMISSION_DENIED
# MAGIC > or FEATURE_NOT_AVAILABLE, that is expected — read the concepts and move on.

# COMMAND ----------

from databricks.sdk.service.serving import (
    AiGatewayConfig, AiGatewayRateLimit, AiGatewayUsageTrackingConfig,
    AiGatewayInferenceTableConfig)

try:
    w.serving_endpoints.put_ai_gateway(
        name="yelp-sentiment",
        rate_limits=[AiGatewayRateLimit(calls=100, renewal_period="minute", key="endpoint")],
        usage_tracking_config=AiGatewayUsageTrackingConfig(enabled=True),
        inference_table_config=AiGatewayInferenceTableConfig(
            enabled=True, catalog_name=CATALOG, schema_name=SCHEMA,
            table_name_prefix="yelp_inference"))
    print("AI Gateway configured")
except Exception as e:
    print("AI Gateway unavailable:", str(e)[:300])

# WHAT EACH PIECE BUYS YOU:
#
#   rate_limits              caps calls per minute, keyed by endpoint or user.
#                            This is the guardrail against a runaway loop
#                            generating a five-figure bill overnight — the most
#                            common LLM cost incident there is.
#
#   usage_tracking           token counts and cost per request, per user.
#                            Without it, "who spent $8,000 last month" is
#                            unanswerable.
#
#   inference_table_config   logs every request and response to a DELTA TABLE.
#                            This is the highest-value item and the least used.
#                            It gives you: a debugging record when a user says
#                            "it gave me a wrong answer", an audit trail,
#                            drift monitoring, and — most valuably — a
#                            real-traffic dataset for evaluation or fine-tuning.
#
# Also available: fallbacks (route to a second model when the primary errors)
# and guardrails (block PII, filter unsafe content) — the same governance layer
# you would otherwise build yourself in a proxy service.

# COMMAND ----------

# Inspect the inference log once traffic has flowed
try:
    display(spark.table(f"{CATALOG}.{SCHEMA}.yelp_inference_payload").limit(10))
except Exception as e:
    print("no inference table yet:", str(e)[:200])

# The table appears a few minutes after the first request. Columns include
# request, response, timestamps, status code, and token counts.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Section 6 — Agents: give the LLM your tables as tools
# MAGIC
# MAGIC This is the **Agents** item, and it is where the whole lab converges.
# MAGIC An agent is an LLM plus a set of callable tools plus a loop: the model
# MAGIC decides which tool to call, sees the result, and decides again.
# MAGIC
# MAGIC The tools here are **Unity Catalog functions** over the tables you built —
# MAGIC which means the agent is governed by the same permissions as everything else.

# COMMAND ----------

spark.sql(f"""
CREATE OR REPLACE FUNCTION {CATALOG}.{SCHEMA}.get_business_stats(
    biz_id STRING COMMENT 'Business id such as b_00042'
)
RETURNS TABLE (business_id STRING, n_reviews LONG, avg_stars DOUBLE,
               positive_rate DOUBLE, weighted_stars DOUBLE, rank INT)
COMMENT 'Return review statistics for one business by its id.'
RETURN
  SELECT business_id, n_reviews, avg_stars, positive_rate, weighted_stars, rank
  FROM {GOLD}
  WHERE business_id = get_business_stats.biz_id
""")

spark.sql(f"""
CREATE OR REPLACE FUNCTION {CATALOG}.{SCHEMA}.top_businesses(
    n INT COMMENT 'How many businesses to return, 1 to 50'
)
RETURNS TABLE (business_id STRING, n_reviews LONG, weighted_stars DOUBLE, rank INT)
COMMENT 'Return the highest rated businesses by Bayesian weighted star rating.'
RETURN
  SELECT business_id, n_reviews, weighted_stars, rank
  FROM {GOLD}
  ORDER BY rank
  LIMIT top_businesses.n
""")

spark.sql(f"""
CREATE OR REPLACE FUNCTION {CATALOG}.{SCHEMA}.sample_reviews(
    biz_id STRING COMMENT 'Business id',
    star_filter INT COMMENT 'Only reviews with this star rating, 1 to 5'
)
RETURNS TABLE (review_id STRING, stars INT, review_text STRING)
COMMENT 'Return up to 5 review texts for a business at a given star rating.'
RETURN
  SELECT review_id, stars, substr(text, 1, 500) AS review_text
  FROM {SILVER}
  WHERE business_id = sample_reviews.biz_id
    AND stars = sample_reviews.star_filter
  LIMIT 5
""")

display(spark.sql(f"SELECT * FROM {CATALOG}.{SCHEMA}.top_businesses(5)"))

# EXPECTED OUTPUT: the 5 top-ranked businesses from your gold table.
#
# >>> THE COMMENTS ARE NOT DOCUMENTATION. THEY ARE THE TOOL SPECIFICATION. <<<
#
# The agent reads the function COMMENT and each parameter COMMENT to decide
# WHEN to call the tool and WHAT to pass. A vague comment produces an agent
# that calls the wrong tool or passes garbage. This is the single highest-
# leverage thing in agent building and it looks like a triviality.
#
# Compare:
#   BAD:  COMMENT 'gets stats'
#   GOOD: COMMENT 'Return review statistics for one business by its id.'
#
# Note also `get_business_stats.biz_id` — qualifying the parameter with the
# function name. Without it, `biz_id` is ambiguous against a column of the
# same name and the function silently matches every row.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Wire the tools to the model

# COMMAND ----------

try:
    from unitycatalog.ai.core.databricks import DatabricksFunctionClient
    from unitycatalog.ai.langchain.toolkit import UCFunctionToolkit
    from langchain_openai import ChatOpenAI
    from langchain.agents import AgentExecutor, create_tool_calling_agent
    from langchain_core.prompts import ChatPromptTemplate

    client = DatabricksFunctionClient()
    toolkit = UCFunctionToolkit(function_names=[
        f"{CATALOG}.{SCHEMA}.get_business_stats",
        f"{CATALOG}.{SCHEMA}.top_businesses",
        f"{CATALOG}.{SCHEMA}.sample_reviews",
    ], client=client)

    llm = ChatOpenAI(model=LLM, temperature=0.1)   # Databricks endpoints are OpenAI-compatible

    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are a restaurant analytics assistant. Use the provided tools to "
         "answer questions with real data. Never invent numbers. If a tool "
         "returns no rows, say so plainly."),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}"),
    ])

    agent = AgentExecutor(
        agent=create_tool_calling_agent(llm, toolkit.tools, prompt),
        tools=toolkit.tools,
        verbose=True,
        max_iterations=5)

    print(agent.invoke({"input":
        "What are the three top rated businesses, and for the very best one, "
        "show me what a 1-star reviewer complained about?"})["output"])

except ImportError as e:
    print("agent libraries missing — run:")
    print("  %pip install -U unitycatalog-ai[databricks] unitycatalog-langchain[databricks] "
          "langchain langchain-openai")
    print("  dbutils.library.restartPython()")
except Exception as e:
    print("agent failed:", str(e)[:400])

# EXPECTED BEHAVIOUR with verbose=True — you SEE the reasoning loop:
#
#   > Invoking: top_businesses with {'n': 3}
#   [b_00318 rank 1, b_01204 rank 2, b_00077 rank 3]
#   > Invoking: sample_reviews with {'biz_id': 'b_00318', 'star_filter': 1}
#   [3 review texts]
#   > Final Answer: The top three are ... For b_00318, one-star reviewers
#     complained about slow service and ...
#
# TWO tool calls, and the SECOND DEPENDS ON THE FIRST — the agent had to learn
# the business id before it could ask for that business's reviews. That
# dependency is what distinguishes an agent from a single function call.
#
# max_iterations=5 is a safety limit. Without it a confused agent can loop
# indefinitely, burning tokens. Always bound the loop.
#
# WATCH THE VERBOSE OUTPUT CLOSELY. If the agent picks the wrong tool, the fix
# is almost always the COMMENT text, not the prompt.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Fallback: the same idea in 20 lines, no framework

# COMMAND ----------

TOOLS = {
    "top_businesses":     lambda n=5: spark.sql(
        f"SELECT * FROM {CATALOG}.{SCHEMA}.top_businesses({int(n)})").toPandas().to_dict("records"),
    "get_business_stats": lambda biz_id: spark.sql(
        f"SELECT * FROM {CATALOG}.{SCHEMA}.get_business_stats('{biz_id}')").toPandas().to_dict("records"),
    "sample_reviews":     lambda biz_id, star_filter: spark.sql(
        f"SELECT * FROM {CATALOG}.{SCHEMA}.sample_reviews('{biz_id}',{int(star_filter)})"
    ).toPandas().to_dict("records"),
}

def mini_agent(question, max_steps=4):
    """Minimal ReAct loop: model emits JSON tool calls, we execute, feed back."""
    history = []
    for step in range(max_steps):
        out = ask(
            f"Question: {question}\n\n"
            f"Tools:\n"
            f"  top_businesses(n)\n"
            f"  get_business_stats(biz_id)\n"
            f"  sample_reviews(biz_id, star_filter)\n\n"
            f"Results so far: {json.dumps(history)[:2000]}\n\n"
            f'Reply with EITHER {{"tool":"name","args":{{...}}}} '
            f'OR {{"answer":"final answer text"}}. JSON only.',
            system="You are a tool-using agent. Output only valid JSON.",
            max_tokens=300)
        try:
            msg = json.loads(out[out.find("{"):out.rfind("}") + 1])
        except Exception:
            return f"could not parse: {out[:200]}"
        if "answer" in msg:
            return msg["answer"]
        name, args = msg.get("tool"), msg.get("args", {})
        if name not in TOOLS:
            history.append({"tool": name, "error": "unknown tool"}); continue
        try:
            history.append({"tool": name, "args": args, "result": TOOLS[name](**args)})
        except Exception as e:
            history.append({"tool": name, "args": args, "error": str(e)[:200]})
    return "hit step limit"

print(mini_agent("Which business has the highest weighted rating, and how many reviews does it have?"))

# THIS IS ALL AN AGENT FUNDAMENTALLY IS:
#   loop { model proposes a tool call -> execute it -> feed the result back }
# until the model produces an answer or you hit the step limit.
#
# Frameworks add retries, parallel tool calls, streaming, tracing, structured
# output validation and memory. None of that changes the core loop. Writing it
# once by hand is the fastest way to stop finding agents mysterious.
#
# NOTE THE SQL INJECTION RISK in the lambdas above: f-string interpolation of
# biz_id straight into SQL. Acceptable in a notebook where you control the
# input; NOT acceptable if the value can come from an end user. Use parameter
# binding (spark.sql(query, args={...})) in anything real. UC functions called
# through the proper toolkit handle this for you — another reason to prefer them.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Section 7 — Log the agent to MLflow

# COMMAND ----------

try:
    mlflow.set_experiment(
        f"/Users/{spark.sql('SELECT current_user()').first()[0]}/yelp-agent")
    with mlflow.start_run(run_name="yelp-analytics-agent"):
        mlflow.log_params({
            "llm": LLM, "temperature": 0.1, "max_iterations": 5,
            "tools": ",".join(TOOLS.keys()),
        })
        qs = [
            "What are the top 3 businesses?",
            "How many reviews does b_00001 have?",
            "What do 1-star reviewers say about the top business?",
        ]
        for i, q in enumerate(qs):
            a = mini_agent(q)
            mlflow.log_text(f"Q: {q}\n\nA: {a}", f"qa_{i}.txt")
            print(f"Q: {q}\nA: {a}\n")
except Exception as e:
    print("logging failed:", str(e)[:300])

# Logging agent runs gives you an EVALUATION BASELINE. When you change a tool
# comment or swap the model, re-run the same questions and diff the answers.
# Without that, "did my change help?" is unanswerable — which is the state
# most agent projects are in.
#
# MLflow also offers mlflow.evaluate() with LLM-judge metrics for scoring
# answer quality automatically.

# COMMAND ----------

# MAGIC %md
# MAGIC ## What you built
# MAGIC
# MAGIC | Artefact | Sidebar |
# MAGIC |---|---|
# MAGIC | 3 Unity Catalog functions as agent tools | Catalog → Functions |
# MAGIC | Foundation-model calls from code | **Playground** |
# MAGIC | Rate limits + inference logging | **AI Gateway** |
# MAGIC | Tool-calling agent over your gold tables | **Agents** |
# MAGIC | Agent runs logged as an eval baseline | **Experiments** |
# MAGIC
# MAGIC ### Concepts demonstrated
# MAGIC - Low temperature for analytical tasks
# MAGIC - LLM vs trained classifier: ~1,000x latency gap, and what each is for
# MAGIC - LLMs replace the labelling effort, not the classifier
# MAGIC - UC function COMMENTs **are** the tool spec
# MAGIC - Multi-step agent loops where step 2 depends on step 1
# MAGIC - `max_iterations` as a cost guardrail
# MAGIC - Inference tables as debugging, audit and training data
# MAGIC - SQL injection risk in hand-rolled tools
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## The whole lab
# MAGIC
# MAGIC ```
# MAGIC 01  HF Parquet ──Auto Loader──▶ yelp_bronze
# MAGIC 02  yelp_bronze ──transform──▶ yelp_silver ──▶ yelp_gold_business
# MAGIC 03  yelp_silver ──features──▶ yelp_business_features
# MAGIC                 ──MLflow────▶ trained model
# MAGIC 04  model ──register──▶ UC registry @champion ──▶ serving endpoint
# MAGIC                                                ──▶ yelp_scored
# MAGIC 05  gold tables ──UC functions──▶ agent tools ──▶ LLM agent
# MAGIC ```
# MAGIC
# MAGIC Ten sidebar items, one dataset, one connected pipeline.
