# Architecture

How JARVIS actually fits together, as of the code in this repository. Every claim
below was read out of the source; where something is unverified or only partially
true, it says so.

References are `path.py:LINE symbol()`. Line numbers drift; the symbol name is the
durable part.

---

## 1. The layer diagram

```
                    ┌──────────────────────────────────────────────┐
   entry points     │  cli.py      voice.py      app.py            │
                    │  argparse    VoiceLoop     build()/shutdown()│
                    └───────────────────┬──────────────────────────┘
                                        │
                    ┌───────────────────▼──────────────────────────┐
   agent            │  agent/orchestrator.py   Orchestrator        │
                    │  agent/task_manager.py   TaskManager         │
                    │  agent/subagent.py       run_agent_loop, SubAgent
                    │  agent/protocol.py       parse_tool_calls    │
                    │  agent/prompts.py        persona + templates │
                    └───┬────────┬─────────┬──────────┬────────────┘
                        │        │         │          │
        ┌───────────────▼┐ ┌─────▼─────┐ ┌─▼────────┐ ┌▼──────────────┐
  leaf  │ llm/           │ │ memory/   │ │ tools/   │ │ speech/       │
        │ vllm  ollama   │ │ store     │ │ registry │ │ stt  tts      │
        │ openai_compat  │ │ context   │ │ file_    │ │ audio_io      │
        │ transformers   │ │ embeddings│ │ system_  │ │ windows_speech│
        │ airllm  stub   │ │           │ │ process_ │ │               │
        │ models (catalog)│ │          │ │ web_ app_│ │               │
        └───────────────┬┘ └─────┬─────┘ └─┬────────┘ └┬──────────────┘
                        │        │         │           │
                    ┌───▼────────▼─────────▼───────────▼───────────┐
   core             │  contracts.py   config.py   events.py        │
                    │  security.py    platform_utils.py            │
                    │  logging_setup.py                            │
                    └──────────────────────────────────────────────┘

   jarvis/win/    (tray, hotkeys, autostart, toasts)
   jarvis/linux/  (systemd user service, notifications, XDG autostart, audio)
   Both sit beside the entry points and depend only on core. Nothing in the agent
   path imports either, and — at the time of writing — neither is wired into
   cli.py. Each reports itself unavailable on the other platform rather than
   raising, so `jarvis doctor` can describe any host honestly.
```

The dependency rule is one-directional: **`core` imports nothing from JARVIS
except itself; leaf packages import `core`; the agent layer imports leaves through
their contracts; entry points import everything.** `jarvis/core/contracts.py` has
no JARVIS imports at all — it is the bottom of the graph.

Two consequences worth internalising:

* The orchestrator never names a concrete backend. It receives an `LLMBackend`, a
  registry, and a context object, and would work identically against a different
  LLM, a different tool set, or a different memory implementation.
* `jarvis/core/platform_utils.py` is the *only* module that is allowed to branch on
  `sys.platform` for path/shell/process behaviour. Other modules import
  `IS_WINDOWS` / `IS_LINUX` from it (tools and speech do this legitimately for
  OS-specific commands), but nothing re-derives them.

---

## 2. The three contracts everything hangs off

All three live in `jarvis/core/contracts.py`. They are ABCs plus dataclasses, with
zero dependencies, so any of them can be implemented in a test file in ten lines.

### `LLMBackend` — `contracts.py:101`

```python
name: str
is_available() -> bool                      # cheap, must never raise
load() -> None                              # idempotent
generate(messages, config) -> LLMResult
stream(messages, config) -> Iterator[str]   # default: yields generate().text once
unload() -> None                            # optional
```

`Message` (`contracts.py:44`) carries `role`, `content`, optional `name` /
`tool_call_id`, free-form `metadata`, and a timestamp. `to_dict()` emits the
OpenAI-shaped subset, which is what the Ollama backend posts on the wire.

`GenerationConfig` is the knob bundle (`max_new_tokens`, `temperature`, `top_p`,
`top_k`, `stop`, `seed`); `LLMResult` is `text` plus token counts, `finish_reason`
and the raw payload.

Concrete implementations live in `jarvis/llm/`. `BaseLLM` (`llm/base.py:202`)
handles the boilerplate — idempotent `load()` delegating to `_do_load()`, merging
a caller's `GenerationConfig` with the config defaults in `_gen_config()`, and a
naive character-buffered `stream()` fallback. A backend author only has to write
`is_available`, `_do_load` and `generate`.

There are two bases to inherit from: `BaseLLM` for anything that loads weights or
speaks a bespoke protocol, and `OpenAICompatBackend` (`llm/openai_compat.py:239`)
for anything that speaks `/v1/chat/completions` — that one already handles
transport, retries, streaming and the in-flight cap, so `VLLMBackend` is barely
sixty lines of defaults on top of it.

### `Tool` / `ToolSpec` / `ToolResult` — `contracts.py:232-297`

```python
ToolParam(name, type, description, required, default, enum)
ToolSpec(name, description, params, dangerous)   # .json_schema()
ToolResult(ok, output, error, is_artifact)       # .success() / .failure()
Tool.spec -> ToolSpec ; Tool.run(**kwargs) -> ToolResult
```

`ToolResult` is the universal failure type. **Public tool functions return
`ToolResult.failure(...)`; they do not raise.** The registry catches exceptions
anyway (`tools/registry.py:429 _execute()`), but a raising tool loses its error
message quality and, if it raises inside a thread with a timeout, is much harder
to diagnose.

Most tools are not `Tool` subclasses. They are plain functions wrapped by
`FunctionTool` (`tools/registry.py:147`), whose `ToolSpec` is derived by
introspecting the signature and type hints (`_spec_from_callable`,
`registry.py:97`) and whose description is the first non-blank line of the
docstring. That is why tool docstrings matter: they become the model's catalogue.

### `MemoryStore` — `contracts.py:204`

```python
add(record) -> str
get(record_id) -> Optional[MemoryRecord]
search(query, *, k=8, kind=None) -> list      # "semantic + recency"
recent(*, k=20, kind=None) -> list
all(*, kind=None) -> Iterable
close() -> None                                # optional
```

`MemoryRecord` is `(id, kind, text, metadata, ts, score)`. The only production
implementation is `SQLiteMemoryStore` (`memory/store.py:99`), which adds
`add_text()`, `delete()`, `count()`, `stats()`, `export_jsonl()` and
`import_jsonl()` beyond the contract — `ContextManager` duck-types for `add_text`
and `delete` rather than requiring them.

A fourth interface pair, `STTEngine` / `TTSEngine` (`contracts.py:139` and `:155`),
governs the speech layer. It is less load-bearing than the three above because
nothing in the agent path depends on it: the orchestrator only ever calls
`.say()` or `.speak()` on whatever it was handed, and tolerates `None`.

---

## 3. The reason-act loop

One function: `run_agent_loop()` at `jarvis/agent/subagent.py:150`. The interactive
orchestrator and every background subagent call it, so their behaviour cannot
drift apart.

```
                     ┌─────────────────────────────────────────┐
                     │  messages: list[Message]  (mutated in    │
                     │  place, so the caller keeps the whole    │
                     │  transcript including tool turns)        │
                     └────────────────┬────────────────────────┘
                                      │
   iteration 1..max_iterations        ▼
        ┌──────────────────────────────────────────────────┐
        │ llm.generate(messages, gen_config).text          │
        │   (or llm.stream(...) joined, when stream=True    │
        │    and on_chunk is not None → ASSISTANT_CHUNK)   │
        └────────────────┬─────────────────────────────────┘
                         ▼
              parse_tool_calls(raw)
                         │
         ┌───────────────┴────────────────┐
         │ [] (no calls)                  │ [calls...]
         ▼                                ▼
  answer = strip_tool_calls(raw)   append Message.assistant(raw,
        or _fallback_answer(turn)      metadata={"tool_calls":[names]})
  append Message.assistant(answer)          │
  return turn                                ▼
                                   for each call:
                                     unknown?  → ToolResult.failure(hint)
                                     known?    → emit TOOL_CALL
                                                 registry.run(name, **args)
                                                 emit TOOL_RESULT
                                     append Message.tool(
                                       render_tool_result(call, result),
                                       name=call.name,
                                       tool_call_id=call.id)
                                                │
                                    ┌───────────┘
                                    ▼  (next iteration)

   loop exhausted → turn.truncated = True
        append Message.user("You have reached the tool-use limit. Give your
                             final answer now ... Do not call any more tools.")
        one final generate() → strip → answer (or _fallback_answer)
```

Points that matter in practice:

* **A response with no parseable tool call is the terminating condition.** There is
  no separate "done" signal. `parse_tool_calls()` returning `[]` *is* the answer.
* **The raw model output is stored verbatim** when it contains tool calls
  (`subagent.py:207`), tags and all, so the transcript stays faithful. Only the
  final answer goes through `strip_tool_calls()`.
* **Empty answers are never returned.** `_fallback_answer()` (`subagent.py:117`)
  synthesises a truthful spoken sentence from what the tools did, because silence
  is the worst possible output for a voice assistant. This fires more often than
  you would expect with small quantised models.
* **A generation exception is caught and converted into the reply text**
  (`subagent.py:189`), not propagated. The user hears "the language model failed:
  ...".
* **Unknown tool names come back as a tool result, not a crash** — with a fuzzy
  suggestion from `_unknown_tool_message()` (`subagent.py:108`), which the model
  usually acts on.

### Parsing `<tool_call>` — `jarvis/agent/protocol.py`

The canonical form, which Qwen3 emits natively and `format_tool_call()` produces:

```
<tool_call>
{"name": "read_file", "arguments": {"path": "C:/notes.txt"}}
</tool_call>
```

`parse_tool_calls()` (`protocol.py:204`) is deliberately forgiving, in this order:

1. Fast bail: no `{` in the text → `[]`.
2. `_TOOL_CALL_RE` (`protocol.py:34`) — `<tool_call>\s*(body)\s*(?:</tool_call>|$)`,
   DOTALL and case-insensitive. **The closing tag is optional**, so a response cut
   off by `max_new_tokens` still yields its complete leading calls.
3. A code fence *inside* the tags is unwrapped (`_FENCE_RE`, `protocol.py:41`).
4. `_balanced_objects()` (`protocol.py:67`) extracts each top-level `{...}` with a
   character scanner that tracks string state and escapes. A regex cannot do this
   once arguments contain braces inside strings — Windows paths and shell commands
   routinely do — hence the scanner.
5. `_loads()` (`protocol.py:133`) escalates: raw `json.loads` → `_tidy` (smart
   quotes, trailing commas) → `_jsonize` (`True`/`False`/`None` → JSON literals) →
   `ast.literal_eval`. The order matters: jsonizing before `literal_eval` would
   break single-quoted Python-dict payloads.
6. `_normalise_call()` (`protocol.py:156`) accepts the shapes models actually
   produce: OpenAI's `{"type":"function","function":{...}}` nesting; `name` under
   any of `name`/`tool`/`tool_name`; arguments under any of
   `arguments`/`args`/`parameters`/`params`/`input`/`kwargs`; arguments delivered
   as a JSON *string*; parameters inlined beside the name.
7. If nothing parsed from tags, bare fenced blocks are tried, but only accepted
   when they look like a call (they contain a `name` key *and* one of
   `arguments`/`args`/`parameters`/`tool`).

Results are fed back with `render_tool_result()` (`protocol.py:260`): dict/list
output is pretty-printed JSON, failures become `ERROR: <error>`, artifacts are
prefixed `[artifact]`, and anything over `limit=4000` characters is middle-elided
(head 70 %, tail 20 %) with a visible truncation marker.

### Why the registry is the only execution path

`ToolRegistry.run()` (`tools/registry.py:322`) is the single choke-point. In order:

1. Unknown tool → `ToolResult.failure`.
2. Unknown parameter names rejected, unless the underlying function declares
   `**kwargs`.
3. Each declared parameter coerced to its JSON type (`_coerce`, `registry.py:199`),
   `enum` membership checked, missing required parameters rejected, defaults filled.
4. `security.check_tool(spec, cleaned)` consulted; a denial returns a failure, and
   a `requires_confirmation` decision is routed through `security.allows()`.
5. `Events.TOOL_CALL` emitted.
6. `_execute()` runs the tool, optionally on a daemon thread with a join timeout.
7. Result normalised to a `ToolResult`, appended to `registry.history`, and
   `Events.TOOL_RESULT` emitted.

`name` is positional-only (`def run(self, name, /, *, timeout=None, **kwargs)`) so
a tool may itself declare a parameter called `name` without colliding with the
dispatcher.

**Known limitation:** the timeout in `_execute()` joins a daemon thread. Python
cannot kill a thread, so a wedged tool is *abandoned*, not stopped — the call
returns a timeout failure while the thread keeps running. Tools that block must
therefore impose their own timeouts (every `subprocess` call in the tree does).

---

## 4. The agent tree

Tasks form a **forest**, not a list: every task records `parent_id` and `depth`,
children are tracked on the parent, and a parent does not settle until its whole
subtree has.

```
   user ──► Orchestrator.chat()                    (agent/orchestrator.py)
              │  holds self._lock for the whole turn: one conversation
              │  turn at a time, in-process
              ├─ _collect_reports()  ← TaskManager.take_reports()
              │      finished background work (at any depth) is injected as
              │      system messages and persisted as memory kind "task"
              ├─ context.build(...) → messages
              ├─ messages[0] replaced with Orchestrator.system_prompt()
              └─ run_agent_loop(..., max_iterations=agent.max_tool_iterations)
                        │
                        │ model calls the `spawn_task` meta-tool
                        ▼
              Orchestrator.spawn_task(goal, context, parent_id=None)
                        │  parent_id defaults to current_agent_context()["task_id"]
                        │  — the thread-local of the subagent that called the tool,
                        │  so parentage is never taken from the model's word for it
                        ▼
              TaskManager.spawn(goal, sub.run, parent_id=...)
                        │  depth = parent.depth + 1  (roots are depth 0)
                        │  refuses over max_depth or max_total_tasks
                        │  submits to _pool_for(depth) — ONE POOL PER DEPTH LEVEL
                        ▼
              SubAgent.run(task, progress)          (agent/subagent.py)
                        │  reads depth/parent_id from task.metadata,
                        │  adds a delegation_note() to its prompt, and runs
                        │  inside `with agent_context(task_id=..., depth=...)`
                        ▼
              the SAME run_agent_loop, over the SAME ToolRegistry, which still
              contains `spawn_task` — recursion is expected, and bounded.
```

### Depth and fan-out

| Bound | Default | Enforced where |
|---|---|---|
| Task depth | `max_depth = 3` (`DEFAULT_MAX_DEPTH`) — root plus three generations | `TaskManager.spawn()` refuses beyond it |
| Tasks tracked at once | `max_total_tasks = 64` (`DEFAULT_MAX_TOTAL_TASKS`) | `TaskManager.spawn()`, after reaping announced leaves |
| Concurrent tasks **per depth level** | `agent.max_concurrent_tasks` (4) | one `ThreadPoolExecutor(max_workers=...)` per depth |
| Thread ceiling | `(max_depth + 1) * max_workers` = **16** | the per-depth pools |
| Tool iterations, main agent | `agent.max_tool_iterations` (8) | `Orchestrator.chat()` |
| Tool iterations, subagent | `max(4, max_tool_iterations * 2)` = 16 | `Orchestrator.spawn_task()` |
| Wall clock per task | `agent.subagent_timeout` (900 s) | checked inside `progress()` |
| In-flight LLM requests | `llm.max_concurrent_requests` (8; `0` = unlimited) | `threading.Semaphore` in `llm/openai_compat.py` — **HTTP backends only** |

**A refusal is a value, not an exception.** Exceeding `max_depth` or
`max_total_tasks` returns a `Task` that is already `FAILED`, is *not* registered
with the manager, and whose `error` names the limit, the current value and what to
do instead ("carry out this step yourself and put the outcome in your report").
That text goes back to the model as a tool result, so the agent that tripped the
limit can read it and adapt rather than retrying forever. Neither limit asks
anyone's permission; both exist so one mis-prompted agent cannot become a fork bomb.

The subagent is also *told* its budget: `delegation_note(depth, max_depth)` is
folded into its system prompt, so it knows how many levels remain before it is
refused.

### The pool deadlock, and why it cannot happen

"A parent is not done until its children are done" is precisely the shape that
deadlocks a fixed worker pool — the parent occupies a worker while waiting for
children that need workers. Two independent mechanisms rule it out
(`agent/task_manager.py` module docstring):

1. **The join is event-driven.** When a runner returns, its outcome is parked on the
   handle and the worker thread is released immediately; the *last child to settle*
   is what finishes the parent. No thread ever blocks on a descendant.
2. **One pool per depth level**, created lazily. A task at depth `d` only ever waits
   on tasks at depth `d + 1`, which live in a different pool, so even a runner that
   deliberately blocks on `wait()` for its own child cannot starve itself.

The documented residual: a task that blocks on a *sibling* at its own depth can
still starve that level. Nothing in JARVIS does that, and it was judged not worth
pessimising the common path for.

### Cancellation and timeout are cooperative

Both are observed only when the runner calls its `progress()` callback:
`progress()` raises `CancelledError` if the cancel event is set, and `TimeoutError`
if the deadline has passed. `run_agent_loop` calls `progress()` once per iteration
and once per tool invocation, so cancellation lands between steps. **A task blocked
inside a single long tool call cannot be cancelled or timed out**, because control
never returns to `progress()`.

### Inspecting the tree

`TaskManager` exposes `depth_of`, `parent_of`, `children`, `descendants`,
`ancestry`, `roots`, `tree`, `render_tree` and a `stats()` that reports `tracked`,
`roots`, `deepest_depth`, `by_depth`, `max_depth`, `max_total_tasks` and `refused`.
The model gets at this through the `task_tree` meta-tool.

```bash
python -c "
from jarvis.agent.task_manager import TaskManager, DEFAULT_MAX_DEPTH, DEFAULT_MAX_TOTAL_TASKS
print(DEFAULT_MAX_DEPTH, DEFAULT_MAX_TOTAL_TASKS)
tm = TaskManager(); print(tm.stats()); tm.shutdown(wait=False)
"
```

### How reports surface

`TaskManager` marks each settled task with an `announced` flag. `take_reports()`
atomically returns the un-announced settled tasks and marks them; a parent's result
has its children's reports folded in as it settles. Two consumers call it:

* `Orchestrator._collect_reports()` at the top of every `chat()` turn — reports
  become `system` messages in that turn's prompt and are written to memory as kind
  `"task"`.
* `Orchestrator.pending_updates()`, used by `cli.py` (printed after each reply) and
  by `VoiceLoop._idle_tick()` (`voice.py:442`, spoken while nobody is talking, and
  deliberately skipped right after a barge-in).

Because both paths call `take_reports()`, **a report is consumed by whichever runs
first** — it is announced once, not twice.

---

## 5. Memory

### Storage — `jarvis/memory/store.py`

SQLite, one file (`Config.db_file()`, default `<data dir>/memory.db`), opened with
`check_same_thread=False`, `isolation_level=None` (autocommit), `journal_mode=WAL`,
`synchronous=NORMAL`, `busy_timeout=5000`, and every statement wrapped in an
internal `threading.RLock`. That combination is what makes it safe to share between
the voice thread, the agent thread and four task workers.

Three tables:

| Table | Contents |
|---|---|
| `records` | `id, kind, text, metadata (JSON), ts` + indexes on `kind` and `ts` |
| `embeddings` | `id, dim, vec` — float32 packed with `struct` (4 bytes/dim, not JSON) |
| `records_fts` | FTS5 external-content virtual table + insert/delete/update triggers |

FTS5 is **detected at runtime** (`_detect_fts5()`, `store.py:46`) by attempting a
`CREATE VIRTUAL TABLE` against `:memory:`, because some Windows Python builds ship
without it. When it is absent — or when the sanitised query comes out empty — the
store falls back to a case-insensitive `LIKE` scan over up to 8 query tokens.

### Hybrid search — `store.py:336 search()`

1. **Keyword.** With FTS5: the query is stripped to word characters and OR-joined
   as quoted phrases (`_sanitise_fts_query`, `store.py:81`) — necessary because
   FTS5 has its own operator syntax that raw punctuation trips. `bm25()` is negated
   (lower bm25 = better) to give a positive score. Without FTS5: one `LIKE '%tok%'`
   query per token, +1.0 per hit. Scores are then normalised to a 0–1 range.
2. **Vector.** If an embedder with `dim > 0` is configured, the query is embedded
   and compared by `cosine()` (`memory/embeddings.py:31`) against **every** stored
   embedding. This is a full table scan — there is no ANN index. Fine at the tens
   of thousands of records a personal assistant accumulates; it will not scale to
   millions.
3. **Pool.** Candidates = keyword hits ∪ cosine hits. **Recency alone never puts a
   record in the pool**, deliberately: otherwise a vague query would surface every
   recent turn as if it were relevant.
4. **Score.** `0.55 * cosine + 0.35 * keyword + 0.10 * recency`, where
   `recency = 0.5 ** (age_seconds / 30 days)`. Sorted descending, deduplicated,
   truncated to `k`.

Embedders (`memory/embeddings.py`), in the order `create_embedder()` prefers:

| Embedder | Dim | Notes |
|---|---|---|
| `SentenceTransformerEmbedder` | model-defined (384 for MiniLM-L6-v2) | lazy import; chosen by `"auto"` when importable |
| `HashEmbedder` | `memory.embed_dim` (384) | zero dependencies; blake2b-hashed char 3/4/5-grams + word unigrams, L2-normalised. Uses blake2b, **not** Python's salted `hash()`, so vectors are stable across processes and machines |
| `NullEmbedder` | 0 | collapses search to keyword-only |

### The rolling summary — `jarvis/memory/context.py`

`ContextManager` keeps a live window `_live: List[Message]` and drives recall.
`maybe_summarize()` (`context.py:219`) fires when `len(_live) > summarize_after_turns`
(default 20): everything except the last `keep_recent_turns` (default 8) is
compressed into `self.summary`, persisted as kind `"summary"`, and dropped from the
window.

`_summarise_text()` prefers an LLM (a 256-token summarisation call) and otherwise
uses a deterministic extractive fallback — join the lines, truncate each to 240
chars, cap the whole thing at 4000 chars.

> **Unverified / worth knowing:** in the production boot path the LLM summariser is
> never used. `app.py:89` calls `create_context(cfg.memory, store=store)` without
> `llm=`, and nothing later assigns `context.llm`. So `ContextManager.llm is None`
> and the *extractive* fallback is what actually runs. The LLM path is exercised
> only by tests that construct `ContextManager` directly.

`build()` (`context.py:154`) assembles the prompt in this order: system prompt →
`Relevant recollections:` block (search hits above `recall_min_score`, with
anything already visible verbatim in the recent window filtered out) →
`Conversation summary so far:` → the last `keep_recent_turns` live messages →
caller `extra` (task reports) → the new user message.

### What "practically forgets nothing" concretely means

| Kind | Written by | When |
|---|---|---|
| `conversation` | `ContextManager.add_user` / `add_assistant` | every user turn and every assistant reply |
| `fact` | `remember_fact()`, exposed as the `remember` tool | when the model decides something is durable |
| `summary` | `maybe_summarize()` | each time the live window is compressed |
| `task` | `Orchestrator._collect_reports()` | when a background task report is folded in |

And the guarantees behind the phrase:

* `add()` is **first-write-wins by id** (`store.py:200`) — an existing id is left
  untouched. The store never overwrites a record.
* Reads (`search` / `recent` / `all`) never mutate.
* Nothing prunes. `memory.prune` exists as a config field
  (`core/config.py:147`) but **is not read anywhere in the codebase** — verify with
  `grep -rn "prune" jarvis/`. There is no retention policy, no TTL, no vacuum.
* `delete()` exists on the store and `ContextManager.forget()` wraps it, but **no
  built-in tool exposes deletion of memory**. The model cannot forget on request.
* A persistence failure is retried once and then recorded: `failed_writes` grows,
  `persistence_healthy` flips to `False`, and `Subsystems.status()` reports
  `sqlite (WRITE FAILURES - see log)` (`app.py:60`). A broken store must not look
  like a healthy one.

The honest caveats: **tool results are not persisted.** `run_agent_loop` appends
`Message.tool(...)` to the transcript list it was handed, which is a fresh list
built by `context.build()` each turn — not the live window. `ContextManager.add_tool`
exists but nothing in the agent path calls it. So memory contains what you said and
what JARVIS said, not what the tools returned. Likewise the *whole* text of each
turn is stored, but recall is `recall_k = 8` records per turn, so what the model
sees is a small ranked slice of what is kept.

---

## 6. The event bus

`jarvis/core/events.py`. Thread-safe pub/sub with an `RLock`, handlers may be sync
or coroutine functions, and `emit()` never raises into the caller — a failing
handler is logged and the next one still runs. `get_bus()` returns a lazily created
process-wide bus; `reset_bus()` exists for tests.

Regenerate this table at any time with
`grep -rn "emit(Events\.\|_emit(Events\." jarvis/ --include=*.py`.

| Channel | Constant | Payload | Emitted by | Listened to by (in-package) |
|---|---|---|---|---|
| `user.utterance` | `USER_UTTERANCE` | `str` | `Orchestrator.chat()`, `VoiceLoop.run()` | — |
| `assistant.reply` | `ASSISTANT_REPLY` | `str` | `Orchestrator.chat()` | — |
| `assistant.chunk` | `ASSISTANT_CHUNK` | `str` | `run_agent_loop()`, streaming only | — |
| `task.created` | `TASK_CREATED` | `Task` | `TaskManager.spawn()` | — |
| `task.update` | `TASK_UPDATE` | `TaskUpdate` | `TaskManager` — `progress()`, `_set_state()`, `_finish()` | — |
| `task.done` | `TASK_DONE` | `Task` | `TaskManager._finish()` | — |
| `task.failed` | `TASK_FAILED` | `Task` | `TaskManager._finish()` | — |
| `tool.call` | `TOOL_CALL` | `{"name","kwargs"}` from the registry; `{"name","arguments"}` from the loop | `ToolRegistry.run()` (`registry.py:393`), `run_agent_loop()` | — |
| `tool.result` | `TOOL_RESULT` | `{"name","ok","error","duration","is_artifact"}` (registry) / `{"name","ok"}` (loop) | `ToolRegistry.run()` (`registry.py:415`), `run_agent_loop()` | — |
| `tool.created` | `TOOL_CREATED` | `{"name","path","tools"}` | `tool_maker.make_tool()` (`tool_maker.py:456`) | — |
| `speak` | `SPEAK` | `str` | `Orchestrator.say()` | — |
| `listen.start` | `LISTEN_START` | `None` | `VoiceLoop._set_state()` (`voice.py:254`) | — |
| `listen.stop` | `LISTEN_STOP` | `None` | `VoiceLoop._set_state()` (`voice.py:251`) | — |
| `wake` | `WAKE` | `str` (the full utterance) | `VoiceLoop._gate_wake_word()` (`voice.py:388`) | — |
| `error` | `ERROR` | — | **never emitted** | — |
| `shutdown` | `SHUTDOWN` | `None` | `Orchestrator.shutdown()` | — |
| `voice.state` | `VOICE_STATE` (module constant in `voice.py:67`, deliberately *not* in `Events`) | `"idle"`/`"listening"`/`"thinking"`/`"speaking"` | `VoiceLoop._set_state()` (`voice.py:252`) | — |

Two things to take from that table:

1. **A tool call is announced twice** — once by `run_agent_loop` and once by
   `ToolRegistry.run` — with different payload shapes. A subscriber that counts
   tool calls will double-count. This is not a bug the code hides: the loop emits
   before dispatch (so a subscriber sees the intent even if the registry refuses),
   and the registry emits around actual execution with timing.
2. **Nothing inside `jarvis/` subscribes to anything.** The bus is emit-only in the
   package; every subscriber is a test, or would be a UI (tray, overlay, web
   front-end). That is the intended shape — the bus exists so a UI can attach
   without the agent knowing — but it means a broken emit has no in-package
   consumer to notice it. `Events.ERROR` is declared and never used.

---

## 7. Design decisions, with their reasons

### Every third-party import is lazy

Enforced by `tests/test_import_hygiene.py`, which both AST-parses every file *and*
re-imports the whole package in a clean subprocess with ~30 heavy packages blocked
by a `sys.meta_path` hook.

The reason is the boot promise in `app.py`: JARVIS starts with **no** optional
dependencies installed, degrades per-subsystem, and tells you what is missing via
`jarvis doctor`. A single `import torch` at module scope would turn "no microphone,
but chat works" into "won't start". It also keeps the test suite fast and hermetic,
and keeps `jarvis doctor` — the command you run *because* something is broken —
from being the thing that breaks.

The same rule covers OS-specific stdlib (`winreg`, `winsound`, `msvcrt`, `_winapi`):
importing those at module level would break Linux outright. They go inside an
`IS_WINDOWS` branch — see `win/autostart.py:73 _winreg()` for the pattern.

### The tool registry is the single execution path

Argument validation, coercion, security consultation, timeout, history and bus
events all live in `ToolRegistry.run()`. Nothing else invokes a tool.

If tools were called directly, each of the ~70 tool functions would need its own copy
of that logic, and the ones written *by the model at runtime* (`tool_maker.py`)
would need it too — which is precisely the code you cannot trust to remember. Put
the policy in the dispatcher and a generated tool is a plain function that cannot
skip it. It also means the audit trail and the event stream are complete by
construction.

### The security layer exists but ships disabled

`SecurityConfig` defaults are `mode="open"`, `protected_paths=[]`,
`dangerous_patterns=[]` (`core/config.py:164`). In `"open"` mode every check
short-circuits through `SecurityGate._open_mode()` (`security.py:343`) to
"allowed, no confirmation".

This is an explicit owner decision, not an oversight: JARVIS is meant to drive the
machine it runs on without asking permission. The *engine* is kept complete —
`"guarded"` and `"readonly"` modes, path normalisation with realpath, destructive
command-shape detectors, confirmation callbacks, the unattended policy — so anyone
who wants restrictions has them one config line away, and so the design intent is
recorded rather than deleted.

Two things deliberately survive open mode, because they are capability switches
rather than mode policy: `allow_file_write=False` and `allow_shell=False`. A mode
must not silently hand back a capability the operator turned off.

The audit log is on by default and is **a record, not a restriction**. It never
refuses anything (`security.py:666 audit()`), it never raises, and a single write
failure disables the sink for the process rather than spamming every subsequent
check.

Separately from the policy engine, `delete_path` refuses four whole-tree targets —
filesystem root, home directory, the working directory, and ancestors of the working
directory (`tools/file_tools.py:447`). Read the docstring there: the argument is not
"you may not", it is "this cannot succeed". A recursive delete of any of those
aborts on the first locked file and leaves an unrecoverable half-state, and every
one of them is reachable another way (delete the children, or use the unrestricted
shell). The rationale matters because it is the one guard that must survive the
"no safety rails" policy.

### The backend auto-probe order

`AUTO_PROBE_ORDER` at `jarvis/llm/__init__.py:66` — currently
`("vllm", "ollama", "openai-compat", "transformers", "airllm")`. Confirm what your
checkout has:

```bash
python -c "from jarvis.llm import BACKENDS, AUTO_PROBE_ORDER; print(sorted(BACKENDS)); print(AUTO_PROBE_ORDER)"
```

The ordering principle is **strongest availability signal first, cheapest lie
last**:

| Position | Backend | `is_available()` actually checks | Why here |
|---|---|---|---|
| 1 | `vllm` | `GET {vllm_host}/models`, 1.5 s, verdict cached 5 s | A vLLM server is never up by accident. It is the only backend that serves a whole tree of concurrent agents from one resident copy of the weights (continuous batching) |
| 2 | `ollama` | `GET /api/tags`, 1.5 s (`ollama_backend.py:58`) | The pragmatic default on a CPU-only laptop; parallelises with `OLLAMA_NUM_PARALLEL>1` |
| 3 | `openai-compat` | same `/models` probe against a different base URL | llama.cpp `llama-server`, LM Studio, TGI, or a hosted endpoint, if one is configured |
| 4 | `transformers` | `transformers` **and** `torch` importable | in-process weights |
| 5 | `airllm` | **only** `importlib.util.find_spec("airllm") is not None` (`airllm_backend.py:46`) | it cannot fail, so it must go last |

That last row is the whole point of the ordering. Probing AirLLM first would select
the disk-paged backend — roughly 0.02–0.1 tok/s on the target CPU, i.e. 10–50
*seconds* per token — even with a healthy server running at 4–8 tok/s on the same
weights. `StubBackend` is always available and is the last-resort fallback inside
`_auto_select()` (`llm/__init__.py:141`).

`VLLMBackend` subclasses `OpenAICompatBackend` (`llm/openai_compat.py:239`), which
is a stdlib-only `urllib` client for the `/v1/chat/completions` shape. It is
written for concurrent callers: no mutable per-request state on `self`, a
`threading.Semaphore` enforcing `llm.max_concurrent_requests`, capped exponential
backoff with jitter on connection errors / 429 / 5xx (vLLM answers `503` while it
is still loading weights, which is exactly when a fleet of agents starts up), and a
5-second cache on the availability probe so every agent does not hit `/models`
before every call.

`jarvis/llm/models.py` is a model catalogue and Hugging Face helper module —
`resolve()`, `estimate_footprint()`, `recommend()`, `hf_token()`, `check_access()`,
`local_models()`. See `docs/MODELS.md` for how to use it.

### Threads, not asyncio, for tasks

`TaskManager` uses `ThreadPoolExecutor`s. The tool layer is overwhelmingly blocking
I/O — subprocess, sqlite, filesystem, `urllib` — and a thread pool keeps that honest
without forcing every tool to be written twice, once sync and once async. The bus
supports coroutine handlers so an async UI can still attach.

The one-pool-per-depth arrangement (§4) is the price of that choice: with a single
pool, "a parent settles only after its children" would deadlock. An asyncio design
would not need it, but would need every tool rewritten.

### Everything degrades instead of failing

`app.build()` (`app.py:134`) wraps each subsystem in its own try/except and logs the
failure. `Subsystems.status()` reports what actually came up. An orchestrator is
only created when both an LLM and a memory context exist; otherwise the CLI prints
"JARVIS could not start. Run 'jarvis doctor' to see why."

Correspondingly, every engine exposes `is_available()` that returns `False` rather
than raising when its dependencies are missing — `LLMBackend`, `STTEngine`,
`TTSEngine`, `Embedder`, and the desktop-integration classes in `jarvis/win/`.

### The persona is short on purpose

`prompts.py:14 JARVIS_PERSONA` is about 25 lines. Every token spent on personality
is a token not spent on context, and a local model only needs enough to establish
register. The tool instructions (`prompts.py:46`) are longer than the persona,
because that is where small models actually go wrong.

---

## 8. If you change X, also change Y

| If you change… | You must also change… | Why |
|---|---|---|
| A field in any `*Config` dataclass (`core/config.py`) | `config.example.yaml` | `_apply_mapping()` (`core/config.py:316`) **silently ignores** keys that are not dataclass fields. A documented-but-nonexistent field is a setting that does nothing. See the note below. |
| The tool-call wire format (`protocol.py`) | `TOOL_INSTRUCTIONS` in `prompts.py:46`, and `format_tool_call()` | The prompt teaches the model the syntax the parser expects; `format_tool_call` is what the tests assert against |
| `ToolSpec` / `ToolResult` (`contracts.py`) | `registry.run()` validation, `render_tool_result()`, `TOOL_TEMPLATE` in `tool_maker.py:37` | Generated tools are written against the template; a contract change orphans every tool already on disk in `<data>/tools/` |
| Add a built-in tool module under `jarvis/tools/` | The `modules` tuple in `ToolRegistry.load_builtin()` (`registry.py:466`) | Modules not listed there are never loaded, and the tuple's ORDER decides which implementation wins a name collision — see the note below |
| Add an LLM backend | `BACKENDS` and `AUTO_PROBE_ORDER` (`llm/__init__.py:56`), `_INSTALL_HINTS`, the `--backend` help text in `cli.py:399`, and `cmd_doctor`'s dependency groups (`cli.py:89`) | Otherwise it is unreachable by name and invisible to the doctor |
| Add an STT or TTS engine | `_ENGINE_CLASSES` + `_AUTO_ORDER` (`speech/stt.py:382`) or `_ENGINE_ORDER` + `_make()` (`speech/tts.py:862`) | Same reason |
| `Events` channel names | Any subscriber — currently none in-package, but tests and any UI | Channels are strings; a rename fails silently |
| The persona or the environment block | `Orchestrator._env` (built once in `__init__`, `orchestrator.py:73`) | It is cached per orchestrator; changing `system_summary()` alone will not refresh a live instance |
| `SecurityGate` check signatures | `registry.run()` (`registry.py:363`) and `file_tools._check()` | Both duck-type on `check_tool` / `check_path` / `allows` |
| `DEFAULT_MAX_DEPTH` / `DEFAULT_MAX_TOTAL_TASKS` (`task_manager.py`) | `Orchestrator.__init__`, `delegation_note()` in `subagent.py`, and the `task_status` / `spawn_task` meta-tool payloads | The budget is reported to the model in three places; a limit the prompt does not know about is a limit the model will keep walking into |
| Add a meta-tool to `Orchestrator._register_meta_tools()` | The tool-count expectations in `tests/test_integration.py` | The integration test asserts specific names are registered |
| `Config.home()` / `db_file()` semantics | `app._build_memory()` (`app.py:87`) and `cli.cmd_memory()` (`cli.py:190`), which both pin `cfg.memory.db_path = str(cfg.db_file())` | Otherwise `create_memory()` falls back to the platform default and you get two databases |
| `jarvis/__init__.py:__version__` | `[project].version` in `pyproject.toml` | They currently disagree (`1.0.0` vs `1.1.0`) |

### Two live coupling defects worth knowing about

Both were verified by running the code, not inferred:

**1. `config.example.yaml` documents settings that do not exist.** `VoiceConfig`
(`core/config.py`) has exactly four fields — `wake_words`, `require_wake_word`,
`allow_interrupt`, `greeting`. The example config additionally documents `mode`,
`interrupt_margin`, `follow_up_seconds`, `continuous_timeout`, `preroll_seconds`,
`min_speech_seconds` and `acknowledge`, plus an entire `windows:` section. Because
`_apply_mapping()` skips keys the dataclass does not have, **setting any of them in
`config.yaml` or via `JARVIS_VOICE_MODE=...` does nothing**; `VoiceLoop` reads them
with `getattr(..., default)` and therefore always gets the default. Verify:

```bash
python -c "from dataclasses import fields; from jarvis.core.config import VoiceConfig; print([f.name for f in fields(VoiceConfig)])"
```

The fix is to add the fields to `VoiceConfig` (and a `WindowsConfig` section), not
to delete the documentation.

The same shape, less visibly, in the agent tree: `Orchestrator.__init__` reads
`getattr(config.agent, "max_agent_depth", DEFAULT_MAX_DEPTH)` and
`getattr(config.agent, "max_total_tasks", DEFAULT_MAX_TOTAL_TASKS)`, but neither is
a field on `AgentConfig`. The `getattr` defaults are deliberate defensive coding
against an older config; the consequence today is that **both limits are constants
you cannot configure**. Adding the two fields to `AgentConfig` is all it takes:

```bash
python -c "from dataclasses import fields; from jarvis.core.config import AgentConfig; print([f.name for f in fields(AgentConfig)])"
```

**2. The current user turn appears twice in the prompt.** `Orchestrator.chat()`
calls `context.add_user(user_input)` — which appends to the live window — and then
`context.build(user_input)`, which emits the last `keep_recent_turns` live messages
*and* appends a fresh `Message.user(user_input)`. Verify:

```bash
python -c "
from jarvis.core.config import load_config
from jarvis.memory import create_memory, create_context
import tempfile, os
cfg = load_config(use_env=False); d = tempfile.mkdtemp()
cfg.memory.db_path = os.path.join(d, 'm.db')
ctx = create_context(cfg.memory, store=create_memory(cfg.memory))
ctx.add_user('hello there')
print([(m.role.value, m.content) for m in ctx.build('hello there')])
"
```

It is harmless to correctness (the model sees the question twice) but wastes context
and is confusing when reading a captured prompt.

### The built-in module list, and why its order matters

`ToolRegistry.load_builtin()` (`registry.py:466`) imports eight modules in a
**significant order**:

```
file_tools, system_tools, process_tools, web_tools,
app_tools, input_tools, window_tools, tool_maker
```

Registration uses `replace=True`, so where two modules define the same tool name
**the later one wins**. `app_tools` and `window_tools` both define `list_windows`
and `focus_window`; `window_tools` is listed afterwards deliberately, because its
title matching tries exact then case-insensitive substring and reports ambiguity
rather than picking silently, and its `focus_window` verifies with
`GetForegroundWindow` that focus actually took — Windows blocks focus stealing, so
the naive call succeeds while doing nothing.

A module not in that tuple is never registered, however complete it is. Check what
you actually have:

```bash
python -c "
from jarvis.core.config import load_config
from jarvis.core.security import SecurityGate
from jarvis.tools import create_registry
cfg = load_config(use_env=False)
reg = create_registry(cfg, SecurityGate(cfg.security))
print(len(reg.names())); print(reg.names())
"
```

At the time of writing that prints **72**; the orchestrator adds 7 meta-tools for
**79** in a running system. Both numbers move — run the command.
