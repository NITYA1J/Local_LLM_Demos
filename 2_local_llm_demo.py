import marimo

__generated_with = "0.23.10"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _():
    # ---- Imports and setup -------------------------------------------------
    # Everything in this notebook is one flat file on purpose: no helper
    # package to jump between, just Ollama's plain HTTP API called with the
    # standard library. If you can read this top to bottom, you've seen the
    # whole pipeline.
    import marimo as mo
    import json
    import os
    import urllib.error
    import urllib.request

    OLLAMA_URL = os.environ.get("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
    return OLLAMA_URL, json, mo, urllib


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # A local LLM, powered entirely by Ollama

    **Module 2 of 3** in this folder. Start with `intro_prompting.py` if you
    haven't. No API key, nothing leaves this machine. Four things, in order:

    1. **Ollama basics** — send a prompt to a local model, get a plain-text
       reply back.
    2. **System prompt vs. user prompt** — same question, different
       instructions for *how* to answer. This is the main lever you have
       over a model's behavior before you ever touch RAG or fine-tuning.
    3. **Where local models break down** — small models (3B parameters here,
       vs. 100B+ for something like GPT-4) are fast and private, but they
       fail in specific, learnable ways. Worth seeing on purpose.
    4. **Basic tool use** — one of those failures (arithmetic) fixed by
       letting the model call real Python code instead of guessing.

    The later *Local LLM and RAG* session (a separate folder) builds on this
    by giving the model access to a document corpus it wasn't trained on —
    that's for later.

    Everything below is a direct HTTP call to Ollama's REST API
    (`http://127.0.0.1:11434`, endpoint `/api/generate`) — no SDK, no
    framework.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1 · Ollama basics — send a prompt, get an answer

    `call_ollama_chat` below is the entire integration: one HTTP POST, one
    JSON body in, one JSON body out. Two kinds of text go into that JSON
    body, and they play different roles:

    - **user prompt** (`prompt`) — the actual question or task.
    - **system prompt** (`system`) — instructions about *how* to behave
      while answering: tone, role, format, constraints. It's optional; if
      you leave it out, the model falls back to whatever default behavior
      it was trained/fine-tuned with.

    Section 2 below makes that second one concrete.
    """)
    return


@app.cell
def _(OLLAMA_URL, json, urllib):
    def call_ollama_chat(prompt: str, model: str = "llama3.2:3b", system: str | None = None) -> str:
        """POST to /api/generate and return the model's plain-text reply.

        stream=False means Ollama sends one JSON object back (instead of a
        stream of partial tokens), which keeps this demo simple to read.
        """
        payload = {"model": model, "prompt": prompt, "stream": False}
        if system:
            payload["system"] = system
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            f"{OLLAMA_URL}/api/generate", data=data,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=120) as resp:
            body = json.loads(resp.read().decode("utf-8"))
        return body["response"]


    def list_ollama_models() -> list[str]:
        """GET /api/tags -> names of models already pulled on this server."""
        try:
            with urllib.request.urlopen(f"{OLLAMA_URL}/api/tags", timeout=5) as r:
                body = json.loads(r.read().decode("utf-8"))
            return [m["name"] for m in body.get("models", [])]
        except Exception:
            return []


    def ollama_up() -> bool:
        try:
            urllib.request.urlopen(f"{OLLAMA_URL}/api/tags", timeout=5)
            return True
        except Exception:
            return False

    return call_ollama_chat, list_ollama_models, ollama_up


@app.cell
def _(list_ollama_models, mo, ollama_up):
    _models = list_ollama_models()
    _status = "🟢 Ollama detected" if ollama_up() else "🔴 Ollama not detected — is 'ollama serve' running?"

    # setup_ollama.sh also pulls embedding models for the later RAG session.
    # Those can't answer chat prompts, so keep them out of this dropdown -
    # otherwise whichever model Ollama happens to list first could be one.
    _chat_models = [m for m in _models if "embed" not in m.lower()]
    _preferred = next((m for m in _chat_models if m.startswith("llama3.2:3b")), None)

    chat_model = mo.ui.dropdown(
        options=_chat_models or ["(none found)"],
        value=(_preferred or (_chat_models[0] if _chat_models else "(none found)")),
        label="chat model",
    )
    mo.vstack([mo.md(_status), chat_model])
    return (chat_model,)


@app.cell
def _(mo):
    basic_prompt = mo.ui.text_area(
        value="What's the best way to learn how to use local LLMs?",
        label="prompt",
        full_width=True,
    )
    basic_prompt
    return (basic_prompt,)


@app.cell
def _(basic_button, basic_prompt, call_ollama_chat, chat_model, mo):
    mo.stop(not basic_button.value, mo.md("_Click ▶ Send to Ollama._"))

    import time as _time
    _t0 = _time.time()
    _reply = call_ollama_chat(basic_prompt.value, model=chat_model.value)
    _dt = _time.time() - _t0

    mo.vstack([
        mo.md(f"**Model:** `{chat_model.value}` · **{_dt:.1f}s**"),
        mo.callout(mo.md(_reply), kind="info"),
    ])
    return


@app.cell
def _(mo):
    basic_button = mo.ui.run_button(label="▶ Send to Ollama")
    basic_button
    return (basic_button,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    ## 2 · System prompt vs. user prompt — same question, different persona

    The user prompt below is fixed — it's the same question every time you
    click the button. What changes is the **system prompt**: pick a preset
    (or write your own) and watch the model's tone, structure, and even the
    *content* of the answer shift, even though the underlying question never
    changed.

    This is the cheapest, fastest lever for shaping model behavior — no
    retraining, no RAG, just a different instruction string prepended to
    every request.
    """)
    return


@app.cell
def _(mo):
    SYSTEM_PRESETS = {
        "(none — model default)": "",
        "Helpful assistant": "You are a helpful, friendly assistant. Answer clearly and concisely.",
        "Terse domain expert": "You are a terse subject-matter expert. Answer in bullet points, no preamble, no pleasantries, no hedging.",
        "Explain like I'm 5": "You are explaining things to a curious 5-year-old. Use simple words, short sentences, and a concrete everyday example.",
        "Skeptical scientist": "You are a skeptical scientist. For every claim, note the evidence quality and what could make it wrong. Avoid overconfidence.",
        "Pirate": "You are a pirate captain. Answer entirely in pirate dialect, but keep the actual information accurate.",
    }
    system_preset = mo.ui.dropdown(
        options=list(SYSTEM_PRESETS),
        value="Helpful assistant",
        label="system prompt preset",
    )
    system_preset
    return SYSTEM_PRESETS, system_preset


@app.cell
def _(SYSTEM_PRESETS, mo, system_preset):
    system_prompt_text = mo.ui.text_area(
        value=SYSTEM_PRESETS[system_preset.value],
        label="system prompt (editable — pick a preset above or write your own)",
        full_width=True,
    )
    system_prompt_text
    return (system_prompt_text,)


@app.cell
def _(mo):
    persona_question = mo.ui.text(
        value="Should I update my existing course notes or start from scratch?",
        label="user prompt (held fixed while you try different system prompts)",
        full_width=True,
    )
    persona_question
    return (persona_question,)


@app.cell
def _(
    call_ollama_chat,
    chat_model,
    mo,
    persona_button,
    persona_question,
    system_prompt_text,
):
    mo.stop(not persona_button.value, mo.md("_Pick a system prompt above and click ▶ Send._"))

    import time as _time2
    _t0 = _time2.time()
    _reply2 = call_ollama_chat(
        persona_question.value,
        model=chat_model.value,
        system=system_prompt_text.value or None,
    )
    _dt2 = _time2.time() - _t0

    mo.vstack([
        mo.md(
            f"**Model:** `{chat_model.value}` · **{_dt2:.1f}s** · "
            f"**system prompt:** {('`' + system_prompt_text.value + '`') if system_prompt_text.value else '_(none)_'}"
        ),
        mo.callout(mo.md(_reply2), kind="info"),
        mo.md(
            "_Try switching the preset above and re-running with the exact "
            "same user prompt — same model, same question, only the system "
            "prompt changed._"
        ),
    ])
    return


@app.cell
def _(mo):
    persona_button = mo.ui.run_button(label="▶ Send with this system prompt")
    persona_button
    return (persona_button,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    ## 3 · Where local models break down

    `llama3.2:3b` is a **3-billion-parameter** model running on this
    machine — small enough to be fast and private, but far smaller than the
    ~100B+ parameter models behind most cloud chat products. That size gap
    shows up as specific, somewhat predictable failure modes. Pick an
    example below and see for yourself.
    """)
    return


@app.cell
def _(mo):
    BREAKDOWN_EXAMPLES = {
        "Letter counting (tokenization blindness)": {
            "prompt": "How many times does the letter 'r' appear in the word 'tangential'?",
            "why": (
                "LLMs don't see individual letters — text is split into "
                "sub-word **tokens** before the model ever sees it, so "
                "'tangential' might be 2-3 opaque chunks, not 10 characters. "
                "Character-level counting requires reasoning the model was "
                "never directly trained to do well, and it gets worse as "
                "model size shrinks."
            ),
        },
        "Multi-digit arithmetic": {
            "prompt": "What is 84,637 multiplied by 92,481? Give the exact number.",
            "why": (
                "Small models pattern-match arithmetic from training "
                "examples rather than actually running an algorithm — "
                "there's no built-in calculator. Confident-sounding wrong "
                "digits are common, especially as the numbers get bigger."
            ),
        },
        "Knowledge cutoff (recent events)": {
            "prompt": "Who won the most recent World Cup, and what was the final score?",
            "why": (
                "The model only knows what was in its training data, which "
                "has a cutoff date. Ask about anything after that date and "
                "it will either say it doesn't know, or — more "
                "concerning — confidently guess based on old patterns."
            ),
        },
        "Multi-constraint instructions": {
            "prompt": (
                "Write exactly three sentences about coffee. The first sentence "
                "must start with the letter B. The second sentence must contain "
                "exactly seven words. Do not use the word 'bean' anywhere."
            ),
            "why": (
                "Small models tend to drop one constraint under load when "
                "asked to satisfy several simultaneously — watch which one "
                "it fails, and how confidently it claims to have followed "
                "all of them anyway."
            ),
        },
        "Confident fabrication (hallucination)": {
            "prompt": "Summarize the plot of the novel 'The Glass Meridian' by Aldous Whitfield.",
            "why": (
                "This book and author don't exist. A well-behaved model "
                "says so. A model prone to hallucination will invent a "
                "plausible-sounding plot summary instead of admitting it "
                "doesn't know — a good reminder that fluent text is not the "
                "same as true text."
            ),
        },
    }
    breakdown_choice = mo.ui.dropdown(
        options=list(BREAKDOWN_EXAMPLES),
        value="Letter counting (tokenization blindness)",
        label="failure mode to try",
    )
    breakdown_choice
    return BREAKDOWN_EXAMPLES, breakdown_choice


@app.cell
def _(BREAKDOWN_EXAMPLES, breakdown_choice, mo):
    _example = BREAKDOWN_EXAMPLES[breakdown_choice.value]
    breakdown_prompt = mo.ui.text_area(
        value=_example["prompt"],
        label="prompt (editable)",
        full_width=True,
    )
    mo.vstack([
        mo.callout(mo.md(f"**Why this tends to break:** {_example['why']}"), kind="warn"),
        breakdown_prompt,
    ])
    return (breakdown_prompt,)


@app.cell
def _(mo):
    breakdown_button = mo.ui.run_button(label="▶ Try to break it")
    breakdown_button
    return (breakdown_button,)


@app.cell
def _(breakdown_button, breakdown_prompt, call_ollama_chat, chat_model, mo):
    mo.stop(not breakdown_button.value, mo.md("_Click ▶ Try to break it._"))

    import time as _time3
    _t0 = _time3.time()
    _reply3 = call_ollama_chat(breakdown_prompt.value, model=chat_model.value)
    _dt3 = _time3.time() - _t0

    mo.vstack([
        mo.md(f"**Model:** `{chat_model.value}` · **{_dt3:.1f}s**"),
        mo.callout(mo.md(_reply3), kind="info"),
        mo.md(
            "_Check the answer carefully — these failures are usually "
            "wrong in a fluent, confident way, not an obviously broken "
            "way. That's what makes them worth demonstrating live._"
        ),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    ## 4 · Basic tool use — giving the model a calculator

    Section 3 showed the model guessing at multi-digit arithmetic and
    getting it wrong with total confidence. **Tool use** fixes that class
    of problem: instead of asking the model to *compute* an answer, teach
    it to *ask for* a calculation, then run real Python code to get the
    exact answer.

    That's three plain steps, and — unlike Section 1's `call_ollama_chat`,
    which is one function — each step below gets its own cell, so you can
    watch the hand-off happen instead of reading it out of a black box:

    1. **Ask** (Step 1 cell) — send the question, with a system prompt
       telling the model it may reply with `CALC: <expression>` instead of
       guessing.
    2. **Check and calculate** (Step 2 cell) — if the reply starts with
       `CALC:`, pull out the expression and evaluate it for real in Python
       (`safe_calculate`, just below) instead of trusting the model's own
       arithmetic.
    3. **Answer again, with the real number** (also Step 2 cell) — send a
       follow-up that hands back the exact result, so the final answer uses
       real math instead of a guess.

    This is the same basic idea behind "function calling" / "tool use" in
    bigger frameworks — just written out by hand so every step is visible.
    """)
    return


@app.cell
def _():
    import re as _re

    def safe_calculate(expr: str) -> float:
        """Evaluate a plain arithmetic expression like '84637 * 92481'.

        We can't just hand model-generated text to Python's real eval() -
        that would let it run arbitrary code, not just math. So this does
        two things first: checks the string contains ONLY digits, decimal
        points, whitespace, thousands-separator commas, and the symbols
        + - * / ** ( ) - nothing else - and only then calls eval() with its
        built-in functions disabled. Between the character whitelist and
        the disabled builtins, there's nothing left for it to do except
        arithmetic.

        The system prompt asks the model for numbers without commas (e.g.
        "84637", not "84,637"), but it doesn't always comply - especially
        when the question itself was written with commas. Rather than fail
        on that, we just strip commas before evaluating: a comma between
        digits is unambiguous here and never means anything else.
        """
        expr = expr.replace(",", "")
        only_math_characters = r"[0-9+\-*/(). ]+"
        if not _re.fullmatch(only_math_characters, expr):
            raise ValueError(f"Expression contains disallowed characters: {expr!r}")
        return eval(expr, {"__builtins__": {}}, {})

    return (safe_calculate,)


@app.cell
def _():
    TOOL_SYSTEM_PROMPT = (
        "You have access to a calculator tool for arithmetic. If answering "
        "the question requires a calculation, respond with ONLY a single "
        "line in the exact form 'CALC: <expression>' using plain "
        "+ - * / ** and parentheses (e.g. 'CALC: 84637 * 92481') - no "
        "commas in numbers, no explanation, no extra text. If the question "
        "does NOT require a calculation, just answer it directly and "
        "normally."
    )
    return (TOOL_SYSTEM_PROMPT,)


@app.cell
def _(mo):
    tool_question = mo.ui.text_area(
        value="What is 84,637 multiplied by 92,481?",
        label="prompt",
        full_width=True,
    )
    tool_question
    return (tool_question,)


@app.cell
def _(mo):
    use_tool = mo.ui.switch(value=True, label="use calculator tool")
    use_tool
    return (use_tool,)


@app.cell
def _(mo):
    tool_button = mo.ui.run_button(label="▶ Ask")
    tool_button
    return (tool_button,)


@app.cell
def _(TOOL_SYSTEM_PROMPT, call_ollama_chat, chat_model, mo, tool_button, tool_question, use_tool):
    mo.stop(not tool_button.value, mo.md("_Click ▶ Ask. Try it with the switch on, then off, on the same question._"))

    import time as _time4

    # --- Step 1: ask -------------------------------------------------------
    # With the tool switch on, TOOL_SYSTEM_PROMPT tells the model it may
    # reply with "CALC: <expression>" instead of guessing. With it off, we
    # ask exactly the way Section 3 did, so the two are a fair comparison.
    system_for_this_call = TOOL_SYSTEM_PROMPT if use_tool.value else None
    _t0 = _time4.time()
    first_reply = call_ollama_chat(tool_question.value, model=chat_model.value, system=system_for_this_call)
    step1_seconds = _time4.time() - _t0
    model_requested_calc = use_tool.value and first_reply.strip().upper().startswith("CALC:")

    mo.md(f"**Step 1 — model's first reply** ({step1_seconds:.1f}s):\n\n> {first_reply}")
    return first_reply, model_requested_calc


@app.cell
def _(
    call_ollama_chat,
    chat_model,
    first_reply,
    mo,
    model_requested_calc,
    safe_calculate,
    tool_question,
):
    import time as _time5

    # --- Step 2: check, calculate, and (if needed) ask again --------------
    if not model_requested_calc:
        # Either the tool was off, or the model decided no calculation was
        # needed - either way, the Step 1 reply above is the final answer.
        final_answer = first_reply
        detail = "**Step 2 —** no calculation requested; the Step 1 reply above is final."
        callout_kind = "warn"
    else:
        expression = first_reply.strip().split(":", 1)[1].strip()
        try:
            exact_result = safe_calculate(expression)
        except Exception as e:
            final_answer = f"(Model requested `{expression}`, but it could not be safely evaluated: {e})"
            detail = f"**Step 2 —** tool call failed: {e}"
            callout_kind = "warn"
        else:
            # This is the hand-off: give the model back the exact number so
            # its final answer uses real math instead of recomputing it
            # itself, the way it did in Section 3.
            follow_up = (
                f"Question: {tool_question.value}\n"
                f"You requested the calculation `{expression}`, and the exact "
                f"result is {exact_result}. Give the final answer to the "
                f"question using this exact result - do not recompute it "
                f"yourself."
            )
            _t0 = _time5.time()
            final_answer = call_ollama_chat(follow_up, model=chat_model.value)
            step2_seconds = _time5.time() - _t0
            detail = (
                f"**Step 2 — tool called** `{expression}` → **exact result:** "
                f"`{exact_result}` (follow-up took {step2_seconds:.1f}s)"
            )
            callout_kind = "success"

    mo.vstack([
        mo.md(detail),
        mo.callout(mo.md(final_answer), kind=callout_kind),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    ### Recap

    Four levers you've now seen, all without leaving this one flat file:

    1. **User prompt** — the question itself.
    2. **System prompt** — instructions for *how* to answer; free to change,
       no retraining required, and it can shift tone, format, even
       willingness to hedge.
    3. **Model capability ceiling** — no amount of prompt engineering fixes
       tokenization blindness or a knowledge cutoff. Some failures need a
       bigger model, and some need external tools.
    4. **Tool use** — the fix for exactly the arithmetic case above: don't
       ask the model to compute, ask it to *delegate* to code that
       actually computes. The same pattern extends to a web search tool, a
       database query tool, or a document retrieval tool. That's what RAG
       is, structurally: a retrieval tool the model calls before answering,
       instead of a calculator.

    **Next:** `structured_output.py` (module 3) — controlling the *format* of
    what comes back, so you get clean data instead of prose. After that, the
    *Local LLM and RAG* session.
    """)
    return


if __name__ == "__main__":
    app.run()
