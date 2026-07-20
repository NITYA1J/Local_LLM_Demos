import marimo

__generated_with = "0.23.14"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _():
    # ---- Imports and setup -------------------------------------------------
    # Same philosophy as the rest of this folder: one flat file, no helper
    # package, no SDK. Every call to the model is a plain HTTP POST to
    # Ollama, written out with the standard library so you can read the
    # whole thing top to bottom.
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
    # Intro: prompting a local LLM and turning the knobs

    **Module 1 of 3** in this folder — the gentlest possible starting point
    for the Applied AI Summer Workshop. No API key, nothing leaves this
    machine.

    Two ideas, and that's the whole notebook:

    1. **Send a prompt, get an answer** — the smallest possible call to a
       model running locally.
    2. **Set the model's parameters from Python** — `temperature`, `top_p`,
       `top_k`, `seed`, `num_predict`, and `stop`. These are the dials that
       control *how* the model generates text: how random, how long, how
       repeatable. You pass them as a plain Python dictionary.

    At the end there's a short **hands-on exercise** where you set the dials
    yourself.

    Everything below is a direct HTTP call to Ollama's REST API
    (`http://127.0.0.1:11434`, endpoint `/api/generate`) — no SDK, no
    framework.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1 · The smallest possible call

    `call_ollama` below is the entire integration: one HTTP POST, one JSON
    body in, one JSON body out. It takes three things you'll recognize, plus
    one new one:

    - **`prompt`** — the question or task (the *user prompt*).
    - **`system`** — optional instructions about *how* to answer.
    - **`options`** — the new part: a dictionary of generation parameters
      (temperature, seed, and so on). Leave it empty and Ollama uses its own
      defaults. The rest of this notebook is all about what goes in here.
    """)
    return


@app.cell
def _(OLLAMA_URL, json, urllib):
    def call_ollama(
        prompt: str,
        model: str = "llama3.2:3b",
        system: str | None = None,
        options: dict | None = None,
    ) -> str:
        """POST to /api/generate and return the model's plain-text reply.

        `options` is a plain dict of generation parameters that Ollama passes
        straight to the model — e.g. {"temperature": 0.8, "seed": 42}. If it's
        None, the model uses its built-in defaults. stream=False means Ollama
        sends back one JSON object instead of a stream of partial tokens,
        which keeps this easy to read.
        """
        payload = {"model": model, "prompt": prompt, "stream": False}
        if system:
            payload["system"] = system
        if options:
            payload["options"] = options
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

    return call_ollama, list_ollama_models, ollama_up


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
    intro_prompt = mo.ui.text_area(
        value="In one sentence, what is a large language model?",
        label="prompt",
        full_width=True,
    )
    intro_prompt
    return (intro_prompt,)


@app.cell
def _(mo):
    intro_button = mo.ui.run_button(label="▶ Send to Ollama")
    intro_button
    return (intro_button,)


@app.cell
def _(call_ollama, chat_model, intro_button, intro_prompt, mo):
    mo.stop(not intro_button.value, mo.md("_Click ▶ Send to Ollama._"))

    import time as _time
    _t0 = _time.time()
    _reply = call_ollama(intro_prompt.value, model=chat_model.value)
    _dt = _time.time() - _t0

    mo.vstack([
        mo.md(f"**Model:** `{chat_model.value}` · **{_dt:.1f}s** · _(no options set — model defaults)_"),
        mo.callout(mo.md(_reply), kind="info"),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    ## 2 · Temperature — the randomness dial

    **Temperature** is the parameter you'll reach for most. Roughly: it
    controls how much the model is allowed to gamble when picking each next
    word.

    - **Low (0.0–0.3)** — nearly deterministic. Picks the most likely next
      word almost every time. Good for facts, code, and anything where you
      want the same answer twice.
    - **High (0.8–1.5)** — adventurous. Willing to pick less likely words,
      which reads as more creative — and more prone to going off the rails.

    To *see* it, the cell below sends the **same prompt twice** at whatever
    temperature you choose. At low temperature the two replies will look
    nearly identical; crank it up and they'll drift apart.
    """)
    return


@app.cell
def _(mo):
    temp_slider = mo.ui.slider(
        start=0.0, stop=1.5, step=0.1, value=0.8,
        label="temperature", show_value=True,
    )
    temp_prompt = mo.ui.text(
        value="Write a one-sentence bedtime story about a sleepy professor.",
        label="prompt",
        full_width=True,
    )
    temp_button = mo.ui.run_button(label="▶ Send this prompt twice")
    mo.vstack([temp_slider, temp_prompt, temp_button])
    return temp_button, temp_prompt, temp_slider


@app.cell
def _(call_ollama, chat_model, mo, temp_button, temp_prompt, temp_slider):
    mo.stop(not temp_button.value, mo.md("_Set a temperature and click ▶. Try it once low (0.0) and once high (1.2)._"))

    # We build the options dict here — this is the whole point of the
    # section. Everything else is display.
    _options = {"temperature": temp_slider.value}
    _reply_a = call_ollama(temp_prompt.value, model=chat_model.value, options=_options)
    _reply_b = call_ollama(temp_prompt.value, model=chat_model.value, options=_options)

    mo.vstack([
        mo.md(f"**temperature = {temp_slider.value}** · same prompt, two runs:"),
        mo.callout(mo.md(f"**Run 1:** {_reply_a}"), kind="info"),
        mo.callout(mo.md(f"**Run 2:** {_reply_b}"), kind="info"),
        mo.md(
            "_Low temperature → the two runs look nearly the same. "
            "High temperature → they diverge. That divergence is the "
            "randomness you just dialed in._"
        ),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    ## 3 · top_p and top_k — narrowing the candidate pool

    Temperature decides *how much* to gamble; **top_p** and **top_k** decide
    *which words are even on the table* before that gamble happens. At each
    step the model has a ranked list of possible next words, and these two
    trim that list:

    - **`top_k`** — keep only the *k* most likely words (e.g. `top_k = 10`
      means "only ever consider the 10 best candidates"). Lower = safer.
    - **`top_p`** — keep the smallest set of top words whose probabilities
      add up to *p* (e.g. `top_p = 0.9` means "the most likely words that
      together cover 90% of the probability"). Also called *nucleus
      sampling*. Lower = safer.

    They work alongside temperature. A common recipe for focused output is a
    low temperature **and** a modest `top_p`/`top_k`; for brainstorming, open
    all three up.
    """)
    return


@app.cell
def _(mo):
    top_p_slider = mo.ui.slider(
        start=0.1, stop=1.0, step=0.05, value=0.9,
        label="top_p", show_value=True,
    )
    top_k_slider = mo.ui.slider(
        start=1, stop=100, step=1, value=40,
        label="top_k", show_value=True,
    )
    sampling_prompt = mo.ui.text(
        value="Suggest a creative name for a campus coffee shop.",
        label="prompt",
        full_width=True,
    )
    sampling_button = mo.ui.run_button(label="▶ Send with these settings")
    mo.vstack([top_p_slider, top_k_slider, sampling_prompt, sampling_button])
    return sampling_button, sampling_prompt, top_k_slider, top_p_slider


@app.cell
def _(
    call_ollama,
    chat_model,
    mo,
    sampling_button,
    sampling_prompt,
    top_k_slider,
    top_p_slider,
):
    mo.stop(not sampling_button.value, mo.md("_Adjust top_p / top_k and click ▶. Try top_k = 1 (most extreme) vs. top_k = 100._"))

    _options = {"top_p": top_p_slider.value, "top_k": int(top_k_slider.value)}
    _reply = call_ollama(sampling_prompt.value, model=chat_model.value, options=_options)

    mo.vstack([
        mo.md(f"**top_p = {top_p_slider.value}** · **top_k = {int(top_k_slider.value)}**"),
        mo.callout(mo.md(_reply), kind="info"),
        mo.md(
            "_With `top_k = 1` the model is forced to take its single most "
            "likely word every time — output gets repetitive and 'safe'. "
            "Wider settings let more variety through._"
        ),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    ## 4 · seed — making randomness repeatable

    Temperature makes output random — but sometimes you want *repeatable*
    randomness, so a demo or a test gives the same result every time. The
    **`seed`** parameter fixes the starting point of the random number
    generator. Same seed + same parameters + same prompt → the same output,
    every run.

    The cell below sends the same prompt **twice with the same seed**. Even
    at a high temperature, the two replies should come back **identical** —
    that's the seed at work. Change the seed number and they change together.
    """)
    return


@app.cell
def _(mo):
    seed_number = mo.ui.number(value=42, start=0, stop=999999, label="seed")
    seed_temp_slider = mo.ui.slider(
        start=0.0, stop=1.5, step=0.1, value=1.0,
        label="temperature", show_value=True,
    )
    seed_prompt = mo.ui.text(
        value="Invent a name and one-line backstory for a time traveling professor.",
        label="prompt",
        full_width=True,
    )
    seed_button = mo.ui.run_button(label="▶ Send twice with this seed")
    mo.vstack([seed_number, seed_temp_slider, seed_prompt, seed_button])
    return seed_button, seed_number, seed_prompt, seed_temp_slider


@app.cell
def _(
    call_ollama,
    chat_model,
    mo,
    seed_button,
    seed_number,
    seed_prompt,
    seed_temp_slider,
):
    mo.stop(not seed_button.value, mo.md("_Pick a seed and click ▶. Note the two runs match; then change the seed and re-run._"))

    _options = {"temperature": seed_temp_slider.value, "seed": int(seed_number.value)}
    _reply_a = call_ollama(seed_prompt.value, model=chat_model.value, options=_options)
    _reply_b = call_ollama(seed_prompt.value, model=chat_model.value, options=_options)
    _identical = _reply_a.strip() == _reply_b.strip()

    mo.vstack([
        mo.md(f"**seed = {int(seed_number.value)}** · **temperature = {seed_temp_slider.value}**"),
        mo.callout(mo.md(f"**Run 1:** {_reply_a}"), kind="info"),
        mo.callout(mo.md(f"**Run 2:** {_reply_b}"), kind="info"),
        mo.callout(
            mo.md("✅ **Identical** — the seed made the randomness repeatable." if _identical
                  else "⚠️ Not identical this time — some model/runtime settings can still introduce tiny differences."),
            kind="success" if _identical else "warn",
        ),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    ## 5 · num_predict and stop — controlling length

    Two dials that decide *when the model stops talking*:

    - **`num_predict`** — a hard cap on how many tokens (roughly, word
      pieces) the model may generate. Small values force short answers; it's
      also the main knob for keeping a slow local model responsive.
    - **`stop`** — a list of strings that, if the model produces one, cut the
      generation off immediately. Handy for structured output: stop at a
      marker like `"END"`, or at `"3."` to cut a numbered list short. (In
      code you can also stop at `"\n"` for a single line — but you can't type
      a real newline into the box below, so try a word instead.)

    Set a small `num_predict` below and watch the answer get truncated
    mid-thought — the cap doesn't ask for a shorter answer, it just stops
    generation when it's hit.
    """)
    return


@app.cell
def _(mo):
    num_predict_slider = mo.ui.slider(
        start=8, stop=300, step=8, value=32,
        label="num_predict (max tokens)", show_value=True,
    )
    stop_text = mo.ui.text(
        value="",
        label="stop sequence (optional — try a word like 'END' or '3.')",
        full_width=True,
    )
    length_prompt = mo.ui.text(
        value="List three tips for using AI effectively.",
        label="prompt",
        full_width=True,
    )
    length_button = mo.ui.run_button(label="▶ Send with length limit")
    mo.vstack([num_predict_slider, stop_text, length_prompt, length_button])
    return length_button, length_prompt, num_predict_slider, stop_text


@app.cell
def _(
    call_ollama,
    chat_model,
    length_button,
    length_prompt,
    mo,
    num_predict_slider,
    stop_text,
):
    mo.stop(not length_button.value, mo.md("_Set a small num_predict (say 16) and click ▶ to see the answer cut off._"))

    _options = {"num_predict": int(num_predict_slider.value)}
    if stop_text.value.strip():
        _options["stop"] = [stop_text.value]
    _reply = call_ollama(length_prompt.value, model=chat_model.value, options=_options)

    mo.vstack([
        mo.md(
            f"**num_predict = {int(num_predict_slider.value)}**"
            + (f" · **stop = `{stop_text.value}`**" if stop_text.value.strip() else "")
        ),
        mo.callout(mo.md(_reply), kind="info"),
        mo.md("_A low cap truncates the reply mid-sentence — the model isn't asked to be brief, it's just cut off._"),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    ## 6 · Your turn — build the options dict yourself

    Everything above just assembled one `options` dictionary and passed it to
    `call_ollama`. Now you do it. In the cell below, **fill in the blanks**
    marked `# TODO` so that:

    - the answer is **highly creative / random** (think about temperature),
    - it's **repeatable** across runs (think about seed),
    - and it's **capped to a short length** (think about num_predict).

    Edit the values, then click ▶. The solution is in the next cell if you
    want to check yourself.
    """)
    return


@app.cell
def _(mo):
    exercise_button = mo.ui.run_button(label="▶ Run my settings")
    exercise_button
    return (exercise_button,)


@app.cell
def _(call_ollama, chat_model, exercise_button, mo):
    mo.stop(not exercise_button.value, mo.md("_Fill in the TODOs below, then click ▶ Run my settings._"))

    # ----------------------------------------------------------------------
    # TODO: fill in the three values. Delete "None" and put a number in.
    #   - high temperature for creativity        (try ~1.2)
    #   - a fixed seed so runs are repeatable     (any integer, e.g. 7)
    #   - a small num_predict to keep it short    (try ~40)
    # ----------------------------------------------------------------------
    my_options = {
        "temperature": None,   # TODO: e.g. 1.2
        "seed": None,          # TODO: e.g. 7
        "num_predict": None,   # TODO: e.g. 40
    }

    my_prompt = "Give me a fun team name for an environmental engineering student club."  # change if you like

    # Drop any blanks you haven't filled in yet so the call still works.
    _clean_options = {k: v for k, v in my_options.items() if v is not None}
    _reply = call_ollama(my_prompt, model=chat_model.value, options=_clean_options)

    mo.vstack([
        mo.md(f"**Your options:** `{_clean_options}`"),
        mo.callout(mo.md(_reply), kind="info"),
        mo.md(
            "_Still seeing `{}` for your options? The TODOs are still `None`. "
            "Fill them in and re-run — then run it a second time and confirm "
            "the seed makes the answer repeat._"
        ),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.accordion({
        "▶ Show a worked solution": mo.md(r"""
        One set of values that satisfies all three goals:

        ```python
        my_options = {
            "temperature": 1.2,   # high → creative / random
            "seed": 7,            # fixed → repeatable across runs
            "num_predict": 40,    # small → capped length
        }
        ```

        With these, running the cell twice gives the **same** short, creative
        answer both times: high temperature supplies the creativity, the seed
        makes that specific creative result repeatable, and `num_predict`
        keeps it brief. Change the seed and you get a different — but again
        repeatable — answer.
        """)
    })
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    ### Recap

    You've now seen the two things this notebook set out to show:

    1. **Prompting** — one function, `call_ollama`, is the whole integration:
       a prompt goes in, text comes back.
    2. **Parameters** — everything else is just building an `options`
       dictionary and passing it along:

       | dial | what it controls |
       |---|---|
       | `temperature` | how random / creative the output is |
       | `top_p`, `top_k` | which candidate words are even considered |
       | `seed` | makes random output repeatable |
       | `num_predict` | hard cap on output length |
       | `stop` | strings that cut generation off early |

    **Next:** `local_llm_demo.py` (module 2) builds on this — system prompts,
    where small models break down, and basic tool use. Then
    `structured_output.py` (module 3) covers controlling the output format.
    """)
    return


if __name__ == "__main__":
    app.run()
