import marimo

__generated_with = "0.23.10"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _():
    # ---- Imports and setup -------------------------------------------------
    # Same philosophy as the rest of this folder: one flat file, no helper
    # package, raw HTTP to Ollama with the standard library. The only new
    # dependency is pydantic, used only in the last section — and it's
    # imported defensively so the notebook still runs without it.
    import marimo as mo
    import json
    import os
    import urllib.error
    import urllib.request

    try:
        from pydantic import BaseModel, ValidationError
        PYDANTIC_AVAILABLE = True
    except Exception:
        BaseModel = None
        ValidationError = Exception
        PYDANTIC_AVAILABLE = False

    OLLAMA_URL = os.environ.get("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
    return (
        BaseModel,
        OLLAMA_URL,
        PYDANTIC_AVAILABLE,
        ValidationError,
        json,
        mo,
        urllib,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Structured output — controlling the *shape* of the reply

    **Module 3 of 3** in this folder. So far we've controlled
    *what* the model says (prompts) and *how randomly* it says it
    (parameters). This notebook controls the **format** of the answer: how do
    you get back clean, machine-readable data — JSON with exactly the fields
    you need — instead of a paragraph of prose you have to parse by hand?

    We build it as a four-step progression, each step fixing the previous
    one's weakness:

    1. **Naive prompt** — just *ask* for JSON. See how it breaks.
    2. **`format: "json"`** — make Ollama guarantee *valid* JSON.
    3. **JSON Schema** — make the JSON the *right shape* (your fields, your
       types).
    4. **Pydantic** — parse and *validate* it into a typed Python object.

    Everything is a direct HTTP call to Ollama (`/api/generate`); the only
    new ingredient is one extra field in the request body: `format`.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## The one addition: a `format` field

    `call_ollama` below is the same tiny function from the other notebooks,
    with one new argument — `response_format` — that becomes the request's
    `format` field. It can be:

    - **`None`** — normal free-text reply (the default).
    - **`"json"`** — Ollama constrains decoding so the reply is guaranteed to
      be syntactically valid JSON.
    - **a JSON Schema `dict`** — the reply is constrained to match that exact
      schema (fields, types, required keys).

    That single field is the whole mechanism. The rest of the notebook is
    just deciding what to put in it.
    """)
    return


@app.cell
def _(OLLAMA_URL, json, urllib):
    def call_ollama(
        prompt: str,
        model: str = "llama3.2:3b",
        system: str | None = None,
        options: dict | None = None,
        response_format=None,
    ) -> str:
        """POST to /api/generate and return the model's plain-text reply.

        `response_format` maps to Ollama's `format` field:
          - None    -> free text
          - "json"  -> guaranteed valid JSON
          - dict    -> a JSON Schema the output must conform to
        """
        payload = {"model": model, "prompt": prompt, "stream": False}
        if system:
            payload["system"] = system
        if options:
            payload["options"] = options
        if response_format is not None:
            payload["format"] = response_format
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

    return (call_ollama, list_ollama_models, ollama_up)


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


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    ## 1 · Naive prompt — just ask for JSON

    The obvious first attempt: write "respond in JSON" into the prompt and
    hope. The cell below does exactly that with **no `format` field**, then
    tries to `json.loads()` the reply and reports whether it actually parsed.

    Run it a few times. Common failures: the model wraps the JSON in prose
    ("Here is the JSON you requested:"), fences it in ```` ```json ````
    markdown, or adds a trailing comma — any of which breaks a strict parser.
    """)
    return


@app.cell
def _(mo):
    naive_prompt = mo.ui.text_area(
        value=(
            "Give me three programming languages, each with a one-word "
            "description. Respond in JSON."
        ),
        label="prompt",
        full_width=True,
    )
    naive_button = mo.ui.run_button(label="▶ Ask (no format field)")
    mo.vstack([naive_prompt, naive_button])
    return naive_button, naive_prompt


@app.cell
def _(call_ollama, chat_model, json, mo, naive_button, naive_prompt):
    mo.stop(not naive_button.value, mo.md("_Click ▶ to send the plain prompt. Try it a few times — the failures are intermittent._"))

    _raw = call_ollama(naive_prompt.value, model=chat_model.value)  # no response_format
    try:
        _parsed = json.loads(_raw)
        _verdict = mo.callout(
            mo.md(f"✅ **Parsed cleanly this time.**\n\n```json\n{json.dumps(_parsed, indent=2)}\n```"),
            kind="success",
        )
    except json.JSONDecodeError as e:
        _verdict = mo.callout(
            mo.md(f"❌ **`json.loads()` failed:** {e}\n\nThe reply is text, not clean JSON — that's the problem this module fixes."),
            kind="danger",
        )

    mo.vstack([
        mo.md("**Raw reply from the model:**"),
        mo.callout(mo.md(f"```\n{_raw}\n```"), kind="neutral"),
        _verdict,
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    ## 2 · `format: "json"` — guarantee *valid* JSON

    Same prompt, but now we pass `response_format="json"`. Ollama constrains
    the model's decoding so the output **must** be syntactically valid JSON —
    no prose, no fences, no trailing commas. `json.loads()` will succeed.

    The catch: valid JSON is not the *same* JSON every time. The model still
    chooses its own keys and structure, so one run might give
    `{"languages": [...]}` and another a bare list. Valid, but unpredictable —
    which is what Section 3 fixes.
    """)
    return


@app.cell
def _(mo):
    jsonmode_prompt = mo.ui.text_area(
        value=(
            "Give me three programming languages, each with a one-word "
            "description. Respond in JSON."
        ),
        label="prompt",
        full_width=True,
    )
    jsonmode_button = mo.ui.run_button(label='▶ Ask with format="json"')
    mo.vstack([jsonmode_prompt, jsonmode_button])
    return jsonmode_button, jsonmode_prompt


@app.cell
def _(call_ollama, chat_model, json, jsonmode_button, jsonmode_prompt, mo):
    mo.stop(not jsonmode_button.value, mo.md("_Click ▶. It parses every time now — but re-run a few times and watch the *shape* wander._"))

    _raw = call_ollama(jsonmode_prompt.value, model=chat_model.value, response_format="json")
    _parsed = json.loads(_raw)  # safe: format="json" guarantees this parses

    mo.vstack([
        mo.md("**Valid JSON, guaranteed:**"),
        mo.callout(mo.md(f"```json\n{json.dumps(_parsed, indent=2)}\n```"), kind="success"),
        mo.md(
            "_Notice the top-level keys the model chose. Re-run — they may "
            "change. You get valid JSON, but not a shape you can rely on._"
        ),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    ## 3 · JSON Schema — lock down the *shape*

    Now we hand Ollama an actual **JSON Schema** as the format. The output is
    constrained to match it: the exact fields you name, with the types you
    declare, and every `required` key present. This is the real workhorse of
    structured output — extracting reliable, predictable records from
    free text.

    The example below extracts a person's details from a sentence. The schema
    says: an object with `name` (string), `age` (integer), `city` (string),
    and `occupation` (string) — all required.

    > **Version note:** schema-constrained output needs a reasonably recent
    > Ollama (roughly late-2024 or newer). If the cell reports an error
    > instead of a result, the server is likely too old — `format:"json"`
    > from Section 2 still works everywhere.
    """)
    return


@app.cell
def _():
    # A plain JSON Schema, written as a Python dict. This is what gets passed
    # to Ollama as the `format` field.
    PERSON_SCHEMA = {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "age": {"type": "integer"},
            "city": {"type": "string"},
            "occupation": {"type": "string"},
        },
        "required": ["name", "age", "city", "occupation"],
    }
    return (PERSON_SCHEMA,)


@app.cell
def _(PERSON_SCHEMA, json, mo):
    schema_prompt = mo.ui.text_area(
        value=(
            "Extract the person's details from this text:\n\n"
            "Maria is a 34-year-old software engineer who lives in Austin, Texas."
        ),
        label="prompt",
        full_width=True,
    )
    schema_button = mo.ui.run_button(label="▶ Extract with schema")
    mo.vstack([
        mo.md(f"**Schema being enforced:**\n\n```json\n{json.dumps(PERSON_SCHEMA, indent=2)}\n```"),
        schema_prompt,
        schema_button,
    ])
    return schema_button, schema_prompt


@app.cell
def _(
    PERSON_SCHEMA,
    call_ollama,
    chat_model,
    json,
    mo,
    schema_button,
    schema_prompt,
    urllib,
):
    mo.stop(not schema_button.value, mo.md("_Click ▶ to extract structured data conforming to the schema._"))

    try:
        _raw = call_ollama(
            schema_prompt.value, model=chat_model.value, response_format=PERSON_SCHEMA
        )
        _parsed = json.loads(_raw)
        _missing = [k for k in PERSON_SCHEMA["required"] if k not in _parsed]
        _shape_ok = not _missing
        _out = mo.vstack([
            mo.md("**Output, conforming to the schema:**"),
            mo.callout(mo.md(f"```json\n{json.dumps(_parsed, indent=2)}\n```"), kind="success"),
            mo.callout(
                mo.md(
                    "✅ Every required field is present — same keys, every run."
                    if _shape_ok else
                    f"⚠️ Missing required field(s): {_missing}"
                ),
                kind="success" if _shape_ok else "warn",
            ),
        ])
    except urllib.error.HTTPError as e:
        _out = mo.callout(
            mo.md(
                f"❌ **Ollama rejected the schema (HTTP {e.code}).** This "
                "server is probably too old for schema-constrained output. "
                "`format:\"json\"` (Section 2) still works — upgrade Ollama "
                "to use full schemas."
            ),
            kind="danger",
        )

    _out
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    ## 4 · Pydantic — a typed, *validated* Python object

    JSON Schema gets you the right shape on the wire. **Pydantic** takes the
    last step: you define the shape once as a typed Python class, let it
    *generate* the schema for you, and then *validate* the reply back into a
    real object — so downstream code gets `product.price_usd` (a `float`),
    not `data["price_usd"]` (a `who-knows`).

    Two payoffs over raw schema:

    - **One source of truth** — `Product.model_json_schema()` produces the
      schema, so the class and the constraint can't drift apart.
    - **Validation** — if anything slips through malformed, `model_validate`
      raises instead of silently handing you bad data.
    """)
    return


@app.cell
def _(BaseModel, PYDANTIC_AVAILABLE):
    # Define the target shape as a typed class — but only if pydantic is
    # installed, so the notebook still loads without it.
    if PYDANTIC_AVAILABLE:
        class Product(BaseModel):
            name: str
            price_usd: float
            in_stock: bool
            tags: list[str]
    else:
        Product = None
    return (Product,)


@app.cell
def _(PYDANTIC_AVAILABLE, Product, json, mo):
    if not PYDANTIC_AVAILABLE:
        _view = mo.callout(
            mo.md(
                "⚠️ **pydantic isn't installed**, so this section is inactive. "
                "Install it with `pip install pydantic` (it's in this folder's "
                "`requirements.txt`) and re-run. Sections 1–3 don't need it."
            ),
            kind="warn",
        )
        pydantic_prompt = None
        pydantic_button = None
    else:
        pydantic_prompt = mo.ui.text_area(
            value=(
                "Extract the product details from this text:\n\n"
                "The UltraGrip water bottle costs $24.99, is currently in "
                "stock, and is great for hiking, cycling, and gym use."
            ),
            label="prompt",
            full_width=True,
        )
        pydantic_button = mo.ui.run_button(label="▶ Extract into a Product object")
        _schema_preview = json.dumps(Product.model_json_schema(), indent=2)
        _view = mo.vstack([
            mo.md(f"**Schema auto-generated from the `Product` class:**\n\n```json\n{_schema_preview}\n```"),
            pydantic_prompt,
            pydantic_button,
        ])
    _view
    return pydantic_button, pydantic_prompt


@app.cell
def _(
    PYDANTIC_AVAILABLE,
    Product,
    ValidationError,
    call_ollama,
    chat_model,
    mo,
    pydantic_button,
    pydantic_prompt,
    urllib,
):
    mo.stop(not PYDANTIC_AVAILABLE, mo.md("_(pydantic not installed — see the note above.)_"))
    mo.stop(not pydantic_button.value, mo.md("_Click ▶ to extract and validate into a typed object._"))

    try:
        _raw = call_ollama(
            pydantic_prompt.value,
            model=chat_model.value,
            response_format=Product.model_json_schema(),
        )
        # Validate the raw JSON string straight into a typed Product instance.
        _product = Product.model_validate_json(_raw)
        _out = mo.vstack([
            mo.md("**Validated `Product` object — now typed Python, not text:**"),
            mo.callout(
                mo.md(
                    f"- `product.name` → `{_product.name!r}` ({type(_product.name).__name__})\n"
                    f"- `product.price_usd` → `{_product.price_usd!r}` ({type(_product.price_usd).__name__})\n"
                    f"- `product.in_stock` → `{_product.in_stock!r}` ({type(_product.in_stock).__name__})\n"
                    f"- `product.tags` → `{_product.tags!r}` ({type(_product.tags).__name__})"
                ),
                kind="success",
            ),
            mo.md("_These are real Python types — you can do math on `price_usd` and iterate `tags` directly._"),
        ])
    except (ValidationError, ValueError) as e:
        _out = mo.callout(
            mo.md(f"❌ **Validation failed:** {e}\n\nThat's the point — bad data is caught here instead of downstream."),
            kind="danger",
        )
    except urllib.error.HTTPError as e:
        _out = mo.callout(
            mo.md(f"❌ **Ollama rejected the schema (HTTP {e.code})** — server likely too old for schema output."),
            kind="danger",
        )

    _out
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    ## 5 · Your turn — write a schema

    Below is a half-finished JSON Schema for extracting a **recipe**. Fill in
    the `# TODO` fields so the schema requires:

    - `title` — a string (done for you as an example)
    - `servings` — an **integer**
    - `ingredients` — an **array of strings**

    Complete the schema, then click ▶ to extract a real recipe into it. The
    solution is in the next cell.
    """)
    return


@app.cell
def _(mo):
    exercise_button = mo.ui.run_button(label="▶ Run my schema")
    exercise_button
    return (exercise_button,)


@app.cell
def _(call_ollama, chat_model, exercise_button, json, mo, urllib):
    mo.stop(not exercise_button.value, mo.md("_Fill in the TODOs below, then click ▶ Run my schema._"))

    # ----------------------------------------------------------------------
    # TODO: complete the two blank field definitions.
    #   - "servings" should be an integer     -> {"type": "integer"}
    #   - "ingredients" should be a list of strings
    #        -> {"type": "array", "items": {"type": "string"}}
    # Also add "servings" and "ingredients" to the "required" list.
    # ----------------------------------------------------------------------
    my_schema = {
        "type": "object",
        "properties": {
            "title": {"type": "string"},   # example — leave this one
            "servings": None,              # TODO: integer
            "ingredients": None,           # TODO: array of strings
        },
        "required": ["title"],             # TODO: add "servings", "ingredients"
    }

    recipe_text = (
        "Classic guacamole serves 4. You'll need avocados, lime juice, salt, "
        "onion, and cilantro."
    )

    # Drop any TODO fields still set to None so the schema is at least valid.
    # We filter "required" the same way: naming a required field that has no
    # definition in "properties" is an invalid schema, and Ollama would reject
    # it with an error that looks like a version problem rather than a typo.
    _clean_props = {k: v for k, v in my_schema["properties"].items() if v is not None}
    _clean_required = [k for k in my_schema["required"] if k in _clean_props]
    _clean_schema = {"type": "object", "properties": _clean_props, "required": _clean_required}

    try:
        _raw = call_ollama(
            f"Extract the recipe details from this text:\n\n{recipe_text}",
            model=chat_model.value,
            response_format=_clean_schema,
        )
        _parsed = json.loads(_raw)
        _has_all = all(k in _parsed for k in ("title", "servings", "ingredients"))
        _out = mo.vstack([
            mo.md("**Extracted with your schema:**"),
            mo.callout(mo.md(f"```json\n{json.dumps(_parsed, indent=2)}\n```"), kind="info"),
            mo.callout(
                mo.md(
                    "✅ All three fields came back — schema complete!"
                    if _has_all else
                    "⚠️ Still missing `servings` or `ingredients` — the TODOs "
                    "above are likely still `None`. Fill them in and re-run."
                ),
                kind="success" if _has_all else "warn",
            ),
        ])
    except urllib.error.HTTPError as e:
        _out = mo.callout(mo.md(f"❌ Ollama rejected the schema (HTTP {e.code}) — server likely too old."), kind="danger")

    _out
    return


@app.cell(hide_code=True)
def _(mo):
    mo.accordion({
        "▶ Show a worked solution": mo.md(r"""
        ```python
        my_schema = {
            "type": "object",
            "properties": {
                "title": {"type": "string"},
                "servings": {"type": "integer"},
                "ingredients": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["title", "servings", "ingredients"],
        }
        ```

        `array` needs an `items` entry describing what each element looks like
        — here, `{"type": "string"}`. Adding all three keys to `required`
        forces the model to return every one, so `servings` comes back as a
        real integer and `ingredients` as a list you can loop over.
        """)
    })
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    ### Recap

    A ladder from "hope it works" to "typed and validated," one `format`
    field doing the work:

    | approach | you get | when to use |
    |---|---|---|
    | naive prompt | maybe-valid text | never, on its own — it's the baseline |
    | `format:"json"` | valid JSON, any shape | quick JSON when the shape can vary |
    | JSON Schema | valid JSON, *your* shape | reliable extraction into known fields |
    | Pydantic | typed, validated object | production code that consumes the data |

    Structured output is also what makes **tool use** (in the main
    `local_llm_demo.py`) robust: when the model must emit a tool call your
    code will parse, a schema is how you stop it from improvising the format.
    """)
    return


if __name__ == "__main__":
    app.run()
