"""
Structured output from a local LLM — plain Python version.
===========================================================

This is the same material as `structured_output.py` (the marimo notebook),
rewritten as an ordinary script for people who'd rather read and edit plain
Python than click through a notebook.

The question it answers: how do you get a model to return clean,
machine-readable data — JSON with exactly the fields you need — instead of a
paragraph of prose you have to parse by hand?

It walks through four steps, each fixing the previous one's weakness:

    Step 1  Naive prompt ......... just ASK for JSON. Watch it break.
    Step 2  format="json" ........ Ollama guarantees VALID JSON.
    Step 3  JSON Schema .......... guarantees valid JSON in YOUR shape.
    Step 4  Pydantic ............. parses it into a TYPED, VALIDATED object.

HOW TO RUN
----------
    1. Make sure Ollama is running in another terminal:   ollama serve
    2. Make sure you have the model:                      ollama pull llama3.2:3b
    3. Run this file:                                     python structured_output.py

Dependencies: none for Steps 1-3 (standard library only). Step 4 needs
pydantic (`pip install pydantic`); if it's missing, the script skips Step 4
and tells you, rather than crashing.

Things to try: change MODEL below, edit the prompts, or add a field to
PERSON_SCHEMA and re-run to see the output change shape.
"""

import json
import os
import urllib.error
import urllib.request

# Pydantic is only needed for Step 4. We import it defensively so that
# Steps 1-3 still work for anyone who hasn't installed it.
try:
    from pydantic import BaseModel, ValidationError
    PYDANTIC_AVAILABLE = True
except ImportError:
    PYDANTIC_AVAILABLE = False


# ---------------------------------------------------------------------------
# Configuration — change these freely
# ---------------------------------------------------------------------------

# Where Ollama is listening. The default is the standard local address; the
# environment variable lets you point somewhere else without editing code.
OLLAMA_URL = os.environ.get("OLLAMA_BASE_URL", "http://127.0.0.1:11434")

# Which model to use. Must already be pulled (`ollama pull llama3.2:3b`).
MODEL = "llama3.2:3b"


# ---------------------------------------------------------------------------
# The one function that talks to the model
# ---------------------------------------------------------------------------

def call_ollama(prompt, model=MODEL, system=None, options=None, response_format=None):
    """Send a prompt to Ollama and return the model's reply as a string.

    This is the entire integration — one HTTP POST, no SDK, no framework.

    Arguments:
        prompt          The question or task (the "user prompt").
        model           Which model to use.
        system          Optional instructions about HOW to answer.
        options         Optional dict of generation parameters,
                        e.g. {"temperature": 0.2}.
        response_format This is the star of this script. It becomes the
                        request's `format` field, and can be:
                          None   -> ordinary free-text reply
                          "json" -> reply is guaranteed to be valid JSON
                          dict   -> a JSON Schema the reply must conform to
    """
    # Build the JSON body Ollama expects. stream=False means we get one
    # complete response back instead of a trickle of partial tokens.
    payload = {"model": model, "prompt": prompt, "stream": False}

    # Only include the optional fields if they were actually provided, so we
    # send the smallest, clearest request possible.
    if system:
        payload["system"] = system
    if options:
        payload["options"] = options
    if response_format is not None:
        payload["format"] = response_format

    # Encode the body as JSON bytes and POST it to the /api/generate endpoint.
    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        f"{OLLAMA_URL}/api/generate",
        data=data,
        headers={"Content-Type": "application/json"},
    )

    # Send it. The reply is a JSON object; the text we want is under "response".
    with urllib.request.urlopen(request, timeout=120) as response:
        body = json.loads(response.read().decode("utf-8"))
    return body["response"]


def ollama_is_running():
    """Return True if we can reach Ollama, so we can fail with a clear message."""
    try:
        urllib.request.urlopen(f"{OLLAMA_URL}/api/tags", timeout=5)
        return True
    except Exception:
        return False


def print_header(title):
    """Just formatting — prints a visually obvious section divider."""
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


# ---------------------------------------------------------------------------
# STEP 1 — The naive approach: just ask for JSON in the prompt
# ---------------------------------------------------------------------------

def step_1_naive_prompt():
    """Ask for JSON in plain English, with NO format field, and see what happens.

    This is what most people try first. It sometimes works — but the model is
    free to wrap its JSON in a friendly sentence ("Sure! Here's the JSON:"),
    fence it in ```json markdown, or add a trailing comma. Any of those break
    a strict parser like json.loads().
    """
    print_header("STEP 1 — Naive prompt (no format field)")

    prompt = (
        "Give me three programming languages, each with a one-word "
        "description. Respond in JSON."
    )

    # Note: we do NOT pass response_format here. This is a plain text request.
    raw_reply = call_ollama(prompt)

    print("Raw reply from the model:")
    print("-" * 70)
    print(raw_reply)
    print("-" * 70)

    # Now try to parse it as JSON. This is the moment of truth.
    try:
        parsed = json.loads(raw_reply)
        print("\n[OK] json.loads() succeeded this time:")
        print(json.dumps(parsed, indent=2))
        print("\n...but this is luck, not a guarantee. Run it again a few times.")
    except json.JSONDecodeError as error:
        print(f"\n[FAIL] json.loads() could not parse the reply: {error}")
        print("The model returned text, not clean JSON. Step 2 fixes this.")


# ---------------------------------------------------------------------------
# STEP 2 — format="json": guarantee the reply is VALID JSON
# ---------------------------------------------------------------------------

def step_2_json_mode():
    """Same prompt, but now Ollama guarantees the output is valid JSON.

    Passing format="json" makes Ollama constrain the model's decoding so it
    can only produce syntactically valid JSON. No prose, no markdown fences,
    no trailing commas — json.loads() will always succeed.

    The catch: "valid JSON" is not "the JSON I wanted". The model still picks
    its own keys and structure, so one run might give {"languages": [...]}
    and the next a bare list. That's what Step 3 fixes.
    """
    print_header('STEP 2 — format="json" (valid JSON guaranteed)')

    prompt = (
        "Give me three programming languages, each with a one-word "
        "description. Respond in JSON."
    )

    # The only change from Step 1 is this one argument.
    raw_reply = call_ollama(prompt, response_format="json")

    # This parse is safe — format="json" guarantees it.
    parsed = json.loads(raw_reply)

    print("Valid JSON, guaranteed to parse:")
    print(json.dumps(parsed, indent=2))
    print(
        "\nNotice which top-level keys the model chose. Re-run the script and "
        "they may well change — you get valid JSON, but not a reliable shape."
    )


# ---------------------------------------------------------------------------
# STEP 3 — JSON Schema: guarantee the SHAPE, not just the syntax
# ---------------------------------------------------------------------------

# A JSON Schema is just a dictionary describing the structure you want:
# what fields exist, what type each one is, and which are mandatory.
PERSON_SCHEMA = {
    "type": "object",
    "properties": {
        "name": {"type": "string"},
        "age": {"type": "integer"},
        "city": {"type": "string"},
        "occupation": {"type": "string"},
    },
    # Fields listed here MUST be present in the output.
    "required": ["name", "age", "city", "occupation"],
}


def step_3_json_schema():
    """Pass a JSON Schema so the output has exactly the fields we asked for.

    This is the real workhorse of structured output: pulling reliable,
    predictable records out of free-form text. Because the shape is fixed,
    downstream code can safely do data["age"] without defensive checks.

    Version note: schema-constrained output needs a reasonably recent Ollama
    (roughly late 2024 or newer). If yours is older it will reject the
    request, and we catch that below.
    """
    print_header("STEP 3 — JSON Schema (valid JSON, in YOUR shape)")

    text_to_extract_from = (
        "Maria is a 34-year-old software engineer who lives in Austin, Texas."
    )
    prompt = f"Extract the person's details from this text:\n\n{text_to_extract_from}"

    print("Schema being enforced:")
    print(json.dumps(PERSON_SCHEMA, indent=2))
    print(f"\nText to extract from:\n  {text_to_extract_from}")

    try:
        # Passing the schema dict (instead of the string "json") constrains
        # the output to match it exactly.
        raw_reply = call_ollama(prompt, response_format=PERSON_SCHEMA)
    except urllib.error.HTTPError as error:
        print(f"\n[FAIL] Ollama rejected the schema (HTTP {error.code}).")
        print("Your Ollama is probably too old for schema-constrained output.")
        print('Step 2 (format="json") still works — consider upgrading Ollama.')
        return

    parsed = json.loads(raw_reply)
    print("\nExtracted, conforming to the schema:")
    print(json.dumps(parsed, indent=2))

    # Confirm every required field actually came back.
    missing = [field for field in PERSON_SCHEMA["required"] if field not in parsed]
    if missing:
        print(f"\n[WARN] Missing required field(s): {missing}")
    else:
        print("\n[OK] Every required field is present — same keys, every run.")
        # Because the shape is guaranteed, this kind of access is now safe:
        print(f"     e.g. parsed['age'] = {parsed['age']}")


# ---------------------------------------------------------------------------
# STEP 4 — Pydantic: a typed, validated Python object
# ---------------------------------------------------------------------------

# Only define the class if pydantic imported successfully, otherwise this
# line itself would crash the script for people who don't have it installed.
if PYDANTIC_AVAILABLE:

    # The shape we want, written once as a normal typed Python class.
    # Pydantic can generate a JSON Schema FROM this class, so the class and the
    # schema can never drift out of sync — one source of truth.
    #
    # (Deliberately no docstring: pydantic copies docstrings into the generated
    # schema as a "description" field, which just adds noise when we print it.)
    class Product(BaseModel):
        name: str
        price_usd: float
        in_stock: bool
        tags: list[str]


def step_4_pydantic():
    """Use a Pydantic class to both constrain the output AND validate it.

    Step 3 gets the right shape on the wire, but you still end up with a plain
    dict. Pydantic goes one step further:

      1. Product.model_json_schema()  generates the schema to constrain output.
      2. Product.model_validate_json() parses the reply into a real object,
         raising an error if anything is malformed.

    The payoff is typed access: product.price_usd is a genuine float you can
    do arithmetic on, not data["price_usd"] of unknown type.
    """
    print_header("STEP 4 — Pydantic (typed and validated object)")

    if not PYDANTIC_AVAILABLE:
        print("[SKIP] pydantic is not installed, so this step can't run.")
        print("Install it with:  pip install pydantic")
        print("Steps 1-3 above don't need it.")
        return

    text_to_extract_from = (
        "The UltraGrip water bottle costs $24.99, is currently in stock, and "
        "is great for hiking, cycling, and gym use."
    )
    prompt = f"Extract the product details from this text:\n\n{text_to_extract_from}"

    # Generate the JSON Schema straight from the class definition.
    schema = Product.model_json_schema()
    print("Schema auto-generated from the Product class:")
    print(json.dumps(schema, indent=2))

    try:
        raw_reply = call_ollama(prompt, response_format=schema)
    except urllib.error.HTTPError as error:
        print(f"\n[FAIL] Ollama rejected the schema (HTTP {error.code}) — likely too old.")
        return

    try:
        # Parse AND validate the reply into a real Product object in one step.
        product = Product.model_validate_json(raw_reply)
    except ValidationError as error:
        # This is a feature, not a bug: bad data is caught right here instead
        # of silently flowing into the rest of your program.
        print(f"\n[FAIL] Validation caught malformed data: {error}")
        return

    print("\nValidated Product object — now real Python types, not text:")
    print(f"  product.name      = {product.name!r}        ({type(product.name).__name__})")
    print(f"  product.price_usd = {product.price_usd!r}   ({type(product.price_usd).__name__})")
    print(f"  product.in_stock  = {product.in_stock!r}    ({type(product.in_stock).__name__})")
    print(f"  product.tags      = {product.tags!r}        ({type(product.tags).__name__})")

    # Because these are real types, we can use them directly — no casting.
    print(f"\n  Arithmetic works: two bottles cost ${product.price_usd * 2:.2f}")
    print(f"  Iteration works:  {len(product.tags)} tags -> {', '.join(product.tags)}")


# ---------------------------------------------------------------------------
# EXERCISE — your turn (optional)
# ---------------------------------------------------------------------------

def exercise_write_your_own_schema():
    """Fill in the TODOs below, then run the script again to test your schema.

    Goal: complete the schema so it extracts a recipe with:
        title       - a string  (done for you as an example)
        servings    - an integer
        ingredients - a list of strings

    Hints:
        an integer         -> {"type": "integer"}
        a list of strings  -> {"type": "array", "items": {"type": "string"}}
    Remember to add the new field names to the "required" list too.
    """
    print_header("EXERCISE — write your own schema")

    # -----------------------------------------------------------------------
    # TODO: replace the two `None` values, and extend "required".
    # -----------------------------------------------------------------------
    my_schema = {
        "type": "object",
        "properties": {
            "title": {"type": "string"},   # example — leave this one alone
            "servings": None,              # TODO: an integer
            "ingredients": None,           # TODO: a list of strings
        },
        "required": ["title"],             # TODO: add "servings" and "ingredients"
    }

    recipe_text = (
        "Classic guacamole serves 4. You'll need avocados, lime juice, salt, "
        "onion, and cilantro."
    )

    # Drop any fields still set to None so the schema stays valid while you
    # work through the TODOs. We filter "required" the same way: naming a
    # required field that has no definition in "properties" is an invalid
    # schema, and Ollama would reject it with an error that looks like a
    # version problem rather than a half-finished exercise.
    filled_in = {
        field: definition
        for field, definition in my_schema["properties"].items()
        if definition is not None
    }
    working_schema = {
        "type": "object",
        "properties": filled_in,
        "required": [field for field in my_schema["required"] if field in filled_in],
    }

    if len(filled_in) < 3:
        print("The TODOs above aren't filled in yet — only extracting:",
              ", ".join(filled_in))
        print("(Edit my_schema in this function and re-run to complete it.)\n")

    try:
        raw_reply = call_ollama(
            f"Extract the recipe details from this text:\n\n{recipe_text}",
            response_format=working_schema,
        )
    except urllib.error.HTTPError as error:
        print(f"[FAIL] Ollama rejected the schema (HTTP {error.code}) — likely too old.")
        return

    parsed = json.loads(raw_reply)
    print("Extracted with your schema:")
    print(json.dumps(parsed, indent=2))

    if all(field in parsed for field in ("title", "servings", "ingredients")):
        print("\n[OK] All three fields came back — schema complete. Nice work.")
    else:
        print("\n[TODO] Still missing 'servings' or 'ingredients'. See the "
              "solution at the bottom of this file if you're stuck.")


# ---------------------------------------------------------------------------
# Main — runs each step in order
# ---------------------------------------------------------------------------

def main():
    print("Structured output from a local LLM")
    print(f"Ollama URL: {OLLAMA_URL}")
    print(f"Model:      {MODEL}")

    # Fail early with a helpful message rather than a confusing stack trace.
    if not ollama_is_running():
        print("\n[ERROR] Can't reach Ollama at", OLLAMA_URL)
        print("Start it in another terminal with:  ollama serve")
        print(f"And make sure the model is pulled:  ollama pull {MODEL}")
        return

    step_1_naive_prompt()
    step_2_json_mode()
    step_3_json_schema()
    step_4_pydantic()
    exercise_write_your_own_schema()

    print_header("RECAP")
    print("naive prompt   -> maybe-valid text  (never rely on this alone)")
    print('format="json"  -> valid JSON, any shape')
    print("JSON Schema    -> valid JSON, YOUR shape")
    print("Pydantic       -> typed, validated Python object")
    print("\nOne `format` field does all the work. The rest is deciding")
    print("what to put in it.")


# This guard means the steps only run when you execute the file directly
# (python structured_output.py), not when you import it from elsewhere.
if __name__ == "__main__":
    main()


# ---------------------------------------------------------------------------
# EXERCISE SOLUTION (don't peek until you've tried it!)
# ---------------------------------------------------------------------------
#
#   my_schema = {
#       "type": "object",
#       "properties": {
#           "title": {"type": "string"},
#           "servings": {"type": "integer"},
#           "ingredients": {"type": "array", "items": {"type": "string"}},
#       },
#       "required": ["title", "servings", "ingredients"],
#   }
#
# An "array" needs an "items" entry describing what each element looks like.
# Adding all three names to "required" forces the model to return every one,
# so servings comes back as a real integer and ingredients as a real list.
# ---------------------------------------------------------------------------
