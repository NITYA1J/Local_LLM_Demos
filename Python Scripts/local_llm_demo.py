"""
A local LLM, powered entirely by Ollama — plain Python version.
===============================================================

This is the same material as the `local_llm_demo.py` marimo notebook,
rewritten as an ordinary script for people who'd rather read and edit plain
Python than click through a notebook.

No API key, nothing leaves this machine. Four things, in order:

    Step 1  Ollama basics ............ send a prompt, get a plain-text reply.
    Step 2  System vs. user prompt ... same question, different instructions
                                       for HOW to answer.
    Step 3  Where local models break . small models fail in specific,
                                       learnable ways. Worth seeing on purpose.
    Step 4  Basic tool use ........... fix one of those failures (arithmetic)
                                       by letting the model call real Python.

HOW TO RUN
----------
    1. Make sure Ollama is running in another terminal:   ollama serve
    2. Make sure you have the model:                      ollama pull llama3.2:3b
    3. Run this file:                                     python local_llm_demo.py

Dependencies: none. Standard library only.

Heads up: this script makes roughly a dozen calls to the model, so on a small
machine expect it to take a minute or two. Comment out steps in main() if
you'd rather focus on one.
"""

import json
import os
import re
import urllib.error
import urllib.request


# ---------------------------------------------------------------------------
# Configuration — change these freely
# ---------------------------------------------------------------------------

# Where Ollama is listening. The default is the standard local address; the
# environment variable lets you point somewhere else without editing code.
OLLAMA_URL = os.environ.get("OLLAMA_BASE_URL", "http://127.0.0.1:11434")

# Which model to use. Must already be pulled (`ollama pull llama3.2:3b`).
# This is a 3-BILLION-parameter model — small enough to be fast and private,
# but far smaller than the ~100B+ models behind most cloud chat products.
# That size gap is the subject of Step 3.
MODEL = "llama3.2:3b"


# ---------------------------------------------------------------------------
# The one function that talks to the model
# ---------------------------------------------------------------------------

def call_ollama(prompt, model=MODEL, system=None, options=None):
    """Send a prompt to Ollama and return the model's reply as a string.

    This is the entire integration — one HTTP POST, no SDK, no framework.

    Two kinds of text go into the request, and they play different roles:

        prompt  (user prompt)   - the actual question or task.
        system  (system prompt) - instructions about HOW to behave while
                                  answering: tone, role, format, constraints.
                                  Optional; leave it out and the model falls
                                  back to its default behavior.

    Step 2 makes that second one concrete.
    """
    # Build the JSON body Ollama expects. stream=False means we get one
    # complete response back instead of a trickle of partial tokens.
    payload = {"model": model, "prompt": prompt, "stream": False}

    if system:
        payload["system"] = system
    if options:
        payload["options"] = options

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
# STEP 1 — Ollama basics: send a prompt, get an answer
# ---------------------------------------------------------------------------

def step_1_basics():
    """The simplest possible use: one question in, one answer out."""
    print_header("STEP 1 — Ollama basics")

    prompt = "What's the best way to learn how to use local LLMs?"
    print(f"Prompt: {prompt}\n")

    reply = call_ollama(prompt)

    print("Reply:")
    print(f"  {reply.strip()}")


# ---------------------------------------------------------------------------
# STEP 2 — System prompt vs. user prompt
# ---------------------------------------------------------------------------

# Each of these is an instruction about HOW to answer, not WHAT to answer.
# The user's question stays identical across all of them.
SYSTEM_PRESETS = {
    "(none - model default)": None,
    "Helpful assistant": (
        "You are a helpful, friendly assistant. Answer clearly and concisely."
    ),
    "Terse domain expert": (
        "You are a terse subject-matter expert. Answer in bullet points, no "
        "preamble, no pleasantries, no hedging."
    ),
    "Explain like I'm 5": (
        "You are explaining things to a curious 5-year-old. Use simple words, "
        "short sentences, and a concrete everyday example."
    ),
    "Skeptical scientist": (
        "You are a skeptical scientist. For every claim, note the evidence "
        "quality and what could make it wrong. Avoid overconfidence."
    ),
    "Pirate": (
        "You are a pirate captain. Answer entirely in pirate dialect, but keep "
        "the actual information accurate."
    ),
}


def step_2_system_prompts():
    """Ask ONE fixed question under several different system prompts.

    This is the cheapest, fastest lever for shaping model behavior — no
    retraining, no RAG, just a different instruction string prepended to every
    request. Watch the tone, structure, and even the CONTENT of the answer
    shift while the question itself never changes.
    """
    print_header("STEP 2 — System prompt vs. user prompt")

    # The user prompt is held completely fixed. Only `system` varies.
    question = "Should I update my existing course notes or start from scratch?"
    print(f"User prompt (identical every time): {question}")

    for preset_name, system_prompt in SYSTEM_PRESETS.items():
        print(f"\n--- system prompt: {preset_name} ---")
        reply = call_ollama(question, system=system_prompt)
        print(f"  {reply.strip()}")

    print("\nSame model, same question — only the system prompt changed.")


# ---------------------------------------------------------------------------
# STEP 3 — Where local models break down
# ---------------------------------------------------------------------------

# Each entry pairs a prompt that tends to fail with an explanation of WHY it
# fails. The "why" matters more than the failure itself.
BREAKDOWN_EXAMPLES = [
    {
        "name": "Letter counting (tokenization blindness)",
        "prompt": "How many times does the letter 'r' appear in the word 'tangential'?",
        "why": (
            "LLMs don't see individual letters. Text is split into sub-word "
            "TOKENS before the model ever sees it, so 'tangential' might be "
            "2-3 opaque chunks rather than 10 characters. Character-level "
            "counting requires reasoning the model was never directly trained "
            "to do well, and it gets worse as model size shrinks."
        ),
    },
    {
        "name": "Multi-digit arithmetic",
        "prompt": "What is 84,637 multiplied by 92,481? Give the exact number.",
        "why": (
            "Small models pattern-match arithmetic from training examples "
            "rather than actually running an algorithm — there's no built-in "
            "calculator. Confident-sounding wrong digits are common, and get "
            "more likely as the numbers grow. Step 4 fixes exactly this."
        ),
    },
    {
        "name": "Knowledge cutoff (recent events)",
        "prompt": "Who won the most recent World Cup, and what was the final score?",
        "why": (
            "The model only knows what was in its training data, which has a "
            "cutoff date. Ask about anything after that date and it will "
            "either say it doesn't know or — more concerning — confidently "
            "guess based on old patterns."
        ),
    },
    {
        "name": "Multi-constraint instructions",
        "prompt": (
            "Write exactly three sentences about coffee. The first sentence "
            "must start with the letter B. The second sentence must contain "
            "exactly seven words. Do not use the word 'bean' anywhere."
        ),
        "why": (
            "Small models tend to drop one constraint under load when asked to "
            "satisfy several at once. Watch WHICH one it fails, and how "
            "confidently it claims to have followed all of them anyway."
        ),
    },
    {
        "name": "Confident fabrication (hallucination)",
        "prompt": "Summarize the plot of the novel 'The Glass Meridian' by Aldous Whitfield.",
        "why": (
            "This book and author don't exist. A well-behaved model says so. A "
            "model prone to hallucination will invent a plausible-sounding "
            "plot summary instead of admitting it doesn't know — a good "
            "reminder that fluent text is not the same as true text."
        ),
    },
]


def step_3_failure_modes():
    """Run each known failure mode and print why it tends to break.

    These failures are usually wrong in a FLUENT, CONFIDENT way rather than an
    obviously broken way. That's exactly what makes them worth demonstrating:
    the danger isn't gibberish, it's plausible-sounding nonsense.
    """
    print_header("STEP 3 — Where local models break down")
    print(f"Using {MODEL}, a 3-billion-parameter model. Read each answer")
    print("CAREFULLY — the failures are fluent, not obviously broken.")

    for example in BREAKDOWN_EXAMPLES:
        print(f"\n--- {example['name']} ---")
        print(f"  Prompt: {example['prompt']}")
        reply = call_ollama(example["prompt"])
        print(f"  Reply:  {reply.strip()}")
        print(f"  WHY IT BREAKS: {example['why']}")


# ---------------------------------------------------------------------------
# STEP 4 — Basic tool use: giving the model a calculator
# ---------------------------------------------------------------------------

# We tell the model it may ASK for a calculation rather than attempting one.
# This is the same basic idea behind "function calling" / "tool use" in bigger
# frameworks — just written out by hand so every step is visible.
TOOL_SYSTEM_PROMPT = (
    "You have access to a calculator tool for arithmetic. If answering the "
    "question requires a calculation, respond with ONLY a single line in the "
    "exact form 'CALC: <expression>' using plain + - * / ** and parentheses "
    "(e.g. 'CALC: 84637 * 92481') - no commas in numbers, no explanation, no "
    "extra text. If the question does NOT require a calculation, just answer "
    "it directly and normally."
)


def safe_calculate(expression):
    """Evaluate a plain arithmetic expression like '84637 * 92481'.

    We can't just hand model-generated text to Python's real eval() — that
    would let it run arbitrary code, not just math. So this does two things
    first: it checks the string contains ONLY digits, decimal points,
    whitespace, and the symbols + - * / ** ( ), and only then calls eval()
    with its built-in functions disabled. Between the character whitelist and
    the disabled builtins, there's nothing left for it to do except arithmetic.

    Note on commas: the system prompt asks for numbers without commas, but the
    model doesn't always comply — especially when the question itself was
    written with them (e.g. "84,637"). Rather than fail on that, we strip
    commas before validating. A comma between digits is unambiguous here and
    never means anything else.
    """
    expression = expression.replace(",", "")

    only_math_characters = r"[0-9+\-*/(). ]+"
    if not re.fullmatch(only_math_characters, expression):
        raise ValueError(f"Expression contains disallowed characters: {expression!r}")

    # Builtins disabled: no open(), no __import__, nothing but arithmetic.
    return eval(expression, {"__builtins__": {}}, {})


def ask_with_calculator(question):
    """Run the full ask -> check -> calculate -> ask-again tool-use loop.

    Three plain steps, printed as they happen so you can watch the hand-off
    instead of reading it out of a black box.
    """
    # --- Step A: ask ------------------------------------------------------
    # The system prompt tells the model it may reply with "CALC: <expression>"
    # instead of guessing at the arithmetic itself.
    first_reply = call_ollama(question, system=TOOL_SYSTEM_PROMPT)
    print(f"  [1] Model's first reply: {first_reply.strip()}")

    # --- Step B: check ----------------------------------------------------
    # Did it actually request a calculation? If not, its first reply is final.
    if not first_reply.strip().upper().startswith("CALC:"):
        print("  [2] No calculation requested — first reply is the final answer.")
        return first_reply

    # Pull the expression out of the "CALC: ..." line.
    expression = first_reply.strip().split(":", 1)[1].strip()

    try:
        exact_result = safe_calculate(expression)
    except (ValueError, SyntaxError, ZeroDivisionError) as error:
        print(f"  [2] Tool call FAILED on {expression!r}: {error}")
        return f"(Could not safely evaluate {expression!r}: {error})"

    print(f"  [2] Tool called: {expression}  ->  exact result: {exact_result}")

    # --- Step C: ask again, with the real number --------------------------
    # This is the hand-off: give the model back the exact number so its final
    # answer uses real math instead of recomputing it (badly) itself.
    follow_up = (
        f"Question: {question}\n"
        f"You requested the calculation `{expression}`, and the exact result "
        f"is {exact_result}. Give the final answer to the question using this "
        f"exact result - do not recompute it yourself."
    )
    final_answer = call_ollama(follow_up)
    print("  [3] Sent the exact result back for a final answer.")
    return final_answer


def step_4_tool_use():
    """Compare the same arithmetic question with the calculator on vs. off.

    Step 3 showed the model guessing at multi-digit arithmetic and getting it
    wrong with total confidence. Tool use fixes that whole class of problem:
    instead of asking the model to COMPUTE an answer, we teach it to ASK FOR a
    calculation, then run real Python to get the exact number.
    """
    print_header("STEP 4 — Basic tool use (giving the model a calculator)")

    question = "What is 84,637 multiplied by 92,481?"
    print(f"Question: {question}")

    # For reference, what the answer actually is.
    correct_answer = 84637 * 92481
    print(f"(The true answer, computed by Python: {correct_answer})")

    # --- Without the tool: the Step 3 failure, reproduced ------------------
    print("\n--- WITHOUT the calculator tool ---")
    guess = call_ollama(question)
    print(f"  {guess.strip()}")
    if str(correct_answer) in guess:
        print("  -> It happened to get it right this time. Run again; it won't last.")
    else:
        print(f"  -> WRONG. The correct answer is {correct_answer}.")

    # --- With the tool: ask, calculate, ask again --------------------------
    print("\n--- WITH the calculator tool ---")
    answer = ask_with_calculator(question)
    print(f"\n  Final answer: {answer.strip()}")
    if str(correct_answer) in answer:
        print("  -> CORRECT. Real Python did the math, not the model.")
    else:
        print("  -> The number didn't survive into the final answer; small")
        print("     models sometimes reword it. The tool result was still exact.")

    # A question needing no arithmetic, to show the model skips the tool.
    print("\n--- A question that needs NO calculation ---")
    no_math = ask_with_calculator("What is the capital of France?")
    print(f"\n  Final answer: {no_math.strip()}")


# ---------------------------------------------------------------------------
# Main — runs each step in order
# ---------------------------------------------------------------------------

def main():
    print("A local LLM, powered entirely by Ollama")
    print(f"Ollama URL: {OLLAMA_URL}")
    print(f"Model:      {MODEL}")

    # Fail early with a helpful message rather than a confusing stack trace.
    if not ollama_is_running():
        print("\n[ERROR] Can't reach Ollama at", OLLAMA_URL)
        print("Start it in another terminal with:  ollama serve")
        print(f"And make sure the model is pulled:  ollama pull {MODEL}")
        return

    # Comment out any of these to focus on one step.
    step_1_basics()
    step_2_system_prompts()
    step_3_failure_modes()
    step_4_tool_use()

    print_header("RECAP")
    print("1. User prompt ............. the question itself.")
    print("2. System prompt ........... instructions for HOW to answer. Free to")
    print("                             change, no retraining, and it shifts")
    print("                             tone, format, even willingness to hedge.")
    print("3. Capability ceiling ...... no amount of prompt engineering fixes")
    print("                             tokenization blindness or a knowledge")
    print("                             cutoff. Some failures need a bigger")
    print("                             model; some need external tools.")
    print("4. Tool use ................ don't ask the model to compute, ask it")
    print("                             to DELEGATE to code that actually")
    print("                             computes. The same pattern extends to a")
    print("                             search tool, a database tool, or a")
    print("                             document retrieval tool — which is what")
    print("                             RAG is, structurally.")


# This guard means the steps only run when you execute the file directly
# (python local_llm_demo.py), not when you import it from elsewhere.
if __name__ == "__main__":
    main()
