"""
Intro: prompting a local LLM and turning the knobs — plain Python version.
==========================================================================

This is the same material as the `intro_prompting.py` marimo notebook,
rewritten as an ordinary script for people who'd rather read and edit plain
Python than click through a notebook.

Two ideas, and that's the whole file:

    1. Send a prompt, get an answer — the smallest possible call to a model
       running locally on this machine.
    2. Set the model's PARAMETERS from Python — the dials that control how
       random, how long, and how repeatable the output is.

The steps:

    Step 1  The smallest possible call
    Step 2  temperature ......... the randomness dial
    Step 3  top_p / top_k ....... which words are even considered
    Step 4  seed ................ makes randomness repeatable
    Step 5  num_predict / stop .. controlling length
    Exercise  build your own options dict

HOW TO RUN
----------
    1. Make sure Ollama is running in another terminal:   ollama serve
    2. Make sure you have the model:                      ollama pull llama3.2:3b
    3. Run this file:                                     python intro_prompting.py

Dependencies: none. Standard library only.

Heads up: this script makes roughly a dozen calls to the model, so on a small
machine expect it to take a minute or two. Comment out steps in main() if
you'd rather focus on one.
"""

import json
import os
import urllib.error
import urllib.request


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

def call_ollama(prompt, model=MODEL, system=None, options=None):
    """Send a prompt to Ollama and return the model's reply as a string.

    This is the entire integration — one HTTP POST, no SDK, no framework.

    Arguments:
        prompt   The question or task (the "user prompt").
        model    Which model to use.
        system   Optional instructions about HOW to answer (tone, role,
                 format). Covered properly in the main local_llm_demo script.
        options  The star of this script: a plain dict of generation
                 parameters, e.g. {"temperature": 0.8, "seed": 42}. Leave it
                 out and the model uses its own defaults.
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
# STEP 1 — The smallest possible call
# ---------------------------------------------------------------------------

def step_1_simplest_call():
    """One prompt in, one string out. No options at all — model defaults."""
    print_header("STEP 1 — The smallest possible call")

    prompt = "In one sentence, what is a large language model?"
    print(f"Prompt: {prompt}\n")

    # No `options` argument at all, so Ollama uses its built-in defaults.
    reply = call_ollama(prompt)

    print("Reply:")
    print(f"  {reply.strip()}")
    print("\nThat's the whole integration. Everything below just adds an")
    print("`options` dictionary to this same call.")


# ---------------------------------------------------------------------------
# STEP 2 — temperature: the randomness dial
# ---------------------------------------------------------------------------

def step_2_temperature():
    """Show the same prompt at low vs. high temperature, twice each.

    Temperature controls how much the model is allowed to gamble when picking
    each next word:

        Low  (0.0-0.3) - nearly deterministic. Picks the most likely word
                         almost every time. Good for facts and code.
        High (0.8-1.5) - adventurous. Willing to pick less likely words,
                         which reads as more creative, and more prone to
                         going off the rails.

    We run each setting TWICE, because the effect is easiest to see by
    comparing two runs: at low temperature they'll look nearly identical,
    at high temperature they'll drift apart.
    """
    print_header("STEP 2 — temperature (the randomness dial)")

    prompt = "Write a one-sentence bedtime story about a sleepy robot."
    print(f"Prompt (held constant): {prompt}")

    for temperature in (0.0, 1.2):
        print(f"\n--- temperature = {temperature} ---")

        # Building this dict is the entire point of the step.
        options = {"temperature": temperature}

        first_run = call_ollama(prompt, options=options)
        second_run = call_ollama(prompt, options=options)

        print(f"  Run 1: {first_run.strip()}")
        print(f"  Run 2: {second_run.strip()}")

        # A crude but honest similarity check, just to make the point visible.
        if first_run.strip() == second_run.strip():
            print("  -> The two runs are IDENTICAL.")
        else:
            print("  -> The two runs DIFFER.")

    print("\nLow temperature tends to repeat itself; high temperature wanders.")
    print("That difference is the randomness you just dialed in.")


# ---------------------------------------------------------------------------
# STEP 3 — top_p and top_k: narrowing the candidate pool
# ---------------------------------------------------------------------------

def step_3_top_p_and_top_k():
    """Compare a very narrow candidate pool against a wide one.

    Temperature decides HOW MUCH to gamble; top_p and top_k decide WHICH
    WORDS ARE EVEN ON THE TABLE before that gamble happens. At each step the
    model has a ranked list of possible next words, and these two trim it:

        top_k - keep only the k most likely words. top_k=1 means "always take
                the single best candidate", which makes output very safe and
                repetitive. Lower = safer.
        top_p - keep the smallest set of top words whose probabilities add up
                to p. top_p=0.9 means "the most likely words covering 90% of
                the probability". Also called nucleus sampling. Lower = safer.
    """
    print_header("STEP 3 — top_p and top_k (narrowing the candidate pool)")

    prompt = "Suggest a creative name for a campus coffee shop."
    print(f"Prompt (held constant): {prompt}")

    # Two deliberately extreme settings so the contrast is obvious.
    settings = [
        ("very narrow", {"top_k": 1, "top_p": 0.1, "temperature": 1.0}),
        ("wide open", {"top_k": 100, "top_p": 1.0, "temperature": 1.0}),
    ]

    for label, options in settings:
        print(f"\n--- {label}: {options} ---")
        reply = call_ollama(prompt, options=options)
        print(f"  {reply.strip()}")

    print("\nWith top_k=1 the model must take its single most likely word every")
    print("time, so output gets repetitive and 'safe'. Wider settings let more")
    print("variety through.")


# ---------------------------------------------------------------------------
# STEP 4 — seed: making randomness repeatable
# ---------------------------------------------------------------------------

def step_4_seed():
    """Show that a fixed seed makes even high-temperature output repeatable.

    Temperature makes output random — but sometimes you want REPEATABLE
    randomness, so a demo or a test gives the same result every time. The
    seed fixes the starting point of the random number generator:

        same seed + same parameters + same prompt  ->  the same output

    We use a high temperature here on purpose: without a seed these replies
    would almost certainly differ, so if they come back identical, that's the
    seed doing the work.
    """
    print_header("STEP 4 — seed (making randomness repeatable)")

    prompt = "Invent a name and one-line backstory for a friendly dragon."
    print(f"Prompt (held constant): {prompt}")

    # High temperature so the seed's effect is unmistakable.
    options = {"temperature": 1.0, "seed": 42}
    print(f"\n--- same seed, twice: {options} ---")

    first_run = call_ollama(prompt, options=options)
    second_run = call_ollama(prompt, options=options)
    print(f"  Run 1: {first_run.strip()}")
    print(f"  Run 2: {second_run.strip()}")

    if first_run.strip() == second_run.strip():
        print("  -> IDENTICAL. The seed made the randomness repeatable.")
    else:
        print("  -> Not identical. Some model/runtime settings can still")
        print("     introduce small differences, but this is unusual.")

    # Now change ONLY the seed and show we get a different (but equally
    # repeatable) result.
    different_options = {"temperature": 1.0, "seed": 12345}
    print(f"\n--- different seed: {different_options} ---")
    third_run = call_ollama(prompt, options=different_options)
    print(f"  Run 3: {third_run.strip()}")
    print("  -> A different answer, because the seed changed. Run the script")
    print("     again and Run 3 will reproduce exactly.")


# ---------------------------------------------------------------------------
# STEP 5 — num_predict and stop: controlling length
# ---------------------------------------------------------------------------

def step_5_length_controls():
    """Two dials that decide when the model stops talking.

        num_predict - a hard cap on how many tokens (roughly, word pieces)
                      the model may generate. Small values force short
                      answers, and it's the main knob for keeping a slow
                      local model responsive.
        stop        - a list of strings that, if generated, cut the output
                      off immediately. Useful for structured output: stop at
                      a newline to get one line, or at a marker like "END".

    Important: num_predict does NOT ask the model to be brief. It just stops
    generation when the cap is hit — which is why the output truncates
    mid-sentence.
    """
    print_header("STEP 5 — num_predict and stop (controlling length)")

    prompt = "List three tips for studying effectively."
    print(f"Prompt (held constant): {prompt}")

    # A very small cap, to make the truncation obvious.
    print("\n--- num_predict = 16 (very short cap) ---")
    short_reply = call_ollama(prompt, options={"num_predict": 16})
    print(f"  {short_reply.strip()}")
    print("  -> Notice it likely cuts off mid-thought. The model wasn't asked")
    print("     to be brief; it was simply stopped.")

    # A roomier cap for comparison.
    print("\n--- num_predict = 200 (room to finish) ---")
    long_reply = call_ollama(prompt, options={"num_predict": 200})
    print(f"  {long_reply.strip()}")

    # A stop sequence: cut off at the first newline to force a single line.
    print("\n--- stop = ['\\n'] (stop at the first line break) ---")
    one_line = call_ollama(prompt, options={"num_predict": 200, "stop": ["\n"]})
    print(f"  {one_line.strip()}")
    print("  -> Generation halted the moment a newline appeared.")


# ---------------------------------------------------------------------------
# EXERCISE — your turn (optional)
# ---------------------------------------------------------------------------

def exercise_build_your_own_options():
    """Fill in the TODOs below, then run the script again to test your dials.

    Goal: set the three values so the answer is
        - highly creative / random  (think temperature)
        - repeatable across runs    (think seed)
        - capped to a short length  (think num_predict)

    The solution is in a comment block at the very bottom of this file.
    """
    print_header("EXERCISE — build your own options dict")

    # -----------------------------------------------------------------------
    # TODO: replace each None with a number.
    #   temperature - high for creativity        (try ~1.2)
    #   seed        - any integer, so runs repeat (try 7)
    #   num_predict - small, to keep it short     (try 40)
    # -----------------------------------------------------------------------
    my_options = {
        "temperature": None,   # TODO
        "seed": None,          # TODO
        "num_predict": None,   # TODO
    }

    prompt = "Give me a fun team name for a robotics club."

    # Drop any blanks still set to None so the call works while you're
    # partway through the exercise.
    filled_in = {key: value for key, value in my_options.items() if value is not None}

    if not filled_in:
        print("The TODOs above aren't filled in yet, so we're calling the model")
        print("with no options at all (pure defaults).")
        print("(Edit my_options in this function and re-run.)\n")

    print(f"Options being used: {filled_in}")

    # Call twice, so you can confirm the seed makes the answer repeat.
    first_run = call_ollama(prompt, options=filled_in)
    second_run = call_ollama(prompt, options=filled_in)
    print(f"\n  Run 1: {first_run.strip()}")
    print(f"  Run 2: {second_run.strip()}")

    # Check the learner's work and give specific feedback.
    goals_met = []
    if filled_in.get("temperature", 0) and filled_in["temperature"] >= 0.8:
        goals_met.append("creative (high temperature)")
    if "seed" in filled_in and first_run.strip() == second_run.strip():
        goals_met.append("repeatable (seed fixed, both runs matched)")
    if filled_in.get("num_predict") and filled_in["num_predict"] <= 60:
        goals_met.append("short (low num_predict)")

    if len(goals_met) == 3:
        print("\n[OK] All three goals met: " + "; ".join(goals_met))
    else:
        print(f"\n[TODO] Goals met so far: {goals_met or 'none'}")
        print("       See the solution at the bottom of this file if stuck.")


# ---------------------------------------------------------------------------
# Main — runs each step in order
# ---------------------------------------------------------------------------

def main():
    print("Intro: prompting a local LLM and turning the knobs")
    print(f"Ollama URL: {OLLAMA_URL}")
    print(f"Model:      {MODEL}")

    # Fail early with a helpful message rather than a confusing stack trace.
    if not ollama_is_running():
        print("\n[ERROR] Can't reach Ollama at", OLLAMA_URL)
        print("Start it in another terminal with:  ollama serve")
        print(f"And make sure the model is pulled:  ollama pull {MODEL}")
        return

    # Comment out any of these to focus on one step.
    step_1_simplest_call()
    step_2_temperature()
    step_3_top_p_and_top_k()
    step_4_seed()
    step_5_length_controls()
    exercise_build_your_own_options()

    print_header("RECAP")
    print("temperature ....... how random / creative the output is")
    print("top_p, top_k ...... which candidate words are even considered")
    print("seed .............. makes random output repeatable")
    print("num_predict ....... hard cap on output length")
    print("stop .............. strings that cut generation off early")
    print("\nEvery one of these is just a key in the `options` dict passed to")
    print("call_ollama(). Next: local_llm_demo.py, which covers system")
    print("prompts, model failure modes, and tool use.")


# This guard means the steps only run when you execute the file directly
# (python intro_prompting.py), not when you import it from elsewhere.
if __name__ == "__main__":
    main()


# ---------------------------------------------------------------------------
# EXERCISE SOLUTION (don't peek until you've tried it!)
# ---------------------------------------------------------------------------
#
#   my_options = {
#       "temperature": 1.2,   # high -> creative / random
#       "seed": 7,            # fixed -> repeatable across runs
#       "num_predict": 40,    # small -> capped length
#   }
#
# With these, both runs return the SAME short, creative answer: high
# temperature supplies the creativity, the seed makes that particular
# creative result repeatable, and num_predict keeps it brief. Change the seed
# and you get a different — but again repeatable — answer.
# ---------------------------------------------------------------------------
