# Local LLM

Teaching material for the **Applied AI Summer Workshop**: running a large
language model locally — on the workshop JupyterHub or your own machine — and
learning to control it.

No API key. No account. Nothing leaves the machine it runs on. Everything
here talks to [Ollama](https://ollama.com) over its plain HTTP API using only
the Python standard library — no SDK, no LangChain, no framework of any kind.

That constraint is deliberate. Every HTTP request in this repository is
written out in full, so you can read any file top to bottom and see exactly
what is being sent to the model and what comes back. There is no hidden layer
doing something clever on your behalf.

## Where this is meant to run

This material was built for the **NC State JupyterHub**, which already has
marimo configured — so on the Hub there is no notebook software to install.
You only need to set up Ollama and the models, which the included script
handles.

It runs fine on a personal machine too; that's just the secondary path. The
only difference is that you'll need to install marimo yourself if you want the
notebooks (`pip install -r requirements.txt`).

📘 **Setting this up on your own computer?** See
**[`Local_LLM_Guide.md`](Local_LLM_Guide.md)** — a step-by-step walkthrough of
installing Ollama, pulling a model, and running it on macOS, Windows, or
Linux, including what to do if you don't have administrator access.

**Don't want marimo at all?** The `Python Scripts/` folder has the same three
modules as ordinary Python scripts you can run with `python <file>.py`. No
notebook software, and for two of the three, nothing to install beyond the
standard library.

---

## Quick start

**0. Open a terminal and get to this folder.** On the JupyterHub, start your
server, then choose **File → New → Terminal**. `cd` into this folder (wherever
you cloned or copied it) so that `ls` shows `README.md` and `setup_ollama.sh`.

> **A note on the notebooks:** these are **marimo** notebooks, not Jupyter
> ones. They're `.py` files, so double-clicking one in the Jupyter file
> browser opens a plain text editor — that's expected. You launch them from
> the terminal with `marimo edit` (step 3 below). If you'd rather skip marimo
> entirely, the `Python Scripts/` folder has the same material as ordinary
> scripts.

**1. Install Ollama and pull the models.** On the Hub (Linux), run the
included setup script — it needs no `sudo` and is safe to re-run:

```bash
bash setup_ollama.sh
```

> On **macOS or Windows**, don't use this script — it's Linux-only. See
> [`Local_LLM_Guide.md`](Local_LLM_Guide.md) instead.

This installs Ollama and pulls the models for **both** workshop sessions: the
chat model (`llama3.2:3b`) used by the modules here, plus the embedding models
(`nomic-embed-text`, `mxbai-embed-large`) needed by the later *Local LLM and
RAG* session. Pulling them together means no second multi-gigabyte download
mid-workshop. The RAG folder ships an equivalent script, so running either one
is enough.

If you already have Ollama on your own machine and only care about this
folder, the chat model alone is sufficient:

```bash
ollama pull llama3.2:3b
```

> ⚠️ **Storage caveat on the Hub.** Models are downloaded to `/tmp`, because
> the Hub home-directory quota (~10 GB) is too small to hold them. `/tmp` is
> not guaranteed to survive a session or pod restart, so **if your session
> restarts you may need to re-run `setup_ollama.sh`**. It's safe to re-run and
> skips anything already done. If a persistent volume is available, point
> `OLLAMA_MODELS` at it instead.

**2. Check the server is running.** The setup script **already starts it** for
you, so normally there's nothing to do here. Confirm with:

```bash
ollama list
```

If that prints your models, you're set. Only if it reports a connection error
(say, after a session restart) do you need to start it yourself:

```bash
nohup ollama serve > ~/ollama/ollama.log 2>&1 & disown
```

**3. Pick your path** — the same material is available two ways:

| | Interactive notebooks | Plain Python scripts |
|---|---|---|
| **Where** | this folder | `Python Scripts/` |
| **Run with** | `marimo edit <file>.py` | `python <file>.py` |
| **Feels like** | click buttons, drag sliders, see results update | read the code, edit values, re-run |
| **Needs** | marimo — already installed on the NC State Hub | usually nothing — see below |

Both cover the same material. Use the notebooks if you want to *play* with the
model; use the scripts if you want to *read and edit* the code, or if you'd
rather not use marimo at all. Neither is a prerequisite for the other.

> **Yes, the filenames are identical in both places** — that's deliberate, so
> each module is easy to match up. The *folder* is the difference: notebooks
> sit in the top folder, scripts in `Python Scripts/`. If a command isn't
> doing what you expect, check which directory you're in.

Run them in order — `intro_prompting.py`, then `local_llm_demo.py`, then
`structured_output.py`.

### Fallback: Google Colab

If the Hub is unavailable, the `Colab/` folder has the whole workshop as two
self-contained Jupyter notebooks. Each installs Ollama and pulls its own
models, so they need nothing else from this folder — you can open either on
its own.

| Notebook | Covers |
| --- | --- |
| `prompting_local_llm_colab.ipynb` | Modules 1–3: parameters, system prompts, model limits, tool use, structured output |
| `rag_local_llm_colab.ipynb` | The RAG session: chunking, embeddings, retrieval, generation |
| `corpus.zip` | The document corpus, for the RAG notebook's manual-upload fallback |

Two things to know. Colab has no marimo, so sliders and buttons become
variables you edit and re-run. And **it is not private** — the model runs on a
Google VM, so prompts (and, in the RAG notebook, your documents) leave your
machine. That's the one claim from the modules that doesn't survive the move
to Colab, and both notebooks say so up front.

---

## The material

Three modules, in a sensible order. Each one is a single self-contained file.

### 1. `intro_prompting.py` — prompting and parameters

**Start here.** The gentlest possible introduction. Two ideas only:

- Send a prompt to a local model and get an answer back.
- Set the model's **generation parameters** from Python by passing an
  `options` dictionary.

The parameters covered, and what each one does:

| parameter | controls |
|---|---|
| `temperature` | how random / creative the output is |
| `top_p`, `top_k` | which candidate words are even considered |
| `seed` | makes random output repeatable |
| `num_predict` | hard cap on output length |
| `stop` | strings that cut generation off early |

Ends with a short fill-in-the-blank exercise where you build the `options`
dictionary yourself.

### 2. `local_llm_demo.py` — prompts, limits, and tools

The core module, in four parts:

1. **Ollama basics** — one function, one HTTP POST, one reply.
2. **System prompt vs. user prompt** — the same question answered under
   different system prompts (helpful assistant, terse expert, ELI5, skeptical
   scientist, pirate). This is the cheapest lever you have on model behavior:
   no retraining, just a different instruction string.
3. **Where local models break down** — five failure modes shown on purpose:
   letter counting, multi-digit arithmetic, knowledge cutoff, multi-constraint
   instructions, and confident fabrication. Each comes with an explanation of
   *why* a 3-billion-parameter model tends to fail it. These failures are
   fluent and confident rather than obviously broken, which is exactly what
   makes them worth seeing.
4. **Basic tool use** — the fix for one of those failures. Instead of asking
   the model to *compute* arithmetic, we teach it to *ask for* a calculation,
   run real Python, and hand the exact result back. This is what "function
   calling" is, written out by hand.

### 3. `structured_output.py` — controlling the output format

How to get clean, machine-readable data instead of prose. Built as a
four-step progression, where each step fixes the previous one's weakness:

| approach | you get | when to use |
|---|---|---|
| naive prompt | maybe-valid text | never on its own — it's the baseline that breaks |
| `format: "json"` | valid JSON, any shape | quick JSON when the shape can vary |
| JSON Schema | valid JSON, *your* shape | reliable extraction into known fields |
| Pydantic | typed, validated object | production code that consumes the data |

Ends with an exercise where you write a JSON Schema yourself.

> **Version note:** the last two steps need a reasonably recent Ollama
> (roughly late 2024 or newer) for schema-constrained output. Both the
> notebook and the script detect an older server and tell you to fall back to
> `format: "json"`, rather than failing mysteriously.

---

## Requirements

**Notebooks** (this folder) — **on the NC State JupyterHub, marimo is already
configured**, so there's nothing to install. Elsewhere, `pip install -r
requirements.txt` gets you `marimo`, plus `pydantic` for the last section of
`structured_output.py`.

**Scripts** (`Python Scripts/`) — mostly nothing to install:

- `intro_prompting.py` — standard library only
- `local_llm_demo.py` — standard library only
- `structured_output.py` — needs `pydantic`, but only for its final section;
  the rest runs without it and the script skips that step gracefully

See `Python Scripts/requirements.txt` if you want that one dependency.

Ollama itself is a separate, non-pip install — that's what `setup_ollama.sh`
is for.

---

## Repository layout

```
Local LLM/
├── README.md                  you are here
├── Local_LLM_Guide.md         setting up Ollama on your own machine
├── .gitignore
├── setup_ollama.sh            no-sudo Ollama install + model pull (Linux)
├── Colab/
│   ├── prompting_local_llm_colab.ipynb   modules 1-3, self-contained
│   ├── rag_local_llm_colab.ipynb         RAG session, self-contained
│   └── corpus.zip                        corpus for the RAG upload fallback
├── requirements.txt           dependencies for the notebooks
├── intro_prompting.py         module 1 — notebook
├── local_llm_demo.py          module 2 — notebook
├── structured_output.py       module 3 — notebook
└── Python Scripts/
    ├── requirements.txt       dependencies for the scripts
    ├── intro_prompting.py     module 1 — script
    ├── local_llm_demo.py      module 2 — script
    └── structured_output.py   module 3 — script
```

---

## Troubleshooting

**"Can't reach Ollama" / connection refused.** The server isn't running.
Restart it with `nohup ollama serve > ~/ollama/ollama.log 2>&1 & disown`, or
just re-run `bash setup_ollama.sh`. Every *script* checks this before doing
anything and tells you plainly; the *notebooks* show a red
"🔴 Ollama not detected" indicator near the model dropdown instead.

**"model not found".** Pull it: `ollama pull llama3.2:3b`.

**A notebook doesn't reflect an edit I made to the file.** marimo caches the
notebook in its running process and won't pick up external changes. Close the
browser tab and restart `marimo edit <file>.py`.

**Ollama is running somewhere else.** Every file honors an environment
variable, so you don't have to edit code:

```bash
OLLAMA_BASE_URL=http://some-host:11434 python "Python Scripts/intro_prompting.py"
```

**The scripts feel slow.** Each one makes roughly a dozen calls to the model,
which takes a minute or two on a small machine. Comment out steps in `main()`
to focus on one.

**The models vanished after my session restarted.** Expected, unfortunately —
see the storage caveat in Quick start. Models live in `/tmp` on the Hub, which
isn't guaranteed to persist. Re-run `bash setup_ollama.sh`; it skips whatever
is still in place.

**I'm on the Hub and `marimo` isn't found.** It should be configured already —
check that you're in the workshop environment/kernel. If you'd rather not
troubleshoot it mid-session, `Python Scripts/` runs the same material with
plain `python` and no marimo at all.

---

## A note on the model

These modules use **`llama3.2:3b`** — a 3-billion-parameter model. It is small
enough to run comfortably on modest hardware and to stay completely private,
but it is far smaller than the 100B+ parameter models behind most commercial
chat products.

That trade-off is a feature of this material, not a limitation of it. Module 2
deliberately shows you where a small model breaks, because knowing the shape of
those failures — and which ones are fixed by better prompting, which by tools,
and which only by a bigger model — is the actual skill worth taking away.
