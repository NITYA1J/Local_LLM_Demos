# Setting Up a Local LLM on Your Own Machine

A step-by-step guide to installing [Ollama](https://ollama.com), downloading a
model, and running it entirely on your own computer.

This is the **personal-machine** counterpart to the workshop material. On the
NC State JupyterHub, `setup_ollama.sh` does all of this for you — this guide is
for taking the same setup home.

**What you'll end up with:** a language model running on your own hardware,
with no API key, no account, no usage charges, and no data leaving your
machine. It will be slower and less capable than a commercial cloud model, but
it is entirely yours and works offline.

---

## Before you start

### Administrative access

**Plan on needing administrator (admin) rights on your machine.** Installing
software system-wide requires it, and on a university-managed or work-issued
computer you may not have it by default — check before you begin, and contact
your IT support if you don't have it.

That said, the requirement varies by platform, and there are ways around it:

| Platform | Admin needed? |
|---|---|
| **Windows** | **No.** The official installer installs into your home folder and does not require Administrator. |
| **macOS** | **Usually yes** — the standard install puts the app in the system-wide `Applications` folder, and it will prompt for your password to link the command-line tool. A no-admin path exists (see below). |
| **Linux** | **Yes** for the standard install — the install script uses `sudo`. A no-admin path exists (see below). |

If you don't have admin rights, skip to
[No admin access?](#no-admin-access) at the end — you have options.

### Hardware and system requirements

| | Minimum | Comfortable |
|---|---|---|
| **RAM** | 8 GB | 16 GB or more |
| **Free disk** | ~10 GB | 20 GB+ if you'll try several models |
| **GPU** | Not required — runs on CPU | Any modern GPU speeds things up a lot |

Operating system versions:

- **macOS** — Sonoma (14) or newer. Apple Silicon (M-series) gets GPU
  acceleration automatically; Intel Macs run CPU-only and will be slower.
- **Windows** — Windows 10 22H2 or newer, Home or Pro.
- **Linux** — a mainstream distribution; Ubuntu is the best-supported.

A note on expectations: the model used here (`llama3.2:3b`) has **3 billion
parameters**. It runs fine on a laptop without a GPU, but responses may take
several seconds. That's normal, not a broken install.

---

## Step 1 — Install Ollama

Ollama is the program that downloads, stores, and runs models for you. Pick
your platform.

### macOS

1. Go to **<https://ollama.com/download/mac>** and download the `.dmg` file.
2. Open the downloaded file. A window will appear showing the Ollama app.
3. **Drag the Ollama icon into the `Applications` folder.** If macOS asks for
   your password here, that's the admin prompt — you'll need it to continue.
4. Open Ollama from your Applications folder (or Launchpad).
5. On first launch it may ask permission to install the command-line tool.
   **Approve this** — the workshop material needs it. This is a second admin
   prompt.

### Windows

1. Go to **<https://ollama.com/download/windows>** and download
   `OllamaSetup.exe`.
2. Run the installer and follow the prompts. **No Administrator rights are
   required** — it installs into your user account.
3. When it finishes, Ollama starts automatically and runs in the background.
   You'll see its icon in the system tray.

> If your `C:` drive is short on space, you can install elsewhere by running
> the installer from a terminal:
> `OllamaSetup.exe /DIR="D:\some\location"`

### Linux

Open a terminal and run:

```bash
curl -fsSL https://ollama.com/install.sh | sh
```

This script uses `sudo` and will ask for your password. It installs Ollama and
sets it up as a background service.

> **Cautious about piping a script into your shell?** That's a reasonable
> instinct. You can read it first at <https://ollama.com/install.sh>, or use
> the manual install documented at <https://docs.ollama.com/linux>.

---

## Step 2 — Confirm it installed

Open a **new** terminal window (important — a new window picks up the updated
PATH) and run:

```bash
ollama --version
```

On Windows, use Command Prompt or PowerShell.

You should see a version number printed. If instead you get "command not
found" or "not recognized," see [Troubleshooting](#troubleshooting).

---

## Step 3 — Download a model

Models are downloaded once and then stored on your machine. Pull the one the
workshop material uses:

```bash
ollama pull llama3.2:3b
```

This downloads roughly **2 GB**, so it may take a few minutes. You'll see a
progress bar.

**If you also plan to do the RAG session**, pull the embedding models now
while you're at it:

```bash
ollama pull nomic-embed-text
ollama pull mxbai-embed-large
```

Confirm what you have:

```bash
ollama list
```

You should see each model you pulled, with its size.

---

## Step 4 — Talk to the model

The quickest way to confirm everything works — start an interactive chat:

```bash
ollama run llama3.2:3b
```

You'll get a prompt where you can type a question:

```
>>> Why is the sky blue?
```

The model will respond. The first response after loading may take a few extra
seconds while the model is read into memory.

**To exit the chat**, type `/bye` or press `Ctrl+D`.

If you got a sensible answer, your local LLM is working. 🎉

---

## Step 5 — Run it as a server (for the workshop code)

Interactive chat is fine for poking around, but the workshop notebooks and
scripts talk to Ollama over HTTP. For that, Ollama needs to be running as a
**server**.

**On macOS and Windows**, this already happens — the Ollama app runs in the
background from the moment you launch it. Look for the icon in your menu bar
(macOS) or system tray (Windows). If it's there, you're set.

**On Linux** with the standard install, it's running as a system service.

If you ever need to start it manually, in its own terminal window:

```bash
ollama serve
```

Leave that window open while you work.

### Confirm the server is reachable

```bash
curl http://localhost:11434/api/tags
```

You should get a JSON list of your installed models. Port **11434** on this
machine is exactly what every file in this repository talks to (the code
writes it as `http://127.0.0.1:11434`, which is the same thing as
`localhost`).

---

## Step 6 — Run the workshop material

With the server running, you can now use either version of the modules:

**Plain Python scripts** (nothing to install beyond Python):

```bash
cd "Python Scripts"
python intro_prompting.py
```

**Interactive notebooks** (needs marimo — `pip install marimo`). Note the
`cd ..` — the notebooks live in the *top* folder, and share filenames with the
scripts, so running this from the wrong directory silently opens the script
instead:

```bash
cd ..                        # back to the Local LLM folder
marimo edit intro_prompting.py
```

Each script checks the connection first and will tell you clearly if Ollama
isn't reachable.

---

## Useful commands

| Command | What it does |
|---|---|
| `ollama list` | Show models you've downloaded |
| `ollama pull <model>` | Download a model |
| `ollama run <model>` | Start an interactive chat |
| `ollama ps` | Show which models are loaded in memory right now |
| `ollama rm <model>` | Delete a model and free its disk space |
| `ollama --version` | Check your Ollama version |
| `ollama serve` | Start the server manually |

Browse other available models at <https://ollama.com/search>.

---

## Managing disk space

Models are large and they accumulate. A few things worth knowing:

- **See what you're using:** `ollama list` shows the size of each model.
- **Delete one you're done with:** `ollama rm <model name>`.
- **Where they live:** `~/.ollama` on macOS and Linux;
  `%HOMEPATH%\.ollama` on Windows.
- **Move them elsewhere:** set the `OLLAMA_MODELS` environment variable to a
  different folder — useful if your main drive is small but you have an
  external or secondary drive.

Bigger models are more capable but slower and larger. A 3B model like the one
here is a good laptop default; 7B–8B models are noticeably better and still
feasible with 16 GB of RAM.

---

## Troubleshooting

**"command not found" / "'ollama' is not recognized"**
Open a *new* terminal window — PATH changes don't apply to windows that were
already open. If it still fails on macOS, the command-line tool may not have
been linked; open the Ollama app and approve the prompt. On Windows, try
signing out and back in.

**"connection refused" when running the workshop code**
The server isn't running. On macOS/Windows, launch the Ollama app and check
for its menu-bar/tray icon. On Linux, run `ollama serve` in a spare terminal,
or `sudo systemctl start ollama`.

**"model not found"**
You haven't pulled it yet, or there's a typo. Run `ollama list` to see what
you actually have, then `ollama pull llama3.2:3b`.

**Responses are very slow**
Expected on a CPU-only machine, especially the first response after loading.
If it's unusably slow, try a smaller model (`ollama pull llama3.2:1b`) or
close memory-hungry applications. Check what's loaded with `ollama ps`.

**The download keeps failing partway**
Re-run the same `ollama pull` command — it resumes rather than starting over.
Corporate or campus networks with strict filtering can interfere; try a
different network if it persists.

**Out of disk space**
`ollama rm` any models you're not using, or point `OLLAMA_MODELS` at a drive
with more room.

---

## No admin access?

If you're on a managed machine and can't install software, you still have
options — in rough order of preference:

1. **You may not actually need admin.** On **Windows**, the standard installer
   requires no Administrator rights at all. Try it before assuming you're
   blocked.

2. **Install without admin on Linux.** Ollama can be installed entirely inside
   your home directory. This repository's own `setup_ollama.sh` does exactly
   that — it was written for the JupyterHub, where nobody has `sudo`. Read it
   for a working no-sudo example you can adapt.

3. **Install without admin on macOS.** Instead of dragging the app to the
   system-wide `Applications` folder, put `Ollama.app` somewhere in your home
   directory and decline the "Move to Applications?" prompt on first launch.
   You'll then need to add the CLI to your PATH yourself; it lives inside the
   app at `Ollama.app/Contents/Resources/ollama`.

4. **Ask IT.** A short, specific request ("please install Ollama from
   ollama.com, it's an open-source tool for running language models locally")
   is often approved quickly.

5. **Just use the JupyterHub.** The workshop environment already has
   everything set up. Running locally is a convenience, not a requirement.

---

## Where to go next

- **This repository's `README.md`** — the workshop modules themselves.
- **Ollama documentation** — <https://docs.ollama.com>
- **Model library** — <https://ollama.com/search>

Once Ollama is running, everything in this repository works identically on
your machine and on the Hub. The code makes no assumptions about where it is.
