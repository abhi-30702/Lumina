# Lumina — Product Requirements Document
### AI Personal Assistant | Local | Voice-Enabled | HUD Interface
**Version:** 1.1.0  
**Status:** Ready for Development  
**Platform:** Windows 10/11 | Python 3.11 | Mid-Range GPU (GTX 1060–3050)  
**Inference Mode:** Hybrid GPU + RAM split (16 layers each)

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Goals & Success Metrics](#2-goals--success-metrics)
3. [System Architecture](#3-system-architecture)
4. [Tech Stack](#4-tech-stack)
5. [Project Structure](#5-project-structure)
6. [Module Specifications](#6-module-specifications)
7. [Voice Pipeline](#7-voice-pipeline)
8. [Memory System](#8-memory-system)
9. [Custom Commands](#9-custom-commands)
10. [Web Search Module](#10-web-search-module)
11. [Notes & Files Module](#11-notes--files-module)
12. [HUD Interface](#12-hud-interface)
13. [Data Models](#13-data-models)
14. [Setup & Installation](#14-setup--installation)
15. [Environment Variables & Config](#15-environment-variables--config)
16. [Error Handling & Fallbacks](#16-error-handling--fallbacks)
17. [Performance Constraints](#17-performance-constraints)
18. [Future Roadmap](#18-future-roadmap)

---

## 1. Executive Summary

**Lumina** is a fully local, privacy-first AI personal assistant with a female voice persona, holographic HUD interface, persistent memory, web search capability, file/notes management, and a custom command system. All AI inference runs on-device using Ollama + Mistral 7B. No data leaves the user's machine.

### Core Principles

- **100% local** — No cloud APIs, no data leakage
- **Always listening** — Wake word activation ("Hey Lumina")
- **Persistent memory** — Remembers context across sessions via vector database
- **Extensible** — Custom commands can be added without touching core code
- **Visual** — Holographic dark HUD built with PyQt6

---

## 2. Goals & Success Metrics

| Goal | Metric | Target |
|------|--------|--------|
| Response latency | Time from query to first spoken word | < 3 seconds |
| STT accuracy | Word error rate on clear speech | < 10% WER |
| Wake word precision | False positive rate | < 1 per hour |
| Memory recall | Relevant context retrieved per query | ≥ 1 relevant chunk |
| Uptime | Crash-free session duration | 4+ hours |
| GPU VRAM usage | Peak VRAM during inference (16 GPU layers) | < 3.5 GB |
| RAM usage | Peak RAM during inference (16 CPU layers) | < 4 GB |

---

## 3. System Architecture

```
┌─────────────────────────────────────────────────────────┐
│                     LUMINA SYSTEM                       │
│                                                         │
│  ┌──────────┐    ┌──────────┐    ┌──────────────────┐  │
│  │ Mic Input│───▶│  Whisper │───▶│  Wake Word Det.  │  │
│  │ (PyAudio)│    │  (STT)   │    │  "Hey Lumina"    │  │
│  └──────────┘    └──────────┘    └────────┬─────────┘  │
│                                           │             │
│                                           ▼             │
│  ┌──────────────────────────────────────────────────┐  │
│  │              COMMAND ROUTER                      │  │
│  │  • Custom Command Parser                         │  │
│  │  • Intent Classifier                             │  │
│  │  • Fallback → LLM                                │  │
│  └───────┬──────────┬──────────┬──────────┬─────────┘  │
│          │          │          │          │             │
│          ▼          ▼          ▼          ▼             │
│       ┌──────┐ ┌────────┐ ┌──────┐ ┌──────────┐       │
│       │ LLM  │ │ Search │ │Notes │ │ Commands │       │
│       │Brain │ │(DDG)   │ │(FS)  │ │(Custom)  │       │
│       └──┬───┘ └────┬───┘ └──┬───┘ └────┬─────┘       │
│          │          │        │           │             │
│          └──────────┴────────┴───────────┘             │
│                          │                             │
│                          ▼                             │
│              ┌──────────────────────┐                  │
│              │   RESPONSE BUILDER   │                  │
│              │  (Context + Memory)  │                  │
│              └──────────┬───────────┘                  │
│                         │                              │
│              ┌──────────┴───────────┐                  │
│              │                      │                  │
│              ▼                      ▼                  │
│        ┌──────────┐          ┌────────────┐            │
│        │ Coqui TTS│          │  PyQt6 HUD │            │
│        │ (Voice)  │          │ (Display)  │            │
│        └──────────┘          └────────────┘            │
└─────────────────────────────────────────────────────────┘
```

### Data Flow (Single Request Cycle)

```
1. Mic captures audio continuously
2. Whisper detects "Hey Lumina" wake phrase
3. Whisper transcribes full user utterance
4. Command Router classifies intent
5. If custom command → execute directly
6. If search needed → DuckDuckGo fetch → inject into prompt
7. ChromaDB retrieves relevant memory chunks
8. Mistral 7B generates response with context
9. New interaction saved to ChromaDB
10. Coqui TTS converts response to speech
11. PyQt6 HUD updates with text + waveform animation
12. Audio plays through speakers
```

---

## 4. Tech Stack

| Layer | Tool | Version | Purpose |
|-------|------|---------|---------|
| Language | Python | 3.11 | Core runtime |
| LLM Runtime | Ollama | Latest | Local model server |
| LLM Model | Mistral 7B Instruct | v0.3 | AI brain |
| Speech-to-Text | OpenAI Whisper | base / small | Voice input |
| Text-to-Speech | Coqui TTS | Latest | Female voice output |
| Vector DB | ChromaDB | Latest | Persistent memory |
| Embeddings | sentence-transformers | all-MiniLM-L6-v2 | Memory encoding |
| Web Search | duckduckgo-search | Latest | Live web queries |
| UI Framework | PyQt6 | Latest | HUD window |
| Audio I/O | PyAudio + sounddevice | Latest | Mic + speaker |
| Config | python-dotenv | Latest | Environment vars |
| Logging | loguru | Latest | Structured logs |

---

## 5. Project Structure

```
lumina/
│
├── main.py                  # Entry point — boots all subsystems
├── config.py                # Central config loader
├── Modelfile                # Ollama GPU/RAM layer split config
├── requirements.txt         # All pip dependencies
├── .env                     # Environment variables (gitignored)
├── README.md
│
├── core/
│   ├── __init__.py
│   ├── brain.py             # LLM orchestration + Ollama interface
│   ├── memory.py            # ChromaDB read/write/search
│   ├── router.py            # Intent detection + command routing
│   └── response_builder.py  # Assembles final prompt with context
│
├── voice/
│   ├── __init__.py
│   ├── listener.py          # Continuous mic capture + wake word
│   ├── transcriber.py       # Whisper STT
│   └── speaker.py           # Coqui TTS + audio playback
│
├── commands/
│   ├── __init__.py
│   ├── registry.py          # Command registration system
│   ├── built_in.py          # Built-in commands (help, clear, etc.)
│   └── user_commands.py     # User-defined custom commands
│
├── tools/
│   ├── __init__.py
│   ├── search.py            # DuckDuckGo web search
│   └── notes.py             # File read/write/list
│
├── ui/
│   ├── __init__.py
│   ├── hud.py               # Main PyQt6 HUD window
│   ├── waveform.py          # Animated waveform widget
│   ├── chat_panel.py        # Scrollable chat log
│   └── assets/
│       └── lumina_icon.png
│
├── memory/                  # ChromaDB persistent storage (auto-created)
│
├── notes/                   # User notes storage (auto-created)
│   └── .gitkeep
│
└── logs/                    # Loguru log files (auto-created)
    └── .gitkeep
```

---

## 6. Module Specifications

### 6.1 `main.py` — Entry Point

**Responsibilities:**
- Initialize all subsystems in order
- Start PyQt6 application loop
- Launch listener thread (non-blocking)
- Handle graceful shutdown (Ctrl+C or window close)

**Boot sequence:**
```
1. Load config (.env)
2. Init ChromaDB (memory/)
3. Load Whisper model
4. Load Coqui TTS model
5. Connect to Ollama (verify model available)
6. Register commands
7. Launch HUD window
8. Start mic listener thread
9. Lumina speaks: "Lumina online. Ready when you are."
```

---

### 6.2 `core/brain.py` — LLM Orchestration

**Responsibilities:**
- Send prompts to Ollama (Mistral 7B) using `lumina-model` (custom Modelfile)
- Maintain conversation history (last N turns)
- Inject memory context into system prompt
- Stream responses token-by-token to HUD
- Pass `num_gpu` and `num_thread` options on every inference call

**Ollama Modelfile (`Modelfile` in project root):**
```
FROM mistral:7b-instruct

PARAMETER num_gpu 16
PARAMETER num_thread 8
```

Create the model once before first run:
```bash
ollama create lumina-model -f Modelfile
```

**System Prompt (Lumina Persona):**
```
You are Lumina, an advanced AI personal assistant. You are intelligent, 
precise, and speak with calm confidence. You are female. Your responses 
are concise unless detail is requested. You have memory of past conversations. 
You never say "As an AI" or "I cannot". If you don't know something, 
you say "Let me find that for you" and trigger a search.
```

**Key methods:**
```python
def chat(user_input: str, context: list[str]) -> str
def stream_chat(user_input: str, context: list[str]) -> Generator[str]
def reset_history() -> None
```

**Ollama call options (passed on every request):**
```python
options={
    "num_gpu": config.ollama_num_gpu,      # 16 layers on GPU
    "num_thread": config.ollama_num_thread, # 8 threads for RAM layers
    "temperature": config.llm_temperature,
    "num_predict": config.llm_max_tokens,
}
```

---

### 6.3 `core/memory.py` — ChromaDB Memory

**Responsibilities:**
- Store every conversation turn as a vector
- Retrieve top-K relevant memories for each new query
- Store and retrieve named facts ("remember that...")
- Support memory deletion by tag or date

**Collections:**
- `conversations` — full turn history (user + Lumina)
- `facts` — explicitly saved facts ("my name is...", "I prefer...")

**Key methods:**
```python
def save_turn(user_msg: str, lumina_msg: str) -> None
def save_fact(fact: str, tag: str) -> None
def retrieve_context(query: str, top_k: int = 5) -> list[str]
def forget_fact(tag: str) -> None
def clear_all() -> None
```

---

### 6.4 `core/router.py` — Intent & Command Router

**Responsibilities:**
- Parse user input for command triggers
- Route to correct handler (command / search / brain)
- Extract command arguments

**Routing Logic:**
```
Input received
│
├── Matches custom command keyword? → commands/registry.py
├── Contains "search for / look up / find"? → tools/search.py
├── Contains "save a note / write this down"? → tools/notes.py
├── Contains "read my notes / what did I save"? → tools/notes.py
├── Contains "remember that"? → core/memory.py (save_fact)
├── Contains "forget"? → core/memory.py (forget_fact)
└── Default → core/brain.py (LLM)
```

---

## 7. Voice Pipeline

### 7.1 Wake Word Detection (`voice/listener.py`)

- Continuously records audio in 2-second chunks using PyAudio
- Runs Whisper `tiny` model on each chunk (low latency, low VRAM)
- Checks transcription for "hey lumina" (fuzzy match, threshold 0.85)
- On detection: plays a soft chime sound, begins full capture

**Capture modes:**
- **Triggered capture:** Records until 1.5s of silence detected
- **Max duration:** 30 seconds per utterance

### 7.2 Transcription (`voice/transcriber.py`)

- Uses Whisper `base` model for full transcription (good accuracy/speed balance on GTX 1060+)
- Language: English (locked for speed, configurable)
- Returns cleaned text (stripped punctuation artifacts)

**Model selection by GPU VRAM:**
```
< 4 GB VRAM  → whisper tiny
4–6 GB VRAM  → whisper base  (default)
6+ GB VRAM   → whisper small
```

### 7.3 Text-to-Speech (`voice/speaker.py`)

- Uses Coqui TTS with `tts_models/en/ljspeech/tacotron2-DDC` (female, natural)
- Generates audio to temp WAV → plays via sounddevice
- Speaking state broadcast to HUD (triggers waveform animation)
- Queue-based: responses queued if already speaking

**Voice parameters:**
```python
SPEECH_SPEED = 1.05       # Slightly faster than default
SPEECH_PITCH = 1.0        # Natural female pitch
```

---

## 8. Memory System

### 8.1 Architecture

ChromaDB runs as an embedded (in-process) database stored in `memory/`. No server needed.

Embeddings generated via `sentence-transformers/all-MiniLM-L6-v2` (80 MB, runs on CPU).

### 8.2 Memory Schema

**Conversations collection:**
```json
{
  "id": "conv_20240115_143022_001",
  "document": "User: What is quantum computing? Lumina: Quantum computing uses qubits...",
  "metadata": {
    "timestamp": "2024-01-15T14:30:22",
    "user_msg": "What is quantum computing?",
    "lumina_msg": "Quantum computing uses qubits...",
    "session_id": "sess_abc123"
  }
}
```

**Facts collection:**
```json
{
  "id": "fact_name_001",
  "document": "The user's name is Alex.",
  "metadata": {
    "tag": "name",
    "timestamp": "2024-01-15T14:30:22",
    "source": "explicit"
  }
}
```

### 8.3 Context Injection

Before each LLM call, retrieve top 5 relevant memory chunks and prepend to system prompt:

```
[MEMORY CONTEXT]
- User's name is Alex (saved 3 days ago)
- User prefers concise answers (saved 1 week ago)
- Previous conversation: User asked about quantum computing...
[END MEMORY]
```

---

## 9. Custom Commands

### 9.1 Command Registry (`commands/registry.py`)

Commands are Python functions decorated with `@lumina_command`:

```python
@lumina_command(
    trigger=["open calculator", "launch calc"],
    description="Opens Windows Calculator"
)
def open_calculator(args: str) -> str:
    import subprocess
    subprocess.Popen("calc.exe")
    return "Opening calculator."
```

### 9.2 Built-in Commands

| Trigger Phrases | Action |
|----------------|--------|
| "clear memory" / "forget everything" | Wipes ChromaDB conversations |
| "what time is it" | Returns current time |
| "what's today's date" | Returns current date |
| "stop" / "cancel" | Interrupts current TTS |
| "help" / "what can you do" | Lists available commands |
| "remember that [fact]" | Saves fact to ChromaDB |
| "forget [tag]" | Deletes fact from ChromaDB |
| "save a note: [content]" | Writes to notes/ folder |
| "read my notes" | Lists and reads back saved notes |
| "search for [query]" | Triggers DuckDuckGo search |
| "goodbye" / "sleep" | Puts Lumina in standby mode |
| "wake up" | Exits standby mode |

### 9.3 Adding Custom Commands

Users add commands to `commands/user_commands.py`:

```python
@lumina_command(
    trigger=["play music", "start spotify"],
    description="Opens Spotify"
)
def play_music(args: str) -> str:
    import subprocess
    subprocess.Popen(r"C:\Users\...\Spotify.exe")
    return "Opening Spotify for you."
```

No restart required — commands reload on `lumina reload commands`.

---

## 10. Web Search Module

### 10.1 Implementation (`tools/search.py`)

Uses `duckduckgo-search` Python library — no API key required.

**Search trigger detection:**
```python
SEARCH_TRIGGERS = [
    "search for", "look up", "find out", "what is the latest",
    "current news", "who is", "what happened to", "price of",
    "weather in", "how much does"
]
```

**Search flow:**
```
1. Extract search query from user input
2. Run DuckDuckGo text search (top 3 results)
3. Extract titles + snippets
4. Inject into LLM prompt as [WEB CONTEXT]
5. LLM synthesizes answer from results
6. Lumina cites sources in response
```

**Key methods:**
```python
def search(query: str, max_results: int = 3) -> list[dict]
def format_for_prompt(results: list[dict]) -> str
def extract_query(user_input: str) -> str
```

---

## 11. Notes & Files Module

### 11.1 Implementation (`tools/notes.py`)

All notes stored as `.md` files in `notes/` directory.

**Supported operations:**

| Command | Action | File Operation |
|---------|--------|---------------|
| "save a note: [content]" | Creates new note | Write new `.md` file |
| "add to my notes: [content]" | Appends to latest note | Append to file |
| "read my notes" | Reads all notes aloud | Read all `.md` files |
| "read note about [topic]" | Reads specific note | Fuzzy search filenames |
| "delete note about [topic]" | Removes note | Delete `.md` file |
| "list my notes" | Lists all note titles | List directory |

**File naming:** `note_YYYYMMDD_HHMMSS.md`

**Key methods:**
```python
def save_note(content: str, title: str = None) -> str
def append_note(content: str) -> str
def read_note(topic: str = None) -> str
def list_notes() -> list[str]
def delete_note(topic: str) -> str
```

---

## 12. HUD Interface

### 12.1 Window Specifications

| Property | Value |
|----------|-------|
| Default size | 900 × 600 px |
| Theme | Dark holographic (`#020C14` background) |
| Primary color | Cyan (`#00D4FF`) |
| Accent color | Green (`#00FF88`) |
| Font (UI) | Orbitron (headers), Share Tech Mono (body) |
| Always on top | Optional (toggle) |
| Frameless | Yes (custom titlebar) |
| Transparency | 95% opacity |

### 12.2 Layout

```
┌──────────────────────────────────────────────────────┐
│  ◉ LUMINA v1.0          [ONLINE ●]        [—][□][✕]  │  ← Custom titlebar
├──────────────────────────────────────────────────────┤
│                                                      │
│              ┌──────────────────┐                    │
│              │   ARC REACTOR    │  ← Animated SVG ring│
│              │    (spinning)    │                    │
│              └──────────────────┘                    │
│                                                      │
│  ┌────────────────────────────────────────────────┐  │
│  │  CONVERSATION LOG                              │  │  ← Scrollable
│  │  ────────────────────────────────────────────  │  │
│  │  > USER // What's the weather in Mumbai?       │  │
│  │  LUMINA // Let me find that. [searching...]    │  │
│  │  LUMINA // Mumbai is currently 32°C, partly... │  │
│  └────────────────────────────────────────────────┘  │
│                                                      │
│  ████████████████████░░░░░░░░░░░░  ← Waveform        │
│                                                      │
│  ┌──────────────────────────────────────────────┐    │
│  │ > _                                    [MIC] │    │  ← Text input fallback
│  └──────────────────────────────────────────────┘    │
│                                                      │
│  STATUS: LISTENING...          SESSION: 00:14:32     │  ← Status bar
└──────────────────────────────────────────────────────┘
```

### 12.3 Animations

| Element | Animation |
|---------|-----------|
| Arc reactor outer ring | Continuous slow rotation (4s/rev) |
| Arc reactor inner ring | Reverse rotation (7s/rev) |
| Arc reactor core | Pulse glow on speaking |
| Waveform | Real-time audio amplitude bars |
| Message appear | Fade in + slide up |
| Status dot | Slow blink |
| Mic button | Pulse when recording |

### 12.4 HUD States

| State | Visual Indicator | Color |
|-------|-----------------|-------|
| Idle / Listening | Status: LISTENING | Cyan dim |
| Wake word detected | Status: ACTIVE | Cyan bright |
| Processing | Status: THINKING... | Amber pulse |
| Speaking | Waveform animated | Green |
| Searching | Status: SEARCHING... | Blue |
| Error | Status: ERROR | Red flash |
| Standby | Status: STANDBY | Gray |

---

## 13. Data Models

### 13.1 Config (`config.py`)

```python
@dataclass
class LuminaConfig:
    # LLM
    ollama_host: str = "http://localhost:11434"
    ollama_model: str = "lumina-model"           # Custom Modelfile model
    ollama_num_gpu: int = 16                     # Mistral layers on GPU (~2.8 GB VRAM)
    ollama_num_thread: int = 8                   # CPU threads for RAM layers
    llm_max_tokens: int = 512
    llm_temperature: float = 0.7
    llm_context_window: int = 4096

    # Voice
    whisper_model: str = "base"
    whisper_language: str = "en"
    wake_word: str = "hey lumina"
    wake_word_threshold: float = 0.85
    silence_timeout: float = 1.5
    max_record_seconds: int = 30
    tts_model: str = "tts_models/en/ljspeech/tacotron2-DDC"

    # Memory
    chroma_path: str = "./memory"
    memory_top_k: int = 5
    conversation_history_turns: int = 10

    # Notes
    notes_path: str = "./notes"

    # UI
    window_width: int = 900
    window_height: int = 600
    always_on_top: bool = False
    opacity: float = 0.95
```

### 13.2 Message Model

```python
@dataclass
class Message:
    role: str           # "user" | "lumina" | "system"
    content: str
    timestamp: datetime
    source: str         # "voice" | "text" | "command"
    session_id: str
```

---

## 14. Setup & Installation

### Prerequisites

```
- Windows 10/11 (64-bit)
- Python 3.11
- NVIDIA GPU with CUDA 11.8+ (GTX 1060 minimum)
- Ollama installed (https://ollama.ai)
- Git
- 15 GB free disk space (models)
```

### Step-by-Step Install

```bash
# 1. Clone repo
git clone https://github.com/yourname/lumina.git
cd lumina

# 2. Create virtual environment
python -m venv venv
venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Pull Mistral base model via Ollama
ollama pull mistral:7b-instruct

# 5. Create lumina-model with GPU/RAM split (Modelfile already in repo)
ollama create lumina-model -f Modelfile

# 5. Download Whisper model (auto on first run)
python -c "import whisper; whisper.load_model('base')"

# 6. Download Coqui TTS model (auto on first run)
python -c "from TTS.api import TTS; TTS('tts_models/en/ljspeech/tacotron2-DDC')"

# 7. Copy environment config
cp .env.example .env

# 8. Run Lumina
python main.py
```

### `requirements.txt`

```
ollama==0.2.1
openai-whisper==20231117
TTS==0.22.0
chromadb==0.5.0
sentence-transformers==3.0.0
duckduckgo-search==6.1.0
PyQt6==6.7.0
PyAudio==0.2.14
sounddevice==0.4.7
soundfile==0.12.1
numpy==1.26.4
python-dotenv==1.0.1
loguru==0.7.2
fuzzywuzzy==0.18.0
python-Levenshtein==0.25.1
torch==2.3.0+cu118
```

---

## 15. Environment Variables & Config

### `.env.example`

```env
# LLM
OLLAMA_HOST=http://localhost:11434
OLLAMA_MODEL=lumina-model
OLLAMA_NUM_GPU=16
OLLAMA_NUM_THREAD=8
LLM_TEMPERATURE=0.7
LLM_MAX_TOKENS=512

# Voice
WHISPER_MODEL=base
WAKE_WORD=hey lumina
WAKE_WORD_THRESHOLD=0.85
SILENCE_TIMEOUT=1.5

# TTS
TTS_MODEL=tts_models/en/ljspeech/tacotron2-DDC

# Memory
CHROMA_PATH=./memory
MEMORY_TOP_K=5

# Notes
NOTES_PATH=./notes

# UI
ALWAYS_ON_TOP=false
WINDOW_OPACITY=0.95

# Logging
LOG_LEVEL=INFO
LOG_PATH=./logs
```

---

## 16. Error Handling & Fallbacks

| Failure | Fallback Behavior |
|---------|-----------------|
| Ollama not running | Lumina says "Brain offline, please start Ollama" + red HUD state |
| Whisper model missing | Auto-download on startup |
| No microphone detected | Enable text-input-only mode |
| DuckDuckGo search fails | Lumina says "Search unavailable, answering from memory" |
| ChromaDB corruption | Backup + reinitialize empty DB |
| TTS generation fails | Print response to HUD only (silent mode) |
| GPU out of memory | Reduce `OLLAMA_NUM_GPU` by 4 and restart (shift layers to RAM) |
| Wake word false positive | Confirmation tone + 2s window for real command |

---

## 17. Performance Constraints

### Hybrid GPU + RAM Memory Budget (GTX 1060 6GB + 16GB RAM)

Mistral 7B has 32 layers. With `num_gpu=16`, exactly half run on GPU and half on CPU RAM.

| Component | Location | Memory Used |
|-----------|----------|-------------|
| Mistral 7B — 16 layers (GPU) | VRAM | ~2.8 GB |
| Whisper base | VRAM | ~150 MB |
| Coqui TTS | VRAM | ~400 MB |
| PyQt6 HUD | VRAM | ~50 MB |
| **Total VRAM** | | **~3.4 GB** ✅ |
| Mistral 7B — 16 layers (CPU) | RAM | ~2.8 GB |
| ChromaDB + embeddings | RAM | ~400 MB |
| Python runtime + misc | RAM | ~300 MB |
| **Total RAM** | | **~3.5 GB** ✅ |

> To increase speed: raise `OLLAMA_NUM_GPU` to 20–24 if VRAM headroom allows after Whisper + TTS load.
> To reduce VRAM: lower `OLLAMA_NUM_GPU` to 8–12 and shift more layers to RAM.

### Layer Tuning Guide

| `num_gpu` | VRAM Used | Speed | Recommended For |
|-----------|-----------|-------|----------------|
| 8 | ~1.5 GB | Moderate | Very low VRAM systems |
| 16 | ~2.8 GB | Good ✅ | GTX 1060 6GB (default) |
| 20 | ~3.5 GB | Fast | GTX 1070+ |
| 24 | ~4.2 GB | Faster | GTX 1080 / RTX 2060 |
| 32 | ~5.5 GB | Full GPU | RTX 3060+ only |

### Latency Targets

| Step | Target |
|------|--------|
| Wake word detection | < 200ms |
| STT transcription (5s audio) | < 1.5s |
| Memory retrieval | < 100ms |
| LLM first token | < 2s |
| TTS generation (20 words) | < 800ms |
| **Total end-to-end** | **< 4.5s** |

---

## 18. Future Roadmap

### v1.1
- PC control (open apps, control volume, take screenshots)
- Emotion detection from voice tone
- Multiple voice profiles

### v1.2
- Local document Q&A (PDF/DOCX ingestion into ChromaDB)
- Calendar integration (local .ics files)
- Lumina's own task list + reminders

### v1.3
- Face/presence detection via webcam (auto wake/sleep)
- Multi-language support
- Plugin system for third-party extensions

### v2.0
- Switch to LLaMA 3 or Gemma 2 for improved quality
- Real-time streaming voice response (overlap TTS + generation)
- Mobile companion app (local network only)

---

## Appendix A — Claude Code Prompt

Use this prompt to start building with Claude Code:

```
I am building Lumina, a local AI assistant for Windows.
Full PRD is in lumina_prd.md — read it before writing any code.

Start with the following build order:
1. Project scaffold (folders + __init__.py files)
2. Modelfile (Ollama GPU/RAM split — num_gpu=16, num_thread=8)
3. config.py (dataclass-based config loader with ollama_num_gpu + ollama_num_thread fields)
4. core/memory.py (ChromaDB integration)
5. core/brain.py (Ollama + lumina-model, pass num_gpu/num_thread on every call)
6. voice/transcriber.py (Whisper STT)
7. voice/speaker.py (Coqui TTS)
8. voice/listener.py (mic capture + wake word)
9. core/router.py (intent routing)
10. commands/registry.py + built_in.py
11. tools/search.py + tools/notes.py
12. ui/hud.py (PyQt6 holographic HUD)
13. main.py (boot sequence + threading)

Rules:
- Python 3.11 only
- Type hints on all functions
- Loguru for all logging
- Every module must have a standalone test block under if __name__ == "__main__"
- Never hardcode paths — always use config.py
- All GPU operations must have CPU fallback
- Ollama model name is always "lumina-model" (never "mistral:7b-instruct" directly)
- Always pass num_gpu and num_thread in Ollama options dict
```

---

*Lumina PRD v1.1 — Updated with Hybrid GPU/RAM Split — Built for Claude Code*
