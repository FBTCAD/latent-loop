# Latent Loop: Iterative LLM Code Generation

> **"The schema is the new prompt."**

Latent Loop is a single-file implementation of iterative LLM inference for code generation. Instead of one-shot generation, it forces the model to externalize reasoning into structured JSON, refine it iteratively, then generate code from a verified specification.

## The Problem

One-shot LLM code generation fails silently. Models reason in transient hidden states that vanish after each forward pass—no audit trail, no refinement, no way to diagnose *why* the code is wrong.

## The Solution

Capture reasoning as a persistent, auditable JSON object:

```
Prompt → [JSON Reasoning State] → Refine → Refine → ... → Code
```

The model can't hallucinate that it solved a task—either the `solution_code` field has valid code or it doesn't.

## How It Works

### Three-Phase Pipeline

```
┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│   INIT              REFINE (×N)              GENERATE           │
│   ─────             ──────────               ────────           │
│                                                                 │
│   Decompose    →    Answer blockers    →    Code from           │
│   tasks into        Move unsolved to        solved tasks        │
│   solved/unsolved   solved with code                            │
│                                                                 │
│   state_init.json   state_refine_N.json     calculator.html     │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### The WRITE/SEARCH/BLOCKER Pattern

- **WRITE:** For tasks you know → write complete code immediately
- **SEARCH:** For blockers → search your knowledge, answer specifically  
- **BLOCKER:** For unknowns → ask specific questions (not vague concerns)

### Key Innovation: Forced JSON Output

Using LM Studio's `response_format` parameter with Pydantic schemas:

```python
from pydantic import BaseModel, Field
import lmstudio as lms

class SolvedTask(BaseModel):
    name: str
    solution_code: str = Field(..., min_length=20, 
        description="COMPLETE executable code, NOT a description")

class UnsolvedTask(BaseModel):
    name: str
    what_i_know: str = ""
    blockers: list[str] = Field(..., min_length=1,
        description="SPECIFIC questions to answer")

class ReasoningState(BaseModel):
    solved_tasks: list[SolvedTask]
    unsolved_tasks: list[UnsolvedTask]
    blocker_answers: list[str] = []

# Force valid JSON output matching schema
model = lms.llm()
result = model.respond(chat, response_format=ReasoningState)
state = ReasoningState(**result.parsed)  # Guaranteed valid!
```

## Installation

### Requirements

```bash
pip install pydantic lmstudio
```

### LM Studio Setup

1. Download [LM Studio](https://lmstudio.ai/)
2. Load a capable model (tested with Qwen, Llama, Mistral)
3. Enable local server (default: `localhost:1234`)

## Usage

```bash
python latent_loop.py
```

### Output Structure

```
output/
└── run_20260131_143022/
    ├── promptlong_instances_latent_v4.py  # Script copy
    ├── run_metadata.json                   # Config & model info
    ├── state_init.json                     # Initial decomposition
    ├── state_refine_1.json                 # Iteration 1
    ├── state_refine_2.json                 # Iteration 2
    ├── ...
    ├── state_final.json                    # Terminal state
    └── calculator.html                     # Generated code
```

## Configuration

```python
MAX_REFINEMENTS = 12          # Max inner loop iterations
MIN_SOLVE_PER_ITERATION = 2   # Target tasks to solve per iteration
OUTPUT_ROOT = Path("output")  # Output directory
```

## Example Output

### state_init.json (excerpt)

```json
{
  "solved_tasks": [
    {
      "name": "display_management",
      "solution_code": "function updateDisplay() { document.getElementById('display').textContent = displayValue; }"
    },
    {
      "name": "clear_function", 
      "solution_code": "function clearCalculator() { displayValue = '0'; operandA = null; pendingOperator = null; currentState = 'idle'; updateDisplay(); }"
    }
  ],
  "unsolved_tasks": [
    {
      "name": "power_function",
      "what_i_know": "Need to compute x^y using Math.pow()",
      "blockers": [
        "Is power immediate like sqrt, or deferred like addition?",
        "Should power button call inputOperator or a special handler?"
      ]
    }
  ],
  "blocker_answers": []
}
```

### Console Output

```
📁 Output directory: output/run_20260131_143022
🔄 Context isolation: FRESH LLM context for each phase
💾 State persistence: Only ReasoningState JSON carries between phases

[1] Decomposing: WRITE what you know, ASK about what you don't...
    ✓ Solved tasks: 4
    ✓ Unsolved tasks: 8
    ✓ Blockers to answer: 24

[2.1] SEARCHING to answer 24 blockers (fresh context)...
    ✓ Blocker answers accumulated: 8
    ✓ Solved: 7 (+3)
    ✓ Unsolved: 5

[2.2] SEARCHING to answer 15 blockers (fresh context)...
    ✓ Solved: 10 (+3)
    ✓ Unsolved: 2

🎉 All tasks solved after 3 refinement(s)!

[3] Generating code from solved tasks (fresh context)...
    ✓ HTML saved to: output/run_.../calculator.html
    ✓ Lines: 450
```

## Architecture

### Fresh Context Per Call

Every LLM call uses a new model handle and chat instance:

```python
def get_fresh_model():
    return lms.llm()

def make_fresh_chat(system_prompt: str):
    return lms.Chat(system_prompt)
```

**Why?** Eliminates KV cache bleeding between phases. Only the `ReasoningState` JSON carries between calls—no hidden context accumulation.

### State Replacement (Not Accumulation)

```python
state = init_state()                    # v1
state = refine_state(state, iteration)  # v2 (replaces v1)
state = refine_state(state, iteration)  # v3 (replaces v2)
code = generate_code(state)             # uses v3 only
```

Each iteration produces a complete new state. No growing context window.

## Part of NDLI

This script implements the **Inner Loop (Latent)** of the Nested Dual-Loop Inference framework:

```
┌─────────────────────────────────────────────────────────┐
│                    NDLI Framework                       │
│                                                         │
│   ┌─────────────────────────────────────────────────┐   │
│   │           Inner Loop (Latent) ← THIS SCRIPT     │   │
│   │   INIT → REFINE → REFINE → ... → GENERATE       │   │
│   └─────────────────────────────────────────────────┘   │
│                          │                              │
│                          ▼                              │
│   ┌─────────────────────────────────────────────────┐   │
│   │           Outer Loop (Ralph)                    │   │
│   │   Playwright tests → Feedback → FIX mode        │   │
│   └─────────────────────────────────────────────────┘   │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

For production use, wrap this script in an outer validation loop (Ralph) that runs behavioral tests and provides execution feedback.

## Inspiration

- **LoopCoder** (Beihang University): Proved architectural recurrence improves code generation
- **NDLI**: Brings iteration to inference layer without custom model training

## Limitations

- Adds latency (multiple LLM calls per generation)
- Schema design requires domain expertise
- No outer validation loop (use Ralph for production)
- Tested primarily on calculator generation task

## License

MIT

## Citation

If you use this work, please cite:

```
@misc{latentloop2026,
  title={Latent Loop: Iterative LLM Code Generation via Structured Reasoning Capture},
  author={Francis Benistant},
  year={2026},
  url={https://github.com/[your-repo]/latent-loop-v4}
}
```

## Related

- [Medium Article: Nested Dual-Loop Inference](#) — Full explanation of the NDLI framework
- [LinkedIn Post](#) — Short introduction to iterative inference

---

**Questions?** Open an issue or connect on [LinkedIn](#).
