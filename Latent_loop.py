"""
HTML Calculator - Latent Loop: Fresh Instances Version
==========================================================

Features:
- output/ directory auto-created
- timestamped run directory per execution
- save intermediate reasoning states as JSON
- save final HTML
- save a copy of this script
- save run_metadata.json (model, config, timestamp, etc.)

Key Changes (instances version):
- FRESH CONTEXT PER CALL: Each LLM call uses a new model handle and Chat instance
- NO MODEL REUSE: Eliminates any possible KV cache bleeding between phases
- EXPLICIT ISOLATION: Only the ReasoningState JSON carries between phases
- All improvements retained (WRITE/SEARCH/BLOCKER pattern)

This version exists to TEST whether fresh instances improve results compared
to reusing a single model handle. The hypothesis is that fresh instances
may reduce "context rot" but the current v4 already works well.
"""

import re
import json
import shutil
from pathlib import Path
from datetime import datetime
from pydantic import BaseModel, Field
from typing import Literal
import lmstudio as lms


# ---------------------------------------------------------------------------
# Task Models - Separate lists for solved vs unsolved (enables numeric forcing)
# ---------------------------------------------------------------------------

class SolvedTask(BaseModel):
    """A task we know how to solve - WITH COMPLETE CODE."""
    
    name: str = Field(
        ...,
        min_length=3,
        max_length=50,
        description="Task identifier, e.g., 'digit_after_operator'"
    )
    
    solution_code: str = Field(
        ...,
        min_length=20,
        max_length=800,
        description="""COMPLETE executable pseudo-code or JavaScript.
        NOT a description - actual CODE.
        Example: 'if (shouldReplaceDisplay) { displayValue = digit; shouldReplaceDisplay = false; } else { displayValue += digit; }'"""
    )


class UnsolvedTask(BaseModel):
    """A task we don't fully know how to solve - WITH SPECIFIC QUESTIONS."""
    
    name: str = Field(
        ...,
        min_length=3,
        max_length=50,
        description="Task identifier"
    )
    
    what_i_know: str = Field(
        default="",
        max_length=300,
        description="What I've figured out so far (partial understanding)"
    )
    
    blockers: list[str] = Field(
        ...,
        min_length=1,
        max_length=5,
        description="""SPECIFIC QUESTIONS I need answered to solve this.
        NOT vague like 'need to understand better'.
        SPECIFIC like 'Is power (^) immediate like sqrt, or deferred like +?'"""
    )


# ---------------------------------------------------------------------------
# Reasoning State - v4 with Active Search Pattern
# ---------------------------------------------------------------------------

class ReasoningState(BaseModel):
    """
    Structured reasoning state with WRITE/SEARCH pattern.
    Uses separate lists for solved vs unsolved (enables numeric forcing).
    
    THIS IS THE ONLY STATE THAT PERSISTS BETWEEN LLM CALLS.
    Each LLM call starts with a fresh context - no memory except this JSON.
    """
    
    # === SOLVED TASKS (with complete code) ===
    solved_tasks: list[SolvedTask] = Field(
        ...,
        min_length=1,
        max_length=15,
        description="Tasks with COMPLETE solution code. Must have at least 1."
    )
    
    # === UNSOLVED TASKS (with specific blockers) ===
    unsolved_tasks: list[UnsolvedTask] = Field(
        default=[],
        max_length=15,
        description="Tasks with SPECIFIC blocker questions. Empty when all solved."
    )
    
    # === BLOCKER ANSWERS (accumulated knowledge) ===
    blocker_answers: list[str] = Field(
        default=[],
        max_length=20,
        description="""Answers to blocker questions from searching knowledge.
        Format: 'Q: Is power deferred? A: Yes, power is DEFERRED like +,-,*,/ because it needs two operands.'"""
    )
    
    # === STATE MACHINE ===
    calculator_states: list[str] = Field(
        ...,
        min_length=4,
        max_length=5,
        description="FSM states: 'idle', 'number_entry', 'operator_pending', 'result_displayed'"
    )
    
    # === DATA MODEL ===
    data_variables: list[str] = Field(
        ...,
        min_length=5,
        max_length=8,
        description="JS variables with types"
    )
    
    # === BEHAVIORAL TESTS ===
    behavioral_tests: list[str] = Field(
        ...,
        min_length=8,
        max_length=15,
        description="Test cases that must work"
    )
    
    # === SYNTHESIS ===
    design_summary: str = Field(
        ...,
        min_length=50,
        max_length=400,
        description="Overall approach"
    )


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

MAX_REFINEMENTS = 12
MIN_SOLVE_PER_ITERATION = 2
OUTPUT_ROOT = Path("output")

SYSTEM_PROMPT = """
You are a frontend engineer who solves problems systematically.

Your method:
1. For things you KNOW: WRITE THE CODE immediately
2. For things you DON'T KNOW: Ask SPECIFIC QUESTIONS, then SEARCH your knowledge to answer them

You are NEVER vague. You either:
- WRITE complete code (for solved tasks)
- Ask SPECIFIC questions (for unsolved tasks)

CALCULATOR KNOWLEDGE YOU CAN SEARCH:
- Transformers/LLMs know how calculators work
- You know JavaScript: Math.pow(), Math.sqrt(), parseFloat(), etc.
- You know state machines: states, transitions, flags
- You know event handling: onclick, updating display

When asked to SEARCH, recall what you know about calculators:
- Unary operations (sqrt, square): apply immediately to ONE number
- Binary operations (+,-,*,/,^): need TWO numbers, computed on equals
- State management: track what mode the calculator is in
- Display semantics: when to replace vs append digits
"""

TASK = """
Create a scientific calculator as a single, self-contained HTML file with embedded CSS and vanilla JavaScript (no external libraries).

=== LAYOUT REQUIREMENTS ===
- Rectangle shape, 350-400px wide × 500-550px tall, 3:4 aspect ratio
- Two-line display at top (expression history + current value)
- CSS Grid for buttons: uniform size, square, evenly spaced
- Must look like a physical handheld calculator
- Dark theme, responsive down to 320px width

=== BUTTONS REQUIRED ===
Numbers: 0-9, decimal point (.)
Operators: + − × ÷ = ( )
Clear: C (clear all), CE (clear entry), ⌫ (backspace)
Memory: MC, MR, M+, M-
Mode: DEG/RAD toggle
Scientific functions: π, 1/x, x², √x, xʸ, ln, log, eˣ, 10ˣ, sin, cos, tan, sin⁻¹, cos⁻¹, tan⁻¹

=== FUNCTION BEHAVIOR (CRITICAL) ===
UNARY FUNCTIONS (apply immediately when clicked):
- 1/x, x², √x, ln, log, eˣ, 10ˣ, sin, cos, tan, sin⁻¹, cos⁻¹, tan⁻¹
- Take the current displayed number, calculate result, show result immediately
- Do NOT insert function names into the display

BINARY OPERATORS (+, −, ×, ÷):
- Store first operand and operator
- Wait for second operand
- Calculate only when = is pressed or another operator is pressed
- Never evaluate incomplete expressions (e.g., "9÷" should wait, not error)

POWER FUNCTION (xʸ):
- Click xʸ: store current display value as base, show "^" in expression
- User enters exponent
- Press = to calculate Math.pow(base, exponent)

PI BUTTON (π):
- Insert the numeric value 3.141592653589793
- Do NOT insert the symbol "π"

=== COMPUTATION ENGINE ===
- Do NOT use eval() or new Function()
- Implement proper operator precedence: parentheses > exponents > multiplication/division > addition/subtraction
- Support chained operations (e.g., 2 + 3 × 4 = 14)
- DEG mode: convert degrees to radians before trig functions, convert radians to degrees for inverse trig results
- RAD mode: use radians directly

=== ERROR HANDLING (exact messages) ===
- Division by zero → "Error: Division by zero"
- √ of negative number → "Error: Invalid input"
- log or ln of zero or negative → "Error: Math domain"
- tan(90°) in DEG mode or tan(π/2) in RAD mode → "Error: Undefined" (check if |cos(x)| < 1e-10)
- asin or acos of values outside [-1, 1] → "Error: Invalid input"
- Result is NaN or Infinity → "Error: Overflow"
- After error: lock calculator until C or CE is pressed

=== MEMORY FUNCTIONS ===
- MC: Clear memory (set to 0)
- MR: Recall memory value to display
- M+: Add current display value to memory
- M-: Subtract current display value from memory

=== KEYBOARD SUPPORT ===
- 0-9, . for number entry
- +, -, *, / for operators
- Enter or = for equals
- Escape for C (clear all)
- Backspace for CE or delete last character

=== OUTPUT ===
Provide ONLY the complete HTML code, ready to save and run keeping the LAYOUT REQUIREMENTS.
No explanations before or after the code.
"""

STATE_CONFIG = {
    "maxTokens": 8192,
    "temperature": 0.2,
}

CODE_CONFIG = {
    "maxTokens": 32768,
    "temperature": 0.1,
}


# ---------------------------------------------------------------------------
# Helpers - FRESH INSTANCES
# ---------------------------------------------------------------------------
model = lms.llm()          # current loaded model
info = model.get_info()
print(f"\n Model info: {info} \n")
identifier = info.identifier       # internal identifier string
model_key  = info.model_key         # the key you pass to lms.llm(...)
print(f"\n Model indentity : {identifier}, and model key: {model_key}\n")

def get_fresh_model():
    """
    Get a fresh model handle for each call.
    This ensures no context leakage between phases.
    """
    
    return lms.llm()


def make_fresh_chat(system_prompt: str) -> lms.Chat:
    """
    Create a fresh Chat instance with no history.
    This is the key to avoiding context rot.
    """
    return lms.Chat(system_prompt)


def extract(result):
    return result.content if hasattr(result, "content") else str(result)


def clean(code: str):
    code = re.sub(r'<think>.*?</think>', '', code, flags=re.DOTALL)
    code = re.sub(r'^```html?\s*', '', code)
    code = re.sub(r'```$', '', code)
    return code.strip()


def all_tasks_solved(state: ReasoningState) -> bool:
    """Check if all tasks are solved."""
    return len(state.solved_tasks) >= 8 and len(state.unsolved_tasks) == 0


def get_all_blockers(state: ReasoningState) -> list[str]:
    """Get all blocker questions from unsolved tasks."""
    blockers = []
    for task in state.unsolved_tasks:
        for blocker in task.blockers:
            blockers.append(f"[{task.name}] {blocker}")
    return blockers


# ---------------------------------------------------------------------------
# Phase 1: DECOMPOSE & INITIAL SOLVE (Fresh Context)
# ---------------------------------------------------------------------------

def init_state() -> ReasoningState:
    """
    Initialize the reasoning state.
    Uses a FRESH model handle and FRESH chat context.
    """
    print("    → Creating fresh LLM context...")
    model = get_fresh_model()
    chat = make_fresh_chat(SYSTEM_PROMPT)
    
    chat.add_user_message(f"""
PHASE 1 — DECOMPOSE AND WRITE WHAT YOU KNOW

Task:
{TASK}

=== YOUR MISSION ===

You MUST categorize ALL 12 required tasks into either solved_tasks or unsolved_tasks.
Do NOT return empty lists - every task must go somewhere!

REQUIRED TASKS (all 12 must appear in either solved or unsolved):
1. display_management
2. digit_input_idle  
3. digit_after_operator
4. digit_after_result
5. operator_handling
6. equals_computation
7. power_function
8. sqrt_function
9. square_function
10. decimal_handling
11. clear_function
12. error_handling

=== FOR TASKS YOU KNOW HOW TO SOLVE → Add to solved_tasks ===

Include COMPLETE CODE (not descriptions):

{{
  "name": "display_management",
  "solution_code": "function updateDisplay() {{ document.getElementById('display').textContent = displayValue; }}"
}}

{{
  "name": "sqrt_function",
  "solution_code": "function computeSqrt() {{ let val = parseFloat(displayValue); if (val < 0) {{ displayValue = 'Error'; }} else {{ displayValue = String(Math.sqrt(val)); }} currentState = 'result_displayed'; updateDisplay(); }}"
}}

{{
  "name": "clear_function",
  "solution_code": "function clearCalculator() {{ displayValue = '0'; operandA = null; operandB = null; pendingOperator = null; currentState = 'idle'; shouldReplaceDisplay = false; updateDisplay(); }}"
}}

=== FOR TASKS YOU DON'T FULLY KNOW → Add to unsolved_tasks ===

Include SPECIFIC QUESTIONS (not vague concerns):

{{
  "name": "digit_after_operator",
  "what_i_know": "Need to handle digits when an operator is pending",
  "blockers": [
    "Should the first digit REPLACE the display or APPEND to it?",
    "What flag controls replace vs append behavior?",
    "Does the state change from operator_pending when entering digits?"
  ]
}}

{{
  "name": "power_function",
  "what_i_know": "Need to compute x raised to power y",
  "blockers": [
    "Is power immediate like sqrt, or deferred like addition?",
    "What JavaScript function computes power?",
    "Should power button call inputOperator or a special handler?"
  ]
}}

=== ALSO REQUIRED ===

- calculator_states: ["idle", "number_entry", "operator_pending", "result_displayed"]
- data_variables: All 6 required variables with types
- behavioral_tests: At least 8 test cases including power function tests
- design_summary: Your overall approach (50+ characters)

=== VALIDATION ===

Before returning, verify:
- solved_tasks has at least 1 task (ideally 4-8 that you're confident about)
- unsolved_tasks has the tasks you need help with
- Total tasks (solved + unsolved) should be around 12
- Every solved task has actual CODE in solution_code
- Every unsolved task has specific QUESTIONS in blockers

Return valid JSON.
""")
    
    out = model.respond(chat, response_format=ReasoningState, config=STATE_CONFIG)
    return ReasoningState(**out.parsed)


# ---------------------------------------------------------------------------
# Phase 2: SEARCH & ANSWER BLOCKERS (Fresh Context Each Iteration)
# ---------------------------------------------------------------------------

def refine_state(prev: ReasoningState, iteration: int) -> ReasoningState:
    """
    Refine the reasoning state by answering blockers.
    Uses a FRESH model handle and FRESH chat context.
    The ONLY input from previous iteration is the ReasoningState JSON.
    """
    print("    → Creating fresh LLM context...")
    model = get_fresh_model()
    chat = make_fresh_chat(SYSTEM_PROMPT)
    
    # Get all blockers that need answers
    all_blockers = get_all_blockers(prev)
    
    # Calculate required progress
    current_solved = len(prev.solved_tasks)
    current_unsolved = len(prev.unsolved_tasks)
    min_to_solve = min(MIN_SOLVE_PER_ITERATION, current_unsolved)
    required_solved = current_solved + min_to_solve
    
    # Format unsolved tasks for the prompt
    unsolved_details = ""
    for task in prev.unsolved_tasks:
        unsolved_details += f"\n\n### {task.name}\n"
        unsolved_details += f"What I know: {task.what_i_know}\n"
        unsolved_details += "Blockers (QUESTIONS TO ANSWER):\n"
        for b in task.blockers:
            unsolved_details += f"  - {b}\n"
    
    chat.add_user_message(f"""
PHASE 2 — SEARCH YOUR KNOWLEDGE AND ANSWER BLOCKERS (Iteration {iteration})

Task:
{TASK}

=== CURRENT STATE ===
Solved tasks: {current_solved}
Unsolved tasks: {current_unsolved}

=== UNSOLVED TASKS WITH BLOCKERS ===
{unsolved_details}

=== YOUR MISSION: SEARCH AND ANSWER ===

For EACH blocker question above, SEARCH your knowledge and ANSWER it.

Example:
  BLOCKER: "Is power (^) immediate like sqrt, or deferred like +?"
  
  SEARCH: I know that:
  - Immediate operations (sqrt, square) take ONE operand and compute immediately
  - Deferred operations (+,-,*,/) take TWO operands and compute on equals
  - Power x^y takes TWO operands (base and exponent)
  
  ANSWER: "Power is DEFERRED because it needs two operands. It works like +,-,*,/."

Add your answers to blocker_answers in format:
"Q: [question] A: [answer]"

=== THEN SOLVE THE TASKS ===

After answering blockers, you should be able to WRITE THE CODE.

Move tasks from unsolved_tasks to solved_tasks with COMPLETE code.

Example - converting unsolved to solved:

BEFORE (unsolved):
{{
  "name": "power_function",
  "blockers": ["Is power deferred?", "What JS function?"]
}}

AFTER (solved):
{{
  "name": "power_function", 
  "solution_code": "// Power is DEFERRED - use inputOperator handler\\nfunction handlePower() {{ inputOperator('^'); }}\\n// In computeResult:\\ncase '^': result = Math.pow(operandA, operandB); break;"
}}

=== MANDATORY PROGRESS ===

Current solved: {current_solved}
REQUIRED solved after this iteration: {required_solved} (must solve at least {min_to_solve} more)

If you don't make progress, we cannot generate code.

TIPS FOR COMMON BLOCKERS:

**digit_after_operator:**
- First digit REPLACES display (starts operandB)
- Use shouldReplaceDisplay flag: if true, replace; if false, append
- After first digit, set shouldReplaceDisplay = false
- State STAYS 'operator_pending' while entering operandB

**power_function:**
- Power is DEFERRED (binary operation)
- Button: onclick="inputOperator('^')"
- Computation: case '^': result = Math.pow(operandA, operandB); break;
- Test: 2^3=8, 2^10=1024, 4^0.5=2

**digit_after_result:**
- After = shows result, digit starts COMPLETELY NEW number
- Operator continues from result (result becomes operandA)
- Set currentState = 'number_entry', displayValue = digit

=== RETURN ===

Updated ReasoningState with:
1. blocker_answers: Your Q&A from searching
2. solved_tasks: Moved tasks here WITH CODE
3. unsolved_tasks: Only tasks still blocked (should be fewer!)

Return valid JSON.
""")
    
    out = model.respond(chat, response_format=ReasoningState, config=STATE_CONFIG)
    return ReasoningState(**out.parsed)


# ---------------------------------------------------------------------------
# Phase 3: GENERATE CODE (Fresh Context)
# ---------------------------------------------------------------------------

def generate_code(state: ReasoningState) -> str:
    """
    Generate final HTML code from solved tasks.
    Uses a FRESH model handle and FRESH chat context.
    """
    print("    → Creating fresh LLM context...")
    model = get_fresh_model()
    chat = make_fresh_chat(SYSTEM_PROMPT)
    
    # Compile all solution code
    solutions = "\n\n".join([
        f"// === {t.name} ===\n{t.solution_code}"
        for t in state.solved_tasks
    ])
    
    # Compile blocker answers as knowledge
    knowledge = "\n".join(state.blocker_answers) if state.blocker_answers else "None"
    
    chat.add_user_message(f"""
PHASE 3 — FINAL CODE GENERATION

Task:
{TASK}

=== SOLVED TASKS (Your solutions) ===

{solutions}

=== KNOWLEDGE FROM SEARCHING ===

{knowledge}

=== NOW GENERATE THE COMPLETE CALCULATOR ===

Combine all your solutions into a single HTML file.

Structure:
1. <!DOCTYPE html>
2. <html><head><style>CSS here</style></head>
3. <body>
   - Display div
   - Button grid
4. <script>
   - Variable declarations
   - All functions from your solutions
   - updateDisplay function
5. </script></body></html>

=== MANDATORY CODE STRUCTURE ===

Variables:
```
let currentState = 'idle';
let displayValue = '0';
let operandA = null;
let operandB = null;
let pendingOperator = null;
let shouldReplaceDisplay = false;
```

Button handlers (onclick ONLY, no addEventListener):
- Digits: onclick="inputDigit('5')"
- Operators: onclick="inputOperator('+')"
- Power: onclick="inputOperator('^')"  // DEFERRED!
- Sqrt: onclick="computeImmediate('sqrt')"
- Square: onclick="computeImmediate('square')"
- Equals: onclick="handleEquals()"
- Clear: onclick="clearCalculator()"

=== TEST CASES THAT MUST WORK ===

1. "2 + 3 =" → 5
2. "2 ^ 3 =" → 8
3. "2 ^ 10 =" → 1024
4. "4 ^ 0.5 =" → 2
5. "9 sqrt" → 3
6. "4 x²" → 16
7. "3.1 + 2.5 =" → 5.6
8. "5 + 3 = + 2 =" → 10
9. "2 ^ 3 = ^ 2 =" → 64
10. "5 + 3 = 7" → 7

Output ONLY the complete HTML file.
No markdown, no explanation, no code fences.
""")
    
    out = model.respond(chat, config=CODE_CONFIG)
    return clean(extract(out))


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main():
    # ----------------------------------------
    # Output + timestamped run folder
    # ----------------------------------------
    OUTPUT_ROOT.mkdir(exist_ok=True)
    run_ts = datetime.now().strftime("run_%Y%m%d_%H%M%S")
    run_dir = OUTPUT_ROOT / run_ts
    run_dir.mkdir()

    print(f"📁 Output directory: {run_dir}")
    print(f"🔄 Context isolation: FRESH LLM context for each phase")
    print(f"💾 State persistence: Only ReasoningState JSON carries between phases")
    print()

    # Save a copy of this script
    script_path = Path(__file__).resolve()
    shutil.copy(script_path, run_dir / script_path.name)

    # ----------------------------------------
    # Run metadata
    # ----------------------------------------
    metadata = {
        "timestamp": run_ts,
        "run_dir": str(run_dir.resolve()),
        "script_name": script_path.name,
        "max_refinements": MAX_REFINEMENTS,
        "min_solve_per_iteration": MIN_SOLVE_PER_ITERATION,
        "task": TASK.strip(),
        "state_config": STATE_CONFIG,
        "code_config": CODE_CONFIG,
        "context_isolation": "fresh_per_call",
        "model identifier": model_key,
    }
    (run_dir / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )

    # ----------------------------------------
    # Phase 1 — DECOMPOSE (Fresh Context)
    # ----------------------------------------
    print("[1] Decomposing: WRITE what you know, ASK about what you don't...")
    state = init_state()
    (run_dir / "state_init.json").write_text(
        state.model_dump_json(indent=2),
        encoding="utf-8",
    )
    
    # Validate we got actual tasks
    total_tasks = len(state.solved_tasks) + len(state.unsolved_tasks)
    if total_tasks < 5:
        print(f"    ⚠️ WARNING: Only {total_tasks} tasks defined! Expected ~12.")
        print(f"    ⚠️ Model may not have understood the prompt.")
    
    print(f"    ✓ Solved tasks: {len(state.solved_tasks)}")
    print(f"    ✓ Unsolved tasks: {len(state.unsolved_tasks)}")
    print(f"    ✓ Blockers to answer: {len(get_all_blockers(state))}")
    
    if state.solved_tasks:
        print(f"    ✓ Already solved: {[t.name for t in state.solved_tasks]}")
    if state.unsolved_tasks:
        print(f"    → Need to solve: {[t.name for t in state.unsolved_tasks]}")

    # ----------------------------------------
    # Phase 2 — SEARCH (Fresh Context Each Iteration)
    # ----------------------------------------
    iteration = 0
    while iteration < MAX_REFINEMENTS:
        iteration += 1
        
        # Check if done
        if all_tasks_solved(state):
            print(f"\n🎉 All tasks solved after {iteration - 1} refinement(s)!")
            break
        
        blockers = get_all_blockers(state)
        print(f"\n[2.{iteration}] SEARCHING to answer {len(blockers)} blockers (fresh context)...")
        
        prev_solved = len(state.solved_tasks)
        prev_unsolved = len(state.unsolved_tasks)
        
        state = refine_state(state, iteration)
        
        (run_dir / f"state_refine_{iteration}.json").write_text(
            state.model_dump_json(indent=2),
            encoding="utf-8",
        )
        
        # Show progress
        newly_solved = len(state.solved_tasks) - prev_solved
        newly_answered = len(state.blocker_answers)
        
        print(f"    ✓ Blocker answers accumulated: {newly_answered}")
        print(f"    ✓ Solved: {len(state.solved_tasks)} (+{newly_solved})")
        print(f"    ✓ Unsolved: {len(state.unsolved_tasks)}")
        
        if newly_solved > 0:
            print(f"    ✓ Newly solved: {[t.name for t in state.solved_tasks][-newly_solved:]}")
        
        if newly_solved == 0:
            print(f"    ⚠️ No progress this iteration!")
    
    else:
        # Hit max iterations
        if not all_tasks_solved(state):
            print(f"\n⚠️ Reached max refinements ({MAX_REFINEMENTS}).")
            print(f"    Still unsolved: {[t.name for t in state.unsolved_tasks]}")

    # ----------------------------------------
    # Save final state
    # ----------------------------------------
    (run_dir / "state_final.json").write_text(
        state.model_dump_json(indent=2),
        encoding="utf-8",
    )

    # ----------------------------------------
    # Phase 3 — GENERATE (Fresh Context)
    # ----------------------------------------
    print("\n[3] Generating code from solved tasks (fresh context)...")
    html = generate_code(state)

    final_path = run_dir / "calculator.html"
    final_path.write_text(html, encoding="utf-8")
    print(f"    ✓ HTML saved to: {final_path}")
    print(f"    ✓ Lines: {len(html.splitlines())}")

    # ----------------------------------------
    # Summary
    # ----------------------------------------
    print("\n" + "="*60)
    print("DONE")
    print("="*60)
    print(f"Output directory: {run_dir}")
    print(f"Context isolation: ✓ Fresh LLM context for each phase")
    print(f"Files generated:")
    for f in sorted(run_dir.iterdir()):
        print(f"  - {f.name}")
    
    # Show solved tasks with code preview
    print(f"\n✅ Solved Tasks ({len(state.solved_tasks)}):")
    for task in state.solved_tasks:
        code_preview = task.solution_code[:60].replace('\n', ' ')
        print(f"  • {task.name}")
        print(f"    Code: {code_preview}...")
    
    # Show any remaining unsolved
    if state.unsolved_tasks:
        print(f"\n⚠️ Unsolved Tasks ({len(state.unsolved_tasks)}):")
        for task in state.unsolved_tasks:
            print(f"  • {task.name}: {task.blockers}")
    
    # Show blocker answers (knowledge accumulated)
    if state.blocker_answers:
        print(f"\n📚 Knowledge Accumulated ({len(state.blocker_answers)}):")
        for answer in state.blocker_answers[:5]:
            print(f"  • {answer[:80]}...")
    
    # Show behavioral tests
    print(f"\n🧪 Behavioral tests ({len(state.behavioral_tests)}):")
    for test in state.behavioral_tests:
        print(f"  • {test}")


if __name__ == "__main__":
    main()
