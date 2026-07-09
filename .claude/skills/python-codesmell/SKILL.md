---
name: python-codesmell
description: Scan Python code for code smells and technical debt (long/complex functions, duplicated logic, mutable default arguments, broad excepts, dead code, naming issues, missing type hints, global mutable state, TODO/FIXME debt markers). Use when the user asks to find code smells, audit Python code quality, look for technical debt, or review a Python file/module/repo for maintainability, outside of the diff-scoped crev/code-review workflow.
---

# Python Code Smell Finder

Standalone Python code-quality scan. Reports findings directly in chat.
Not a diff review and not an auto-fixer.

## Scope & non-goals

- Scans whatever the user names: a file, a directory, or the whole
  repo's `*.py`. Excludes `tests/` by default (fixtures and assertions
  read differently from production code and generate noise) unless the
  user asks for tests to be included.
- Not a diff review. For reviewing pending changes against acceptance
  criteria, defer to `/dyalog:crev` or the built-in `code-review` skill.
- Not an auto-fixer. For applying simplification fixes to changed code,
  defer to the built-in `simplify` skill. This skill only changes code
  if the user explicitly asks after seeing the report.
- Never installs a linter or edits `requirements.txt`/`pyproject.toml`
  to add one.
- Never writes the report to `docs/` or any other file. Output goes
  directly in the conversation.

## Workflow

1. Resolve scope from the user's request; default to non-test `*.py`
   files if unspecified.
2. Run the AST structural pass (mutable defaults, broad/bare excepts,
   long functions, long parameter lists, nesting/branch complexity).
3. Run the grep heuristic pass (debt markers, narrative and apologetic
   comments, naming, global mutable state, unused-import candidates).
4. Manually confirm the fuzzy findings before reporting: cross-check
   candidate-unused symbols with `rg -n '\bNAME\b'` across the repo, and
   compare candidate duplicate functions side by side. Drop anything not
   genuinely confirmed rather than reporting speculative noise.
5. If `ruff`, `radon`, or `vulture` happen to already be on `PATH`, run
   them read-only to corroborate findings. Skip silently if absent; do
   not suggest installing them.
6. Assign severity per the calibration below, de-duplicate overlapping
   findings, and produce the report using the Output Format.
7. Only apply fixes if the user explicitly asks after seeing the
   report, then re-run the relevant check to confirm resolution.

## Smell checklist

**Critical** (correctness-adjacent):
- Mutable default arguments (`def f(x=[])`, `def f(x={})`): the
  container is created once at function-definition time and shared
  across every call that omits the argument.
- Bare `except:` or `except Exception:` whose body silently swallows
  the error (a no-op `pass` or equivalent).

**Major** (structural):
- Long functions (>40 logical lines, `end_lineno - lineno`).
- God classes: an outsized method count or clearly unrelated
  responsibilities bundled together.
- Deep nesting (more than 3-4 levels of `if`/`for`/`while`/`try`).
- Long parameter lists (>5 positional/keyword parameters).
- High branch count (>10 `if`/`for`/`while`/`try`/`BoolOp` nodes in one
  function) as a proxy for cyclomatic complexity.
- Duplicated logic: near-identical function or method bodies differing
  only in names or literals. No dependency-free duplicate-code detector
  exists in the stdlib, so this is a manual side-by-side comparison, not
  a mechanical check.
- Global mutable state: a module-level container or singleton instance
  that is mutated from multiple functions, or reached into directly by
  callers/tests instead of being passed as a dependency.

**Minor** (readability):
- Non-descriptive names (`tmp`, `data`, `obj`, `foo`, `val`, `res`) used
  pervasively rather than locally and briefly.
- Inconsistent naming convention (mixed `camelCase`/`snake_case` in the
  same module).
- Missing type hints on public function signatures.
- Dead code: unused imports, unreachable code after `return`/`raise`,
  commented-out blocks. Confirm with a cross-file `rg` reference check
  before flagging: framework hooks, `__all__` re-exports, and
  dynamically dispatched methods produce false positives.

**Note** (technical debt markers):
- `TODO`/`FIXME`/`HACK`/`XXX` comments, especially undated or lacking
  actionable context.
- Narrative comments explaining implementation history ("we tried",
  "originally", "used to", "for now", "first attempt") rather than
  stating the current invariant.
- Apologetic hedging ("this is a bit hacky but…", "not sure if this is
  right…").
- Inconsistent error-handling style across sibling code paths: one
  raises, another returns `None`, another logs and swallows.

## Commands

Structural AST pass:

```bash
python3 - "$@" <<'PY'
import ast, sys

THRESH_LINES = 40
THRESH_PARAMS = 5
THRESH_BRANCHES = 10

def branch_count(node):
    return sum(
        1 for n in ast.walk(node)
        if isinstance(n, (ast.If, ast.For, ast.While, ast.Try, ast.BoolOp))
    )

for filename in sys.argv[1:]:
    tree = ast.parse(open(filename).read(), filename=filename)
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            length = (node.end_lineno or node.lineno) - node.lineno
            nparams = len(node.args.args) + len(node.args.kwonlyargs)
            branches = branch_count(node)
            if length > THRESH_LINES:
                print(f"{filename}:{node.lineno} LONG_FUNCTION {node.name} ({length} lines)")
            if nparams > THRESH_PARAMS:
                print(f"{filename}:{node.lineno} LONG_PARAM_LIST {node.name} ({nparams} params)")
            if branches > THRESH_BRANCHES:
                print(f"{filename}:{node.lineno} HIGH_COMPLEXITY {node.name} (~{branches} branches)")
            for default in node.args.defaults:
                if isinstance(default, (ast.List, ast.Dict, ast.Set)):
                    print(f"{filename}:{node.lineno} MUTABLE_DEFAULT_ARG {node.name}")
        if isinstance(node, ast.ExceptHandler):
            if node.type is None or (isinstance(node.type, ast.Name) and node.type.id == "Exception"):
                body_is_pass = len(node.body) == 1 and isinstance(node.body[0], ast.Pass)
                tag = "BARE_EXCEPT_SWALLOWED" if body_is_pass else "BROAD_EXCEPT"
                print(f"{filename}:{node.lineno} {tag}")
PY
```

Debt-marker and naming heuristics:

```bash
rg -n -i 'TODO|FIXME|HACK|XXX' --glob '*.py'
rg -n -i 'we (tried|used to|originally|first)|previously used|used to|for now' --glob '*.py'
rg -n '\b(tmp|temp|data|obj|foo|bar|baz|val|res)\b\s*=' --glob '*.py'
rg -n '^\w+\s*=\s*(\{|\[|list\(|dict\(|set\()' --glob '*.py'   # module-level mutable state
rg -n '\bglobal\b' --glob '*.py'
```

Opportunistic tool corroboration, only if already installed:

```bash
command -v ruff >/dev/null && ruff check <scope>
command -v radon >/dev/null && radon cc <scope> -s -a
command -v vulture >/dev/null && vulture <scope>
```

## Output format

Post the report directly in chat, in this shape:

```markdown
## Code Smell Report: <scope>

### Summary
<n> findings across <m> files: <x> Critical, <y> Major, <z> Minor, <w> Note

### Findings

#### [Critical] MUTABLE_DEFAULT_ARG — shared-state bug risk
- Location: path/to/file.py:42
- Smell: mutable default argument on `def add_item(items=[])`
- Why it matters: the list is created once at function-definition time
  and shared across every call that omits the argument.
- Suggested fix: default to `None` and create the list inside the
  function body.

...

### Not flagged
<Patterns considered and judged acceptable, so the report doesn't read
as exhaustive-but-shallow>
```

Severity calibration:
- Critical: real correctness/security risk (mutable defaults, silently
  swallowed exceptions).
- Major: meaningful maintainability cost (god functions/classes, high
  complexity, duplication, global mutable state, deep nesting).
- Minor: readability (naming, missing type hints, borderline-long
  parameter lists).
- Note: tracked but not urgent (TODO/FIXME markers, narrative comments,
  unconfirmed dead-code candidates).

## Working style

- Ground every structural claim in the AST script's actual output;
  never assert "too long" without the line count it produced.
- Confirm dead-code and duplication suspicions with a cross-file
  reference check before reporting. False positives on framework hooks
  erode trust fast.
- Default to fewer, high-confidence findings; only go broad and
  low-confidence if the user explicitly asks for a thorough or deep
  scan.
- Write the report prose to this repo's conventions: invariants over
  narrative in "why it matters", no emojis, no em-dashes, UK spelling,
  no apologetic hedging, and never cite a GitHub issue ID in a
  suggested code comment.
- Never write the report to a file. Persisting it is a distinct,
  explicit request, not this skill's default.
- Stay out of `code-review`/`crev`/`simplify` territory: this skill's
  job is a standalone Python-smell vocabulary scan, not a diff review or
  an auto-fix pass.

## Examples

User asks: "Can you check `app/store.py` and `app/main.py` for code
smells?"
- Scope is the two files. The AST pass returns nothing: no long
  functions, no mutable defaults, no broad excepts.
- The debt-marker grep returns nothing: no TODO/FIXME or narrative
  comments.
- Surface the one real finding even though no mechanical check flagged
  it: `app/main.py` instantiates `store = UserStore()` at module scope,
  and `tests/conftest.py` reaches into `app.main.store` directly to
  reset it between tests, a Major global-mutable-state coupling.
  Suggested fix: inject the store via FastAPI `Depends` instead of
  importing the module-level instance.
- Report: 1 Major finding, 0 Critical, 0 Minor, 0 Note.

User asks: "Do a deep pass over the whole app for technical debt."
- Scope is all non-test `*.py`. Run all three passes plus the
  opportunistic tool check; report that none of `ruff`/`radon`/`vulture`
  are installed and proceed on the heuristic passes alone.
- Present the full report grouped by severity, ending with a "Not
  flagged" section noting deliberately acceptable patterns (for
  example, a `Lock`-guarded store method is fine and not over-engineering).
