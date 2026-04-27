# Examples

## basic_extraction.py

Extract questions and figures from an exam question paper.

```bash
export MATH_OCR_API_KEY="your-key"
python examples/basic_extraction.py
```

Expected output:
```
  p1: 1  Solve the inequality \(x^2 - 5x + 6 < 0\). [3]...
  p1: 2  The diagram shows the curve \(y = x^3 - 4x + 1\) intersecting... [FIGURE]
  p2: (a)  Find the coordinates of \(A\) and \(B\). [4]...

Extracted 3 items total
```

See [`example_output.json`](example_output.json) for the full structured output.

## marking_scheme.py

Extract answers and mark allocations from a marking scheme, with custom rules.

```bash
python examples/marking_scheme.py
```
