# Examples

## basic_extraction.py

Extract questions and figures from a Cambridge A-Level exam paper (9709 Paper 1, June 2022).

```bash
export MATH_OCR_API_KEY="your-key"
python examples/basic_extraction.py
```

Expected output:
```
  p0: 3  The coefficient of \(x^4\) in the expansion of \(\left(2x^2 + \frac{k^2}{x}\right)^5\)...
  p4: 5  [FIGURE] sector_diagram.png (857x482)
  p4:  The diagram shows a sector \(ABC\) of a circle with centre \(A\)...
  p8: 7  [FIGURE] curve_line_intersection.png (833x548)
  p8:  (a)  Find the coordinates of \(A\) and \(B\). [4]

Extracted 6 items
```

See [`example_output.json`](example_output.json) for the full structured output, and [`figures/`](figures/) for the extracted PNG + SVG images.

## marking_scheme.py

Extract answers and mark allocations from a marking scheme, with custom rules.

```bash
python examples/marking_scheme.py
```
