# NeuroTensor LaTeX technical report

This directory contains the modular LaTeX source for the comprehensive project
report. The report is a reproducibility-oriented record through 2026-09-06.

## Build

From this directory, run the direct MiKTeX sequence (this avoids the optional
Perl dependency used by `latexmk`):

```powershell
pdflatex --enable-installer -interaction=nonstopmode -halt-on-error main.tex
bibtex main
pdflatex --enable-installer -interaction=nonstopmode -halt-on-error main.tex
pdflatex --enable-installer -interaction=nonstopmode -halt-on-error main.tex
```

The compiled report in this repository is [`main.pdf`](main.pdf).

## Structure

The revised edition uses embedded NewTX Times-style text and matching maths,
compact continuous chapter headings, and native-size multi-row diagrams.
MiKTeX may download the required LaTeX font/style packages on the first build.

- `main.tex`: document entry point.
- `preamble.tex`: style, colors, reusable commands, and diagram defaults.
- `frontmatter/`: title, scope declaration, and executive summary.
- `sections/`: main report chapters.
- `figures/`: native TikZ/PGFPlots figures.
- `appendices/`: code, hyperparameter, artifact, and terminology references.
- `references.bib`: bibliography.

The original BIDS data and all versioned experiment artifacts remain outside
this report directory and are not modified by the report build.
