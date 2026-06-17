#!/bin/bash
# Compile ADAPT-Rehab paper for ATC 2026
# Requires: pdflatex, bibtex

echo "Compiling ADAPT-Rehab paper..."

# Clean previous builds
rm -f *.aux *.bbl *.blg *.log *.out *.toc sections/*.aux

# First pass
pdflatex main.tex

# Run bibtex for references
bibtex main

# Second pass (for references)
pdflatex main.tex

# Third pass (for cross-references)
pdflatex main.tex

echo "Done! Output: main.pdf"
echo ""
echo "To view: xdg-open main.pdf (Linux) or open main.pdf (Mac)"
