#!/usr/bin/env bash
set -euo pipefail

latexmk -pdf -interaction=nonstopmode main.tex
