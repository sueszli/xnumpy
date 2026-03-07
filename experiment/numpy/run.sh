#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

echo "compiling report.Rmd -> report.pdf"
Rscript -e 'rmarkdown::render("report.Rmd", quiet = TRUE)'
echo "done: report.pdf"
