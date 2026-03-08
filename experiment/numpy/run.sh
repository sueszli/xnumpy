#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
Rscript -e 'rmarkdown::render("report.Rmd", quiet = TRUE)'
