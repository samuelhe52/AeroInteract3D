REPORT ?= final-tech-report.md
OUTDIR ?= .build

.PHONY: help report clean

help:
	@echo "Available targets:"
	@echo "  make report - Build REPORT into OUTDIR"
	@echo "  make clean  - Remove hidden report build artifacts"

report:
	bash ./build-pdf.sh "$(REPORT)" "$(OUTDIR)"

clean:
	rm -rf .build
