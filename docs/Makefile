# Minimal makefile for Sphinx documentation
#

# You can set these variables from the command line, and also
# from the environment for the first two.
SPHINXOPTS    ?=
SPHINXBUILD   ?= sphinx-build
SOURCEDIR     = source
BUILDDIR      = build

# Put it first so that "make" without argument is like "make help".
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

.PHONY: help Makefile

# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
%: Makefile
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

# Multi-language build rule
.PHONY: multi
multi:
	@$(SPHINXBUILD) -D language=en -b html "$(SOURCEDIR)" "$(BUILDDIR)/html/en"
	@$(SPHINXBUILD) -D language=ko -b html "$(SOURCEDIR)" "$(BUILDDIR)/html/ko"
	@echo "Multi-language build complete. Results in $(BUILDDIR)/html"
