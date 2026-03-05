TEX=src/main.tex
OUTDIR=build

all:
	latexmk -pdf -interaction=nonstopmode -halt-on-error -outdir=$(OUTDIR) $(TEX)

clean:
	latexmk -C -outdir=$(OUTDIR)

blind: all
	cp $(OUTDIR)/main.pdf submissions/NiDS2026_blind.pdf