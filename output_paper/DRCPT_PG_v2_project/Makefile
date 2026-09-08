PDF=main.pdf
TEX=main.tex

all: $(PDF)

$(PDF): $(TEX) references.bib
	pdflatex -interaction=nonstopmode main.tex
	bibtex8 main
	pdflatex -interaction=nonstopmode main.tex
	pdflatex -interaction=nonstopmode main.tex

clean:
	rm -f main.aux main.bbl main.blg main.log main.out main.toc main.fls main.fdb_latexmk

.PHONY: all clean
