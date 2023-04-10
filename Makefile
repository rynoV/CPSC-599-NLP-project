build-pdf:
	latexmk -pvc -pdf -lualatex --synctex=1 -interaction=nonstopmode -output-directory=./.output $(FILE)
