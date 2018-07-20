TARGETS=thoughts
all: $(TARGETS)

$(TARGETS): %: html/%.html

html/%.html: %.md
	pandoc --self-contained -s -o $@ --highlight-style pygments --data-dir . --columns 1000 --css css/md.css -S --normalize --ascii --number-sections --mathjax="" $^

clean:
	rm -f html/*.html
