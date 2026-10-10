.PHONY: test

test:
	@set -e; for suite in pipeline pipeline/edit pipeline/h3 pipeline/open_media; do \
		PYTHONPATH=.:pipeline:pipeline/edit:pipeline/h3:pipeline/open_media \
		uv run --locked python -m unittest discover -s $$suite -p 'test_*.py'; \
	done
