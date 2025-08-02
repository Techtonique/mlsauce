.PHONY: clean clean-test clean-pyc clean-build docs help
.DEFAULT_GOAL := help

define BROWSER_PYSCRIPT
import os, webbrowser, sys, mkdocs

from urllib.request import pathname2url

webbrowser.open("file://" + pathname2url(os.path.abspath(sys.argv[1])))
endef
export BROWSER_PYSCRIPT

define PRINT_HELP_PYSCRIPT
import re, sys

for line in sys.stdin:
	match = re.match(r'^([a-zA-Z_-]+):.*?## (.*)$$', line)
	if match:
		target, help = match.groups()
		print("%-20s %s" % (target, help))
endef
export PRINT_HELP_PYSCRIPT

BROWSER := python3 -c "$$BROWSER_PYSCRIPT"

help:
	@python3 -c "$$PRINT_HELP_PYSCRIPT" < $(MAKEFILE_LIST)

clean: clean-build clean-pyc clean-test ## remove all build, test, coverage and Python artifacts

clean-build: ## remove build artifacts
	rm -fr build/	
	rm -fr .eggs/
	find . -name '*.egg-info' -exec rm -fr {} +
	find . -name '*.egg' -exec rm -rf {} +
	find . -name '*.so' -exec rm -f {} +
	find . -name '*.c' -exec rm -f {} +
	find . -name '*.html' -exec rm -f {} +	

clean-pyc: ## remove Python file artifacts
	find . -name '*.pyc' -exec rm -f {} +
	find . -name '*.pyo' -exec rm -f {} +
	find . -name '*~' -exec rm -f {} +
	find . -name '__pycache__' -exec rm -fr {} +

clean-test: ## remove test and coverage artifacts
	rm -fr .tox/
	rm -f .coverage
	rm -fr htmlcov/
	rm -fr .pytest_cache

lint: ## check style with flake8
	flake8 mlsauce tests

test: ## run tests quickly with the default Python
	python setup.py test

test-all: ## run tests on every Python version with tox
	tox

coverage: ## check code coverage quickly with the default Python
	coverage run --source mlsauce setup.py test
	coverage report -m
	coverage html
	$(BROWSER) htmlcov/index.html

docs: install ## compile the docs watching for change	 	
	pip install black 
	pip install pdoc
	black mlsauce/* --line-length=80	
	pdoc -t docs mlsauce/* --output-dir mlsauce-docs
	find . -name '__pycache__' -exec rm -fr {} +

servedocs: install ## compile the docs watching for change	 	
	pip install black 
	pip install pdoc
	black mlsauce/* --line-length=80	
	pdoc -t docs mlsauce/* 
	find . -name '__pycache__' -exec rm -fr {} +

build-site: docs ## export mkdocs website to a folder		
	cp -rf mlsauce-docs/* ../../Pro_Website/Techtonique.github.io/mlsauce
	find . -name '__pycache__' -exec rm -fr {} +

dist: clean ## builds source and wheel package
	python3 setup.py sdist
	python3 setup.py bdist_wheel
	ls -l dist

install: clean ## install the package to the active Python's site-packages
	uv pip install -e . --verbose

run-examples: ## run all examples with one command
	find examples -maxdepth 2 -name "*.py" -exec  python3 {} \;

run-booster: ## run all boosting estimators examples with one command
	find examples -maxdepth 2 -name "*boost*.py" -exec  python3 {} \;

run-lazy: ## run all lazy estimators examples with one command
	find examples -maxdepth 2 -name "*lazy*.py" -exec  python3 {} \;

docker-build: ## Build Docker image for mlsauce and create dist artifacts
	docker build -t mlsauce .
	docker run --rm -v $(PWD)/dist:/app/dist mlsauce sh -c "python3 setup.py sdist bdist_wheel"

docker-shell: ## Run an interactive shell inside the mlsauce Docker container
	docker run -it --rm mlsauce bash

docker-run-examples: ## Run all example scripts inside Docker
	docker run --rm mlsauce sh -c "pip install -e . && find examples -maxdepth 2 -name '*.py' -exec python3 {} \;"

docker-run-booster: ## Run boosting example scripts inside Docker
	docker run --rm mlsauce sh -c "pip install -e . && find examples -maxdepth 2 -name '*boost*.py' -exec python3 {} \;"

docker-run-lazy: ## Run lazy estimator example scripts inside Docker
	docker run --rm mlsauce sh -c "pip install -e . && find examples -maxdepth 2 -name '*lazy*.py' -exec python3 {} \;"

