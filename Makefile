lint:
	black --check src/levy_stable_jax tests
	ruff check src/levy_stable_jax tests
	mypy src/levy_stable_jax

test:
	PYTHONPATH=$(PWD) pytest tests src/levy_stable_jax -v --doctest-modules

generate:
	PYTHONPATH=$(PWD)/src python3 -m levy_stable_jax._generate_data

doc:
	PYTHONPATH=$(PWD)/src mkdocs serve
