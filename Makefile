.PHONY: create-env
create-env: ## Creates conda environment
	conda env create -f environment.yml --force

.PHONY: setup-env
setup-env: ## Sets up conda environment
	conda install pytorch torchvision numpy -c pytorch -y
	pip install gym-retro
	pip install "gym[atari]==0.21.0"
	pip install importlib-metadata==4.13.0

.PHONY: run
run: ## Runs
	python ./src/airstriker-genesis/run.py
