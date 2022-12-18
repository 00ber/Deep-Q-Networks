.PHONY: create-atari-env
create-atari-env: ## Creates conda environment
	conda env create -f environment.atari-yml --force

.PHONY: create-procgen-env
create-procgen-env: ## Creates conda environment
	conda env create -f environment.procgen.yml --force

.PHONY: setup-env
setup-env: ## Sets up conda environment
	conda install pytorch torchvision numpy -c pytorch -y
	pip install gym-retro
	pip install "gym[atari]==0.21.0"
	pip install importlib-metadata==4.13.0

.PHONY: run-air-dqn
run-air-dqn: ## Runs
	python ./src/airstriker-genesis/run-airstriker-dqn.py

.PHONY: run-air-ddqn
run-air-ddqn: ## Runs
	python ./src/airstriker-genesis/run-airstriker-ddqn.py

.PHONY: run-starpilot-dqn
run-starpilot-dqn: ## Runs
	python ./src/procgen/run-starpilot-dqn.py

.PHONY: run-starpilot-ddqn
run-starpilot-ddqn: ## Runs
	python ./src/procgen/run-starpilot-ddqn.py
