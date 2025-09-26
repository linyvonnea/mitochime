.PHONY: activate align features train predict eval
ENV?=mitochime
CONF?=configs/config.example.yaml

activate:
	@echo "conda activate $(ENV)"

align:
	python -m mitochime.cli align --config $(CONF)

features:
	python -m mitochime.cli features --config $(CONF)

train:
	python -m mitochime.cli train --config $(CONF)

predict:
	python -m mitochime.cli predict --config $(CONF)

eval:
	python -m mitochime.cli eval --config $(CONF)
