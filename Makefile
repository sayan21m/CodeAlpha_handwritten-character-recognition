.PHONY: install train train-digit train-character test run backend frontend

install:
	cd backend && pip install -r requirements.txt

train:
	cd backend && python train.py --model both

train-digit:
	cd backend && python train_digit.py

train-character:
	cd backend && python train_character.py

test:
	cd backend && pytest tests/ -v

backend:
	cd backend && python app.py

frontend:
	python -m http.server 8080 --directory docs

run:
	./run.sh
