.PHONY: train evidence test api dashboard
train:      ## entrena, evalúa y escribe models/ y reports/
	cd src && python -m pqm.train
test:
	python -m pytest -q
evidence:
	cd src && python -m pqm.evidence
api:
	PYTHONPATH=src uvicorn pqm.api.app:app --reload
dashboard:
	streamlit run app/streamlit_app.py
