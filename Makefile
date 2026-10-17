.PHONY: train evidence test api dashboard dictamen
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
dictamen:   ## dictamen de viabilidad de un proceso: make dictamen LOTES=lotes.csv
	PYTHONPATH=src python -m pqm.viability_report $(LOTES) dictamen_viabilidad.html
