
train:
	python $(CURDIR)/train.py -v T

register_model:
	python $(CURDIR)/register_mlflow_model.py -r 334250d4c08241308592fac52a5ae7f8 -to_p T -exn "Best Practice"

predict:
	python $(CURDIR)/predict.py -d $(CURDIR)/data/prediction_samples/prediction_sample_3359.csv
