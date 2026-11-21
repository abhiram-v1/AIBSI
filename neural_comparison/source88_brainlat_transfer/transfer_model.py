class SourceModel:
    def __init__(self,pipeline,calibrator,config):
        self.pipeline=pipeline;self.calibrator=calibrator;self.config=config

    def predict(self,x):return self.pipeline.predict(x)

    def decision_function(self,x):return self.pipeline.decision_function(x)

    def predict_proba(self,x):return self.calibrator.predict_proba(self.decision_function(x)[:,None])

    @property
    def probability_operating_threshold(self):return float(self.calibrator.predict_proba([[0.]])[0,1])
