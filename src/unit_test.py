import unittest
import Complete_code

class TestCompleteCode(unittest.TestCase):
    def test_xgbboost123(self):
        model1 = Complete_code()
        X=all_train.drop(["Survived"],axis=1)
        y=all_train['Survived']
        model = XGBClassifier()
        model.fit(X,y)
        assert type(model1) == type(model_)
        assert model1 is not model_
