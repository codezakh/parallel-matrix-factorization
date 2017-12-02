import unittest

from pyspark import SparkContext
import numpy as np

from MFspark import predict, pred_diff, gradient_u, gradient_v


class GradientPredictionTestCase(unittest.TestCase):
    def test_predict(self):
        user_profile = np.ones((3,1))
        item_profile = np.ones((3,1))
        self.assertEqual(predict(user_profile, item_profile), 3)
        user_profile = np.array([1,1,1])
        item_profile = np.array([1,1,1])
        self.assertEqual(predict(user_profile, item_profile), 3)


# class ParallelLogisticRegressionTestCase(unittest.TestCase):
#     @classmethod
#     def setUpClass(cls):
#         super(ParallelLogisticRegressionTestCase, cls).setUpClass()
#         cls.sc = SparkContext(appName="ParallelLogisticRegressionTestCase")
#
#     @classmethod
#     def tearDownClass(cls):
#         super(ParallelLogisticRegressionTestCase, cls).tearDownClass()
#         cls.sc.stop()
