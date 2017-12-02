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

    def test_pred_diff(self):
        user_profile = np.ones((3,1))
        item_profile = np.ones((3,1))
        rating = 3
        difference = pred_diff(rating, user_profile, item_profile)
        self.assertEqual(difference, 0)

    def test_gradient_u(self):
        user_profile = np.ones((3,1))
        item_profile = np.ones((3,1))
        computed_gradient_v = gradient_u(1, user_profile, item_profile)
        self.assertEqual(list(computed_gradient_v), list(2 * item_profile))


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
