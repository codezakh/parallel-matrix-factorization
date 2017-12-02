import unittest

from pyspark import SparkContext
import numpy as np

from MFspark import (predict, pred_diff, gradient_u, gradient_v,
                    generateItemProfiles, generateUserProfiles)

import MFspark

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


class ItemProfileTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        super(ItemProfileTestCase, cls).setUpClass()
        cls.sc = SparkContext(appName="ItemProfileTestCase")

    @classmethod
    def tearDownClass(cls):
        super(ItemProfileTestCase, cls).tearDownClass()
        cls.sc.stop()

    def test_normSqRdd(self):
        rdd = self.sc.parallelize([
        (1, np.ones(4)),
        (2, np.ones(4)),
        (3, np.ones(4)),
        ])
        self.assertEqual(MFspark.normSqRDD(rdd, 1), 12)




if __name__ == "__main__":
    unittest.main()
