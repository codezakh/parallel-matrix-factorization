from pyspark import SparkContext
import MFspark

from pyspark.mllib.random import RandomRDDs

sc = SparkContext(appName="ItemProfileTestCase")

R = sc.parallelize([
    (1, 1, 0.5),
    (1, 2, 0.7),
    (2, 2, 0.9)

])
d = 4
N = 1
MFspark.generateItemProfiles(R, 4, 1, sc, 1)
U = sc.parallelize([
(1, 0.5),
(2, 0.3)
])
# print MFspark.joinAndPredictAll(R, U, U, 1).collect()


R = sc.parallelize([
    ('user1', 'item1', 'rating1'),
    ('user1', 'item2', 'rating2'),
    ('user1', 'item3', 'rating3'),
    ('user2', 'item1', 'rating4'),
    ('user2', 'item2', 'rating5'),
])

U = sc.parallelize([
('user1', 'user1_profile'),
('user2', 'user2_profile'),
('user3', 'user3_profile'),
])

V = sc.parallelize([
('item1', 'item1_profile'),
('item2', 'item2_profile'),
('item3', 'item3_profile'),
])

#now actually does a numeric calculation, so the above code cannot be uysed
# print MFspark.joinAndPredictAll(R, U, V, 1).collect()
