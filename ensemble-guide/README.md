Kaggle-Ensemble-Guide

Kaggle???????? ?????Model Ensemble)
https://mp.weixin.qq.com/s?__biz=MzAxMTU3NTkzOQ==&mid=2662346169&idx=1&sn=d2ad8f439cb525d6b6b85bc292173947&chksm=80fa7a55b78df343da8f8a5104106b791c32255ced9f9ce5991edcfb311c28bed0fbfbd821a4&scene=0&key=566c72c25179d812daee90d00030adee68e63100da72dcc4b887fe2d6dfee84ed897782a94aff1e24dc51cb8097274b9ca58ca834bcaa8bf4696e7817d9ced4a7b51d5f8328e81c0991c23c9f3a94cff&ascene=0&uin=MjIzNDEyODY0MA%3D%3D&devicetype=iMac+MacBookPro14%2C2+OSX+OSX+10.12.5+build(16F2104)&version=12020810&nettype=WIFI&fontScale=100&pass_ticket=G9DgKBxdOdwM%2BPdru%2BpYP25WhAszt%2FcpS4SAzFXlaKd8VjksB1dwyXxxRFgs4ghc
=====================

A combination of Model Ensembling methods that is extremely useful for increasing accuracy of Kaggle's submission.
For more information: http://mlwave.com/kaggle-ensembling-guide/

## Example:

    $ python correlations.py ./samples/method1.csv ./samples/method2.csv
    Finding correlation between: ./samples/method1.csv and ./samples/method2.csv
    Column to be measured: Label
    Pearson's correlation score: 0.67898
    Kendall's correlation score: 0.66667
    Spearman's correlation score: 0.71053

    $ python kaggle_vote.py "./samples/method*.csv" "./samples/kaggle_vote.csv"
    parsing: ./samples/method1.csv
    parsing: ./samples/method2.csv
    parsing: ./samples/method3.csv
    wrote to ./samples/kaggle_vote.csv

    $ python kaggle_vote.py "./samples/_*.csv" "./samples/kaggle_vote.csv" "weighted"
    parsing: ./samples/_w3_method1.csv
    Using weight: 3
    parsing: ./samples/_w2_method2.csv
    Using weight: 2
    parsing: ./samples/_w2_method3.csv
    Using weight: 2
    wrote to ./samples/kaggle_vote.csv

    $ python kaggle_rankavg.py "./samples/method*.csv" "./samples/kaggle_rankavg.csv"
    parsing: ./samples/method1.csv
    parsing: ./samples/method2.csv
    parsing: ./samples/method3.csv
    wrote to ./samples/kaggle_rankavg.csv

    $ python kaggle_avg.py "./samples/method*.csv" "./samples/kaggle_avg.csv"
    parsing: ./samples/method1.csv
    parsing: ./samples/method2.csv
    parsing: ./samples/method3.csv
    wrote to ./samples/kaggle_avg.csv

    $ python kaggle_geomean.py  "./samples/method*.csv" "./samples/kaggle_geomean.csv"
    parsing: ./samples/method1.csv
    parsing: ./samples/method2.csv
    parsing: ./samples/method3.csv
    wrote to ./samples/kaggle_geomean.csv

## Result:

    ==> ./samples/method1.csv <==
    ImageId,Label
    1,1
    2,0
    3,9
    4,9
    5,3

    ==> ./samples/method2.csv <==
    ImageId,Label
    1,2
    2,0
    3,6
    4,2
    5,3

    ==> ./samples/method3.csv <==
    ImageId,Label
    1,2
    2,0
    3,9
    4,2
    5,3

    ==> ./samples/kaggle_avg.csv <==
    ImageId,Label
    1,1.666667
    2,0.000000
    3,8.000000
    4,4.333333
    5,3.000000

    ==> ./samples/kaggle_rankavg.csv <==
    ImageId,Label
    1,0.25
    2,0.0
    3,1.0
    4,0.5
    5,0.75

    ==> ./samples/kaggle_vote.csv <==
    ImageId,Label
    1,2
    2,0
    3,9
    4,2
    5,3

    ==> ./samples/kaggle_geomean.csv <==
    ImageId,Label
    1,1.587401
    2,0.000000
    3,7.862224
    4,3.301927
    5,3.000000
