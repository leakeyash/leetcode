# LeetCode Practice

This repository is used to record the progress in leetcode.

My target is 1800 in weekly contests, I suppose the target date is before the end day of 2022.

| Date | Contest | AC | Rank | Global Rank | Score | change |
| ---------- | ------- | -- | ---- | ----- | ----- | ------ |
| 2022-08-28 | 308 Weekly | 3 | 2905/6394 | 7764/24178 |1534| - |
| 2022-09-03 | 86 Bi-Weekly | 1 | 3392/4401 | 16498/26069 |1478| -56|
| 2022-09-04 | 309 Weekly | 2 | 3308/7972 | 8466/25750 |1503| +25|
| 2022-09-17 | 87 Bi-Weekly | 2 | 1287/4005 | 4488/23589 |1544| +41|
| 2022-09-18 | 311 Weekly | 3 | 3552/6710 | 10209/25376 |1545| +1|
| 2022-09-25 | 312 Weekly | 1 | 4171/6638 | 12737/24973 |1527| -18|
| 2022-10-01 | 88 Bi-Weekly | 2 | 1700/3905 | 6275/20075 |1538| +11|
| 2022-10-02 | 313 Weekly | 3 | 2271/5445 | 7130/22878 |1553| +15|
| 2022-10-08 | 314 Weekly | 3 | 1151/4838 | 3806/22620 |1591 | +38 |
| 2022-10-15 | 89 Bi-Weekly | 2 | 1437/3984 | 3975/20877 |1611 | +20 |
| 2022-10-16 | 315 Weekly | 3 | 2384/6490 | 6273/23449 |1625 | +14 |
| 2022-10-23 | 316 Weekly | 2 | 1913/6387 | 6142/23398 |1636 | +11 |
| 2022-10-29 | 90 Bi-Weekly | 3 | 1196/3624 | 3577/19748 |1656 | +20 |
| 2022-10-30 | 317 Weekly | 3 | 1994/5660 | 5207/20633 |1669 | +13 |
| 2022-11-06 | 318 Weekly | 3 | 1157/5670 | 3052/22819 |1702 | +33 |
| 2022-11-13 | 319 Weekly | 3 | 1546/6175 | 4575/21766 |1716 | +14 |
| 2022-11-26 | 92 Bi-Weekly | 3 | 838/3055 | 1933/15664 | 1745 | +29 |
| 2022-11-27 | 321 Weekly | 3 | 1380/5115 | 3463/17826 | 1759 | +14 |
| 2022-12-04 | 322 Weekly | 3 | 1498/5085 | 4122/19626 | | |

## leetcode links

Weekly contest: <https://leetcode.cn/contest/weekly-contest-86/ranking/>  
Bi-Weekly contest: <https://leetcode.cn/contest/biweekly-contest-86/ranking/>  
Predictor: https://lcpredictor.onrender.com/  
Stat: https://clist.by/coder/leakeyash/

## 寻找和自己竞赛积分接近的题

地址在此：<https://zerotrac.github.io/leetcode_problem_rating/>

包含了从第 63 场周赛、第 1 场双周赛开始的所有竞赛题的难度分数。

难度分数与用户的竞赛积分是在同一套积分体系下的（即 ELO Rating System），也就是说，如果用户的分数为 AA，某一道竞赛题的分数是
BB，那么用户能够独立解决通过这道题的概率为：

$\frac{1}{1 + 10^{(B-A)/400}}$

​
通过这个概率以及所有用户在比赛中通过的情况，求一个极大似然估计就可以得到竞赛题的分数了。
