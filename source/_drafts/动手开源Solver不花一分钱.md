---
title:  Pyomo开源优化建模解决方案
author: Yan
tags: open source, solver, optimization
---

### Pyomo
Pyomo是一个基于python的开源优化建模框架。他的功能和OPL, AMPL一样，都是建模的语言。因而只负责完成建模部门，不负责求解(solve)，求解是solver的职责，稍后会讲到。 Pyomo是跨平台的，不论是win/mac/linux都没问题。python 2或3也都没问题。安装也十分简便挺简单的。

首次使用，我强烈建议使用Anaconda环境进行开发和安装。如果没有的话，赶紧去装一个吧，Conda可以免除很多不必要的麻烦。[Anaconda如何安装请戳这里](https://conda.io/docs/user-guide/install/index.html)

使用Anaconda环境，安装过程如下
```
// 安装本体
conda install -c conda-forge pyomo
// 安装附加程序
conda install -c conda-forge pyomo.extras
```
如果动手能力强，想使用pip安装，也没问题。（但pip安装可能会有各种意外的错误，初次尝试还是建议用conda）
```
// 安装本体
pip install pyomo
// 安装Pyomo的附加程序
pyomo install -extras
```

### Solver
有了模型，就要靠求解器 (solver) 来求解。大名鼎鼎的Gurobi, Cplex 和 MPExpress都是知名的求解器。但我们追求的目标是可靠又便宜，装备不花一分钱，所以以上就不多讨论了。开源的solver选择并不多，达到Gurobi和Cplex水平的还尚(chi1)未(ren2)出(shuo1)现(meng4)。市面上有点人气的，有以下几种
1. GLPK (Gnu Linear Programming KIt)
1. ipopt
1. Google OR tool

接下来就是安装GLPK
```
conda install -c conda-forge glpk
```
使用conda来安装，轻松又愉快。
```
sudo chown -R user anaconda3
```