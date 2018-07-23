---
title: 自己动手丰衣足食：Pyomo开源优化建模解决方案
tags: 'open source, solver, optimization'
author: Yan
date: 2018-07-22 23:49:53
---
<img  src="https://farm1.staticflickr.com/932/43581388701_349b9aebd5.jpg" width="400">
搞运筹优化，离不开建模和求解。市面上主流的优化平台如cplex和gurobi对于各种优化问题都有极佳的表现。但缺点也十分明显，那就是贵！ 早先Gurobi在同IBM cplex竞争的时候，价格十分低廉，但随着Gurobi慢慢证明了自己一哥的地位之后，定价策略也就不那么亲民了。颇有屠龙勇士变成恶龙的感觉。

经费在燃烧，老板在咆哮，对于初创团队和小型项目，模型一般不大，有没有便宜又可靠的优化解决方案呢？ 有的，我们自己动手丰衣足食，可以完全用开源工具高效的工作！ 这就是今天要介绍的**pyomo + GLPK/ipopt**.
<!--more-->
### Pyomo
Pyomo是一个基于python的开源优化建模框架。他的功能和OPL, AMPL一样，都是建模的语言。因而只负责完成建模部门，不负责求解(solve)，求解是solver的职责，稍后会讲到。 Pyomo是跨平台的，不论是win/mac/linux都没问题。python 2或3也都没问题。安装也十分简便。

首次尝试，强烈建议在Anaconda环境下进行。如果没有的话，赶紧去装一个吧，Conda可以真切免除很多不必要的配置麻烦。[Anaconda如何安装请戳这里](https://conda.io/docs/user-guide/install/index.html)

使用Anaconda环境，安装pyomo过程如下
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
有了模型，就要靠求解器 (solver) 来求解。大名鼎鼎的Gurobi, Cplex 和 ExpressMP都是知名的求解器。然而开源的solver选择并不多，能达到Gurobi和Cplex水平的还尚(chi1)未(ren2)出(shuo1)现(meng4)。市面上比较有人气同时pyomo也支持的有：
1. GLPK (Gnu Linear Programming KIt)： 顾名思义，只能解线性问题。
1. ipopt（Interior Point OPTimizer, 读作eye-pea-Opt，不是爱婆婆忒）针对非线性问题（non-linear programming）

两个都能使用conda来安装，这也是为什么强烈推荐使用anaconda环境，实在是轻松加愉快。
```
conda install -c conda-forge glpk
conda install -c conda-forge ipopt
```

### 测试
让我们来测试一个简单的二次规划的模型。

$$
\begin{aligned}
& \text{minimize} & & x^2 + y \\
& & &
x, y \in [-2, 2]
\end{aligned}
$$

下面是实现过程。
```python
from pyomo.environ import *

# 定义模型对象model
model = ConcreteModel()
# 定义决策变量，定义区间
model.x = Var(initialize = -1.2, bounds = (-2, 2))
model.y = Var(initialize = 1.0, bounds = (-2, 2))
# 定义目标函数
model.obj = Objective(
    expr = model.x**2 + model.y,
    sense = minimize
)

# 解完之后，用post process来显示x和y的数值
def pyomo_postprocess(options=None, instance=None, results=None):
    model.x.display()
    model.y.display()

# 主函数，模型求解
if __name__ == '__main__':
    from pyomo.opt import SolverFactory
    import pyomo.environ
    # 选择ipopt作为solver。因为模型是二次，需要用非线性问题的solver
    opt = SolverFactory('ipopt')
    # 求解
    results = opt.solve(model)
    # 显示求解过程的信息
    results.write()
    print("\n Solution: \n" + '-'*60)
    # 调用之前定义的post process，显示决策变量最终的结果
    pyomo_postprocess(None, model, results)
```
如果pyomo和ipopt都安装正确，那么将会看到以下结果：
```
# ==========================================================
# = Solver Results                                         =
# ==========================================================
# ----------------------------------------------------------
#   Problem Information
# ----------------------------------------------------------
Problem: 
- Lower bound: -inf
  Upper bound: inf
  Number of objectives: 1
  Number of constraints: 0
  Number of variables: 2
  Sense: unknown
# ----------------------------------------------------------
#   Solver Information
# ----------------------------------------------------------
Solver: 
- Status: ok
  Message: Ipopt 3.12.10\x3a Optimal Solution Found
  Termination condition: optimal
  Id: 0
  Error rc: 0
  Time: 0.04419064521789551
# ----------------------------------------------------------
#   Solution Information
# ----------------------------------------------------------
Solution: 
- number of solutions: 0
  number of solutions displayed: 0

 Solution: 
------------------------------------------------------------
x : Size=1, Index=None
    Key  : Lower : Value                   : Upper : Fixed : Stale : Domain
    None :    -2 : -2.5183844110924177e-23 :     2 : False : False :  Reals
y : Size=1, Index=None
    Key  : Lower : Value : Upper : Fixed : Stale : Domain
    None :    -2 :  -2.0 :     2 : False : False :  Reals
obj : Size=1, Index=None, Active=True
    Key  : Active : Value
    None :   True :  -2.0
```