package golang

import "math"

// 😭😭😭😭😭😭😭😭😭😭😭😭😭😭😭😭😭😭😭😭😭0/1背包😭😭😭😭😭😭😭😭😭😭😭😭😭😭😭😭😭😭😭😭//
// 给你一个非负整数数组 nums 和一个整数 target 。
//
// 向数组中的每个整数前添加 '+' 或 '-' ，然后串联起所有整数，可以构造一个 表达式 ：
//
// 例如，nums = [2, 1] ，可以在 2 之前添加 '+' ，在 1 之前添加 '-' ，然后串联起来得到表达式 "+2-1" 。
// 返回可以通过上述方法构造的、运算结果等于 target 的不同 表达式 的数目。
func findTargetSumWays(nums []int, target int) int {
	//需要先推导出来，设加+的数总和为p，加-的数总和则是sum-p，则可以得到p-(sum-p)=t => p=(sum+t)/2 即是01中的cap
	var sum int
	for _, num := range nums {
		sum += num
	}
	capacity := target + sum
	//边界条件
	if capacity < 0 || capacity%2 != 0 {
		return 0
	}
	capacity /= 2
	n := len(nums)
	var f = make([][]int, n+1)
	f[0] = make([]int, capacity+1)
	//cap=0的时候，方案数为1，f[i][j]代表只有前i个数的情况下，和为j的方案数
	f[0][0] = 1
	for i, num := range nums {
		f[i+1] = make([]int, capacity+1)
		for j := 0; j < capacity+1; j++ {
			if j < num {
				//放不下
				f[i+1][j] = f[i][j]
			} else {
				//选了和不选的方案数加起来
				f[i+1][j] = f[i][j] + f[i][j-num]
			}
		}
	}
	return f[n][capacity]
}

// 给你一个整数数组 coins ，表示不同面额的硬币；以及一个整数 amount ，表示总金额。
// 计算并返回可以凑成总金额所需的 最少的硬币个数 。如果没有任何一种硬币组合能组成总金额，返回 -1 。
// 你可以认为每种硬币的数量是无限的。
func coinChange(coins []int, amount int) int {
	if amount < 0 {
		return -1
	}
	n := len(coins)
	f := make([][]int, n+1)
	//f[i][j]代表前i个coin可以组成j金额的最小硬币数 f[i][j] = min(f[i-1][j],f[i][j-coin[i]]+1)
	f[0] = make([]int, amount+1)
	inf := math.MaxInt / 2
	for i := range f[0] {
		f[0][i] = inf
	}
	//需要构成和为0的最少硬币个数为0
	f[0][0] = 0
	for i, x := range coins {
		f[i+1] = make([]int, amount+1)
		for c := 0; c < amount+1; c++ {
			if x > c {
				f[i+1][c] = f[i][c]
			} else {
				//可以无限使用，所以可能用了第i+1个
				f[i+1][c] = min(f[i][c], f[i+1][c-x]+1)
			}
		}
	}
	res := f[n][amount]
	if res < inf {

		return res
	} else {
		return -1
	}
}
