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

// https://leetcode.cn/problems/length-of-the-longest-subsequence-that-sums-to-target/
// 给你一个下标从 0 开始的整数数组 nums 和一个整数 target 。
// 返回和为 target 的 nums 子序列中，子序列 长度的最大值 。如果不存在和为 target 的子序列，返回 -1
func lengthOfLongestSubsequence(nums []int, target int) int {
	n := len(nums)
	// dfs表示nums[0:i]之间，能组成t的和的最长子序列长度
	var dfs func(i int, t int) (r int)
	cache := makeMatrixWithInitialFunc[int](n, target+1, func(i, j int) int {
		return -1
	})
	dfs = func(i int, c int) (r int) {
		if i < 0 {
			if c == 0 {
				return 0
			}
			return math.MinInt
		}
		if cache[i][c] != -1 {
			return cache[i][c]
		}
		defer func() {
			cache[i][c] = r
		}()
		if c == 0 {
			return 0
		}
		if c < nums[i] {
			return dfs(i-1, c)
		}
		return max(dfs(i-1, c-nums[i])+1, dfs(i-1, c))
	}
	res := dfs(n-1, target)
	if res < 0 {
		return -1
	}
	return res
}

func lengthOfLongestSubsequenceIter(nums []int, target int) int {
	n := len(nums)
	f := makeMatrixWithInitialFunc[int](n+1, target+1, func(i, j int) int {
		return math.MinInt
	})
	//f[i+1][c] = max(f[i][c],f[i][c-nums[i]+1),f[0][0]=0
	f[0][0] = 0
	for i := 0; i < n; i++ {
		x := nums[i]
		for c := 0; c <= target; c++ {
			if c < x {
				f[i+1][c] = f[i][c]
			} else {
				f[i+1][c] = max(f[i][c], f[i][c-x]+1)
			}
		}
	}
	res := f[n][target]
	if res < 0 {
		return -1
	}
	return res
}

// https://leetcode.cn/problems/ways-to-express-an-integer-as-sum-of-powers/description/
// 给你两个 正 整数 n 和 x 。
// 请你返回将 n 表示成一些 互不相同 正整数的 x 次幂之和的方案数。换句话说，你需要返回互不相同整数 [n1, n2, ..., nk] 的集合数目，满足 n = n1^x + n2^x + ... + nk^x 。
func numberOfWays(n int, x int) int {
	m := 1
	for ; pow(m, x) <= n; m++ {
	}
	f := makeMatrixWithInitialFunc[int](m+1, n+1, nil)
	f[0][0] = 1
	for i := 1; i <= m; i++ {
		v := pow(i, x)
		for c := 0; c <= n; c++ {
			if c < v {
				f[i][c] = f[i-1][c]
			} else {
				f[i][c] = f[i-1][c-v] + f[i-1][c]
			}
		}
	}
	return f[m][n] % 1_000_000_007
}

func pow(i, x int) int {
	return int(math.Pow(float64(i), float64(x)))
}

// https://leetcode.cn/problems/ones-and-zeroes/description/
// 给你一个二进制字符串数组 strs 和两个整数 m 和 n 。
// 请你找出并返回 strs 的最大子集的长度，该子集中 最多 有 m 个 0 和 n 个 1 。
// 如果 x 的所有元素也是 y 的元素，集合 x 是集合 y 的 子集 。
func findMaxForm(strs []string, m int, n int) int {
	l := len(strs)
	count0 := make([]int, l)
	count1 := make([]int, l)
	for i, str := range strs {
		for _, c := range str {
			if c == '0' {
				count0[i]++
			} else {
				count1[i]++
			}
		}
	}
	f := make([][][]int, l+1)
	for i := 0; i < l+1; i++ {
		f[i] = make([][]int, m+1)
		for j := 0; j < m+1; j++ {
			f[i][j] = make([]int, n+1)
		}
	}
	for i := 0; i < l; i++ {
		for j := 0; j < m+1; j++ {
			for k := 0; k < n+1; k++ {
				if j < count0[i] || k < count1[i] {
					f[i+1][j][k] = f[i][j][k]
				} else {
					f[i+1][j][k] = max(f[i][j][k], f[i][j-count0[i]][k-count1[i]]+1)
				}
			}
		}
	}
	return f[l][m][n]
	//var dfs func(i, zeroc, onec int) (r int)
	//dfs = func(i, zeroc, onec int) (r int) {
	//	if i < 0 {
	//		return 0
	//	}
	//
	//	if count1[i] > onec || count0[i] > zeroc {
	//		return dfs(i-1, zeroc, onec)
	//	} else {
	//		return max(dfs(i-1, zeroc, onec), dfs(i-1, zeroc-count0[i], onec-count1[i])+1)
	//	}
	//}
	//return dfs(l-1, m, n)
}

// 😭😭😭😭😭😭😭😭😭😭😭😭😭😭😭😭😭😭😭😭😭完全背包😭😭😭😭😭😭😭😭😭😭😭😭😭😭😭😭😭😭😭😭//
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
