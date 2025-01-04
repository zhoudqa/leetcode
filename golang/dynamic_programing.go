package golang

import "sort"

//😭😭😭😭😭😭😭😭😭😭😭😭😭😭😭😭😭😭😭😭动态规划😭😭😭😭😭😭😭😭😭😭😭😭😭😭😭😭😭😭😭😭//

// 如果 nums 的子集中，任意两个整数的绝对差均不等于 k ，则认为该子数组是一个 美丽 子集。
// 返回数组 nums 中 非空 且 美丽 的子集数目。
// nums 的子集定义为：可以经由 nums 删除某些元素（也可能不删除）得到的一个数组。只有在删除元素时选择的索引不同的情况下，两个子集才会被视作是不同的子集。
func beautifulSubsets(nums []int, k int) int {
	groups := map[int]map[int]int{}
	for _, n := range nums {
		m := n % k
		if groups[m] == nil {
			groups[m] = map[int]int{}
		}
		//保存对应数字的数量
		groups[m][n]++
	}
	type pair struct{ num, cnt int }
	ans := 1
	for _, group := range groups {
		//同一个mod下面的分组
		m := len(group)
		g := make([]pair, 0)
		for num, cnt := range group {
			g = append(g, pair{num: num, cnt: cnt})
		}
		sort.Slice(g, func(i, j int) bool {
			return g[i].num < g[j].num
		})
		f := make([]int, m+1)
		f[0] = 1
		f[1] = 1 << g[0].cnt
		//f[i] 表示考虑前 i 个 key 的方案数
		for i := 1; i < m; i++ {
			if g[i].num-g[i-1].num == k {
				// (第i的所有不选i+1) + (第i-2的所有 选了i+1+不选i+1(全不选的那种情况在f[i]中包含了)
				f[i+1] = f[i] + f[i-1]*(1<<g[i].cnt-1)
			} else {
				//乘法原理，选了+不选
				f[i+1] = f[i] << g[i].cnt
			}
		}
		ans *= f[m]

	}
	//去掉空集
	return ans - 1
}

// 你是一个专业的小偷，计划偷窃沿街的房屋。每间房内都藏有一定的现金，影响你偷窃的唯一制约因素就是相邻的房屋装有相互连通的防盗系统，如果两间相邻的房屋在同一晚上被小偷闯入，系统会自动报警。
//
// 给定一个代表每个房屋存放金额的非负整数数组，计算你 不触动警报装置的情况下 ，一夜之内能够偷窃到的最高金额。
func rob(nums []int) int {
	return robRec(nums)
}

// 递推&最小状态
func robRetMinSpace(nums []int) int {
	f0 := 0
	f1 := 0
	for i := 0; i < len(nums); i++ {
		nextF := max(f0+nums[i], f1)
		f0 = f1
		f1 = nextF
	}
	return f1
}

// 递推
func robRet(nums []int) int {
	var f = make([]int, len(nums)+2)
	f[0] = 0
	f[1] = 0
	for i := 0; i < len(nums); i++ {
		f[i+2] = max(f[i+1], f[i]+nums[i])
	}
	return f[len(nums)+1]
}

// 递归
func robRec(nums []int) int {
	//f[i]表示前i个房子能偷到的最大金额
	var cache = make([]int, len(nums))
	for i := 0; i < len(nums); i++ {
		cache[i] = -1
	}
	var dfs func(i int) int
	dfs = func(i int) int {
		if i < 0 { // 递归边界（没有房子）
			return 0
		}
		if cache[i] != -1 {
			return cache[i]
		}
		cache[i] = max(nums[i]+dfs(i-2), dfs(i-1))
		return cache[i]
	}
	return dfs(len(nums) - 1)
}

// 给你两个单词 word1 和 word2， 请返回将 word1 转换成 word2 所使用的最少操作数  。
// 你可以对一个单词进行如下三种操作：
//
// 插入一个字符
// 删除一个字符
// 替换一个字符
func minDistance(word1 string, word2 string) int {
	//f[i][j]代表将word1的前i个字符转换成word2的前j个字符需要的最少操作数，
	// 当 word1[i]==word2[j]时，f[i][j]=f[i-1][j-1] 不需要插入删除或者替换，否则选以下三种最小的
	// 1. 插入不等的 f[i][j-1] 2.删除不等的 f[i-1][j] 3.替换f[i-1][j-1]+1
	//i=0或者j=0时 f为另一个坐标值
	len1 := len(word1)
	len2 := len(word2)
	f := make([][]int, len1+1)
	for j := 0; j < len1+1; j++ {
		f[j] = make([]int, len2+1)
	}
	for i := 0; i < len2+1; i++ {
		f[0][i] = i
	}
	for i, c1 := range word1 {
		f[i+1][0] = i + 1
		for j, c2 := range word2 {
			if c1 == c2 {
				f[i+1][j+1] = f[i][j]
			} else {
				f[i+1][j+1] = min(f[i+1][j], f[i][j+1], f[i][j]) + 1
			}
		}
	}
	return f[len1][len2]
}
