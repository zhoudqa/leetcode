package golang

import (
	"math"
	"sort"
)

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

// 给定两个单词 word1 和 word2 ，返回使得 word1 和  word2 相同所需的最小步数。
//
// 每步 可以删除任意一个字符串中的一个字符。
func minDistanceDel(word1 string, word2 string) int {
	n := len(word1)
	m := len(word2)
	f := make([][]int, n+1)
	f[0] = make([]int, m+1)
	for i := 1; i < m+1; i++ {
		f[0][i] = i
	}
	for i := 1; i < n+1; i++ {
		f[i] = make([]int, m+1)
		f[i][0] = i
	}
	f[0][0] = 0
	for i := 1; i <= n; i++ {
		for j := 1; j <= m; j++ {
			if word1[i-1] == word2[j-1] {
				f[i][j] = f[i-1][j-1]
			} else {
				f[i][j] = min(f[i-1][j]+1, f[i][j-1]+1, f[i-1][j-1]+2)
			}
		}
	}
	return f[n][m]
}

// 给你一个整数数组 nums ，找到其中最长严格递增子序列的长度。
//
// 子序列 是由数组派生而来的序列，删除（或不删除）数组中的元素而不改变其余元素的顺序。
func lengthOfLIS(nums []int) int {
	//g[i]代表包含nums[i]的最长的子序列
	g := make([]int, 0)
	for _, num := range nums {
		firstGTNumIndex := sort.SearchInts(g, num)
		if firstGTNumIndex == len(g) {
			//不存在比num小的，可以继续增加g
			g = append(g, num)
		} else {
			//替换对应的位置，仅仅代表增加了这个数后最大长度还是原来的长度
			g[firstGTNumIndex] = num
		}
	}
	return len(g)
}

// 给你一个整数数组 nums 。nums 的每个元素是 1，2 或 3。在每次操作中，你可以删除 nums 中的一个元素。返回使 nums 成为 非递减 顺序所需操作数的 最小值。
func minimumOperations(nums []int) int {
	//g[i]代表加入nums[i]后的最长序列
	g := make([]int, 0)
	for _, x := range nums {
		firstNumIndex := sort.Search(len(g), func(i int) bool {
			return g[i] > x
		})
		if firstNumIndex == len(g) {
			g = append(g, x)
		} else {
			g[firstNumIndex] = x
		}
	}
	return len(nums) - len(g)

}

//😭😭😭😭😭😭😭😭😭😭😭😭😭😭😭😭😭😭😭😭动态规划-区间😭😭😭😭😭😭😭😭😭😭😭😭😭😭😭😭😭😭😭😭//

// 给你一个字符串 s ，找出其中最长的回文子序列，并返回该序列的长度。
// 子序列定义为：不改变剩余字符顺序的情况下，删除某些字符或者不删除任何字符形成的一个序列。
func longestPalindromeSubseq(s string) int {
	//dp[i,j]表示s[i,j]的子序列中最长子序列长度，dp[i,i]=1
	n := len(s)
	cache := makeMatrixWithInitialFunc(n, n, func(i, j int) int {
		if i == j {
			return 1
		}
		return -1
	})
	var dfs func(i, j int) int
	dfs = func(i, j int) int {
		if i > j {
			return 0
		}
		if cache[i][j] != -1 {
			return cache[i][j]
		}
		var res int
		defer func() {
			cache[i][j] = res
		}()
		if s[i] == s[j] {
			res = dfs(i+1, j-1) + 2
		} else {
			res = max(dfs(i+1, j), dfs(i, j-1))
		}
		return res
	}
	return dfs(0, n-1)
}

func longestPalindromeSubseqRet(s string) int {
	n := len(s)
	f := makeMatrixWithInitialFunc[int](n, n, nil)
	//i从i+1来，j从j-1来，所以一个倒序一个正序，且初始状态一定是f[i][i] 所以j必须从i+1开始枚举
	for i := n - 1; i >= 0; i-- {
		f[i][i] = 1
		for j := i + 1; j < n; j++ {
			if s[i] == s[j] {
				f[i][j] = f[i+1][j-1] + 2
			} else {
				f[i][j] = max(f[i+1][j], f[i][j-1])
			}
		}
	}
	return f[0][n-1]
}

func makeMatrixWithInitialFunc[T comparable](length, width int, initFunc func(i, j int) T) [][]T {
	cache := make([][]T, width)
	for i := 0; i < width; i++ {
		cache[i] = make([]T, length)
		if initFunc != nil {
			for j := range cache[i] {
				cache[i][j] = initFunc(i, j)
			}
		}
	}
	return cache
}

// 给你一个字符串 s，请你将 s 分割成一些子串，使每个子串都是
// 回文串
// 返回符合要求的 最少分割次数 。
func minCut(s string) int {
	n := len(s)
	f := make([]int, n+1)
	for i := range f {
		f[i] = math.MaxInt32
	}
	f[0] = -1
	//缓存从i,j是否回文的信息
	cache := makeMatrixWithInitialFunc[*bool](n, n, nil)
	for i := 0; i < n; i++ {
		for k := i; k >= 0; k-- {
			if isPalindromeFunc(s, k, i, cache) {
				f[i+1] = min(f[i+1], 1+f[k])
			}
		}
	}
	return f[n]
}

func minCutMemo(s string) int {
	n := len(s)
	//dfs搜索以i结尾的s最少分割数，dfs[0]=0，代表一个字符就是回文不需要分割
	var dfs func(i int) int
	isPalindrome := func(l, r int) bool {
		for l < r {
			if s[l] != s[r] {
				return false
			}
			l++
			r--
		}
		return true
	}
	memo := make([]int, n)

	for i := 1; i < n; i++ {
		memo[i] = -1
	}
	dfs = func(i int) int {
		if i < 0 {
			//因为下面遍历dfs(j)的时候可能会小于0，所以保证dfs(0)=dfs(-1)+1即dfs(-1)=-1
			return -1
		}
		if memo[i] != -1 {
			return memo[i]
		}
		//min需要初始化为inf
		memo[i] = math.MaxInt32
		for j := i; j >= 0; j-- {
			if isPalindrome(j, i) {
				memo[i] = min(memo[i], 1+dfs(j-1))
			}
		}
		return memo[i]
	}
	return dfs(n - 1)
}

func Ptr[T any](a T) *T {
	return &a
}

func isPalindromeFunc(s string, l, r int, cache [][]*bool) (ret bool) {
	if cache[l][r] != nil {
		return *cache[l][r]
	}
	defer func() {
		cache[l][r] = Ptr(ret)
	}()
	for l < r {
		if s[l] != s[r] {
			return false
		}
		l++
		r--
	}
	return true
}
