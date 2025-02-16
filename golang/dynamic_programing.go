package golang

import (
	"math"
	"slices"
	"sort"
	"strings"
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

//给定两个字符串 text1 和 text2，返回这两个字符串的最长 公共子序列 的长度。如果不存在 公共子序列 ，返回 0 。
//
//一个字符串的 子序列 是指这样一个新的字符串：它是由原字符串在不改变字符的相对顺序的情况下删除某些字符（也可以不删除任何字符）后组成的新字符串。

// 例如，"ace" 是 "abcde" 的子序列，但 "aec" 不是 "abcde" 的子序列。
// 两个字符串的 公共子序列 是这两个字符串所共同拥有的子序列。
func longestCommonSubsequence(text1 string, text2 string) int {
	//dp[i,j]代表text1的前i位和text2的前j位的最长公共子序列 dp[0,0]为0
	cache := makeMatrixWithInitialFunc(len(text1)+1, len(text2)+1, func(i, j int) int {
		if i == 0 || j == 0 {
			return 0
		}
		return -1
	})
	var dfs func(i, j int) int
	dfs = func(i, j int) int {
		if cache[i][j] == -1 {
			if text1[i-1] == text2[j-1] {
				cache[i][j] = dfs(i-1, j-1) + 1
			} else {
				cache[i][j] = max(dfs(i, j-1), dfs(i-1, j))
			}
		}
		return cache[i][j]
	}
	return dfs(len(text1), len(text2))
}
func longestCommonSubsequenceRet(text1 string, text2 string) int {
	m := len(text1)
	n := len(text2)
	f := make([][]int, m+1)
	for i := 0; i <= m; i++ {
		f[i] = make([]int, n+1)
	}

	for i := 1; i <= m; i++ {
		for j := 1; j <= n; j++ {
			if text1[i-1] == text2[j-1] {
				f[i][j] = f[i-1][j-1] + 1
			} else {
				f[i][j] = max(f[i][j-1], f[i-1][j])
			}
		}
	}
	return f[m][n]
}

// https://leetcode.cn/problems/taking-maximum-energy-from-the-mystic-dungeon/
// 在神秘的地牢中，n 个魔法师站成一排。每个魔法师都拥有一个属性，这个属性可以给你提供能量。有些魔法师可能会给你负能量，即从你身上吸取能量。
// 你被施加了一种诅咒，当你从魔法师 i 处吸收能量后，你将被立即传送到魔法师 (i + k) 处。这一过程将重复进行，直到你到达一个不存在 (i + k) 的魔法师为止。
// 换句话说，你将选择一个起点，然后以 k 为间隔跳跃，直到到达魔法师序列的末端，在过程中吸收所有的能量。
// 给定一个数组 energy 和一个整数k，返回你能获得的 最大 能量。
func maximumEnergy(energy []int, k int) int {
	//遍历n-1到n-k，计算这几个结尾的最大值
	n := len(energy)
	suffixSum := make([]int, k)
	ans := math.MinInt
	for j := 1; j <= k; j++ {
		for i := n - j; i >= 0; i -= k {
			suffixSum[j-1] += energy[i]
			ans = max(ans, suffixSum[j-1])
		}
	}
	return ans
}

// https://leetcode.cn/problems/maximum-difference-score-in-a-grid/description/?slug=maximum-difference-score-in-a-grid&region=local_v2
// 给你一个由 正整数 组成、大小为 m x n 的矩阵 grid。你可以从矩阵中的任一单元格移动到另一个位于正下方或正右侧的任意单元格（不必相邻）。从值为 c1 的单元格移动到值为 c2 的单元格的得分为 c2 - c1 。
// 你可以从 任一 单元格开始，并且必须至少移动一次。
// 返回你能得到的 最大 总得分。
func maxScore(grid [][]int) int {
	// f[i+1][j+1]代表[0][0]到[i][j]构成的矩形中的最小值
	m := len(grid)
	n := len(grid[0])
	ans := math.MinInt
	f := makeMatrixWithInitialFunc(m+1, n+1, func(i, j int) int {
		return math.MaxInt
	})
	for i, row := range grid {
		for j, v := range row {
			//假如以当前格子为终点的话，那只要找到左上的最小值，就可以得到最大结果了，中间如何移动都会被抵消掉
			upLeftMin := min(f[i][j+1], f[i+1][j])
			ans = max(ans, v-upLeftMin)
			f[i+1][j+1] = min(v, upLeftMin)
		}
	}
	return ans
}

// https://leetcode.cn/problems/climbing-stairs/
func climbStairs(n int) int {
	var dfs func(i int) int
	var cache = make([]int, n)
	for i := range cache {
		cache[i] = -1
	}
	dfs = func(i int) (r int) {
		if i < 2 {
			return 1
		}
		if cache[i-1] > 0 {
			return cache[i-1]
		} else {
			defer func() {
				cache[i-1] = r
			}()
		}
		return dfs(i-1) + dfs(i-2)
	}
	return dfs(n)
}

func climbStairsIter(n int) int {
	f0 := 1
	f1 := 1
	for i := 2; i <= n; i++ {
		f := f0 + f1
		f0 = f1
		f1 = f
	}
	return f1
}

// https://leetcode.cn/problems/min-cost-climbing-stairs/description/
// 给你一个整数数组 cost ，其中 cost[i] 是从楼梯第 i 个台阶向上爬需要支付的费用。一旦你支付此费用，即可选择向上爬一个或者两个台阶。
// 你可以选择从下标为 0 或下标为 1 的台阶开始爬楼梯。
// 请你计算并返回达到楼梯顶部的最低花费。
func minCostClimbingStairs(cost []int) int {
	n := len(cost)
	cache := makeCacheWithInitialFunc(n, func(i int) int {
		return -1
	})
	var dfs func(i int) int
	dfs = func(i int) (r int) {
		if i < 2 {
			return 0
		}
		if cache[i] >= 0 {
			return cache[i]
		} else {
			defer func() {
				cache[i] = r
			}()
		}
		return min(dfs(i-1)+cost[i-1], dfs(i-2)+cost[i-2])
	}
	return dfs(n)
}

func minCostClimbingStairsIter(cost []int) int {
	var f0, f1 int
	n := len(cost)
	for i := 2; i <= n; i++ {
		//只和前面2个的状态有关系，i-2跳2步上来或者i-1跳一步上来
		f := min(f0+cost[i-2], f1+cost[i-1])
		f0 = f1
		f1 = f
	}
	return f1
}

// https://leetcode.cn/problems/count-ways-to-build-good-strings/description/
// 给你整数 zero ，one ，low 和 high ，我们从空字符串开始构造一个字符串，每一步执行下面操作中的一种：
// 将 '0' 在字符串末尾添加 zero  次。
// 将 '1' 在字符串末尾添加 one 次。
// 以上操作可以执行任意次。
// 如果通过以上过程得到一个 长度 在 low 和 high 之间（包含上下边界）的字符串，那么这个字符串我们称为 好 字符串。
// 请你返回满足以上要求的 不同 好字符串数目。由于答案可能很大，请将结果对 109 + 7 取余 后返回。
func countGoodStrings(low int, high int, zero int, one int) int {
	mod := 1_000_000_007
	var dfs func(l int) int
	cache := makeCacheWithInitialFunc(high+1, func(i int) int {
		return -1
	})
	//长度为l时的组合数
	dfs = func(l int) (r int) {
		if l < 0 {
			return 0
		}
		//空串的方法数为1
		if l == 0 {
			return 1
		}
		c := &cache[l-1]
		if *c < 0 {
			*c = (dfs(l-zero) + dfs(l-one)) % mod
		}
		return *c
	}
	ans := 0
	for l := low; l <= high; l++ {
		ans += dfs(l)
	}
	return ans % mod
}

func countGoodStringsIter(low int, high int, zero int, one int) int {
	mod := 1_000_000_007
	f := make([]int, high+1) //f[i]表示构造长度为i的方案数
	f[0] = 1
	ans := 0
	for l := 1; l <= high; l++ {
		if l >= zero {
			f[l] += f[l-zero]
		}
		if l >= one {
			f[l] = (f[l] + f[l-one]) % mod
		}
		if l >= low {
			ans = (ans + f[l]) % mod
		}
	}
	return ans % mod
}

// https://leetcode.cn/problems/combination-sum-iv/description/
// 给你一个由 不同 整数组成的数组 nums ，和一个目标整数 target 。请你从 nums 中找出并返回总和为 target 的元素组合的个数。
// 题目数据保证答案符合 32 位整数范围。
func combinationSum4(nums []int, target int) int {
	sort.Ints(nums)
	//dfs 代表总和为t的组合个数
	var dfs func(t int) int
	cache := makeCacheWithInitialFunc(target+1, func(i int) int {
		return -1
	})
	dfs = func(t int) (r int) {
		if t < 0 {
			return 0
		}
		if t == 0 {
			return 1
		}
		if cache[t] != -1 {
			return cache[t]
		} else {
			defer func() {
				cache[t] = r
			}()
		}
		for _, n := range nums {
			if t >= n {
				r += dfs(t - n)
			}
		}
		return
	}
	return dfs(target)
}

// https://leetcode.cn/problems/count-number-of-texts/description/

const mod = 1_000_000_007

// 测试长度上限
const mx = 100_001

// 重复字符为3个可能性长度为i的时候的结果
var f = [mx]int{1, 1, 2, 4}

// 重复字符为4个可能性长度为i的时候的结果
var g = f

func init() {
	for i := 4; i < mx; i++ {
		f[i] = (f[i-1] + f[i-2] + f[i-3]) % mod
		g[i] = (g[i-1] + g[i-2] + g[i-3] + g[i-4]) % mod
	}
}

func countTexts(s string) int {
	ans, cnt := 1, 0
	for i, c := range s {
		cnt++
		//直到不重复，计算当前重复
		if i == len(s)-1 || byte(c) != s[i+1] {
			if c != '7' && c != '9' {
				ans = ans * f[cnt] % mod
			} else {
				ans = ans * g[cnt] % mod
			}
			cnt = 0
		}
	}
	return ans
}

func countTextsMySelf(s string) int {
	cache3 := make(map[int]int)
	cache4 := make(map[int]int)
	var chainedChars []string
	sb := strings.Builder{}
	n := len(s)
	for i, c := range s {
		sb.WriteByte(s[i])
		if i+1 != n && byte(c) != s[i+1] {
			sb.WriteByte(',')
		}
	}
	chainedChars = strings.Split(sb.String(), ",")

	var dfs func(l int, c uint8) int
	dfs = func(l int, c uint8) (r int) {
		if l < 0 {
			return 0
		}
		var ch map[int]int
		hit3 := c != '7' && c != '9'
		if hit3 {
			ch = cache3
		} else {
			ch = cache4
		}
		if cache3[l] != 0 {
			return cache3[l]
		}
		defer func() {
			ch[l] = r
		}()
		if l == 0 || l == 1 {
			return 1
		}
		if l == 2 {
			return 2
		}
		if l == 3 {
			return 4
		}
		if hit3 {
			return (dfs(l-1, c) + dfs(l-2, c) + dfs(l-3, c)) % mod
		} else {
			return (dfs(l-1, c) + dfs(l-2, c) + dfs(l-3, c) + dfs(l-4, c)) % mod
		}
	}
	var ans = 1
	for _, chainedChar := range chainedChars {
		ans = (ans * dfs(len(chainedChar), chainedChar[0])) % mod
	}
	return ans
}

// https://leetcode.cn/problems/delete-and-earn/description/
// 给你一个整数数组 nums ，你可以对它进行一些操作。
// 每次操作中，选择任意一个 nums[i] ，删除它并获得 nums[i] 的点数。之后，你必须删除 所有 等于 nums[i] - 1 和 nums[i] + 1 的元素。
// 开始你拥有 0 个点数。返回你能通过这些操作获得的最大点数。
func deleteAndEarn(nums []int) int {
	l := slices.Max(nums)
	//ints[i]代表nums中所有值为i的总和，比如[2,2,3,3,3,4,4]对应的就是[0,0,4,9,8]，那么限制就变成了打家劫舍
	ints := make([]int, l)
	for _, n := range nums {
		ints[n] += n
	}
	var f0, f1 int
	for i := 0; i < l; i++ {
		f0, f1 = f1, max(f1, f0+ints[i])
	}
	return f1
}

// https://leetcode.cn/problems/count-number-of-ways-to-place-houses/
func countHousePlacements(n int) int {
	mod := 1_000_000_007
	f0 := 1 //相邻2块都没有放
	f1 := 2 //相邻2块放了1块
	f2 := 1 //相邻2块放了2块
	for i := 2; i <= n; i++ {
		newF0 := (f0 + f1 + f2) % mod
		newF1 := (f0*2 + f1) % mod
		newF2 := f0
		f0 = newF0
		f1 = newF1
		f2 = newF2
	}
	return (f0 + f1 + f2) % mod
}

// https://leetcode.cn/problems/maximum-subarray/description/
// 给你一个整数数组 nums ，请你找出一个具有最大和的连续子数组（子数组最少包含一个元素），返回其最大和。
func maxSubArray(nums []int) int {
	ans := math.MinInt
	//f[i]代表以nums[i]结尾的连续子数组最大和
	f := nums[0]
	for i, n := range nums {
		if i > 0 {
			f = max(n, n+f)
		}
		ans = max(ans, f)
	}
	return ans
}

// https://leetcode.cn/problems/find-the-substring-with-maximum-cost/
func maximumCostSubstring(s string, chars string, vals []int) int {
	//当前前缀和-最小前缀和=最大
	preSum := 0
	minPreSum := 0
	ans := math.MinInt
	costMap := map[int]int{}
	for i, c := range chars {
		costMap[int(c)] = vals[i]
	}
	cost := func(c int, costMap map[int]int) int {
		if v, ok := costMap[c]; ok {
			return v
		} else {
			return c - 96
		}
	}
	for _, c := range s {
		preSum += cost(int(c), costMap)
		minPreSum = min(minPreSum, preSum)
		ans = max(ans, preSum-minPreSum)
	}
	return ans
}

// https://leetcode.cn/problems/maximum-product-subarray/description/
// 给你一个整数数组 nums ，请你找出数组中乘积最大的非空连续 子数组 （该子数组中至少包含一个数字），并返回该子数组所对应的乘积。
func maxProduct(nums []int) int {
	maxProd := 1 //包含当前元素的最大积
	minProd := 1 //包含当前元素的最小积
	ans := math.MinInt
	for _, n := range nums {
		//不用管正负，比大小即可
		maxProd, minProd = max(maxProd*n, n, minProd*n), min(maxProd*n, n, minProd*n)
		ans = max(ans, maxProd)
	}
	return ans
}
