package main

import (
	"sort"
	"testing"
)

func max(a, b int) int {
	if a > b {
		return a
	} else {
		return b
	}
}

func min(a, b int) int {
	if a > b {
		return b
	} else {
		return a
	}
}

// 2396. 严格回文的数字
// 如果一个整数 n 在 b 进制下（b 为 2 到 n - 2 之间的所有整数）对应的字符串 全部 都是 回文的 ，那么我们称这个数 n 是 严格回文 的。
// 给你一个整数 n ，如果 n 是 严格回文 的，请返回 true ，否则返回 false 。
// 如果一个字符串从前往后读和从后往前读完全相同，那么这个字符串是 回文的 。
// 代数求余 n=qb+r n>4时，取b=n-2，则q=1,r=2，那么n-2进制表达就是12，不是回文，当n=4，二进制是100，不是回文
func isStrictlyPalindromic(n int) bool {
	return false
}

func TestLC(t *testing.T) {
	minSubArrayLen(4, []int{1, 4, 4})
}

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

//😭😭😭😭😭😭😭😭😭😭😭😭😭😭😭😭😭😭😭😭双指针😭😭😭😭😭😭😭😭😭😭😭😭😭😭😭😭😭😭😭😭//

// 给定一个含有 n 个正整数的数组和一个正整数 target 。
//
// 找出该数组中满足其总和大于等于 target 的长度最小的 子数组
// [numsl, numsl+1, ..., numsr-1, numsr] ，并返回其长度。如果不存在符合条件的子数组，返回 0 。
func minSubArrayLen(target int, nums []int) int {
	//通过while循环让左指针不断向右移动寻找最小长度（前提是当前窗口和>=target了）
	n := len(nums)
	left := 0
	right := 0
	ans := n + 1
	sum := 0
	for ; right < n; right++ {
		sum += nums[right]
		for sum >= target {
			ans = min(ans, right-left+1)
			sum -= nums[left]
			left++
		}
	}
	if ans == n+1 {
		return 0
	}
	return ans
}

// 给你一个整数数组 nums 和一个整数 k ，请你返回子数组内所有元素的乘积严格小于 k 的连续子数组的数目。
func numSubarrayProductLessThanK(nums []int, k int) int {
	if k <= 1 {
		return 0
	}
	ans := 0
	prod := 1
	var left int
	for right, num := range nums {
		prod *= num
		for prod >= k {
			prod /= nums[left]
			left++
		}
		//元素个数为[l...r]包含r的连续子数组个数，即[l,r],[l+1,r]...[r,r]，等于l到r的长度
		ans += right - left + 1
	}
	return ans
}

// 给定一个字符串 s ，请你找出其中不含有重复字符的 最长 子串 的长度。
func lengthOfLongestSubstring(s string) int {
	ans := 0
	cntMap := map[int]int{}
	left := 0
	for right, char := range s {
		cntMap[int(char)]++
		//右指针向右之后，可能重复的只有右指针这个字符的重复，
		for cntMap[int(char)] > 1 {
			cntMap[int(s[left])]--
			left++
		}
		ans = max(ans, right-left+1)
	}
	return ans
}

//😭😭😭😭😭😭😭😭😭😭😭😭😭😭😭😭😭😭😭😭😭相向双指针😭😭😭😭😭😭😭😭😭😭😭😭😭😭😭😭😭😭😭😭//

// 给定 n 个非负整数表示每个宽度为 1 的柱子的高度图，计算按此排列的柱子，下雨之后能接多少雨水。
func trap(height []int) int {
	//思路：当前柱子能接的雨水取决于前缀高度最大值和后缀高度最大值小的-当前柱子高度
	//遍历3次 O(N)时间复杂度 O(N)空间复杂度
	n := len(height)
	preMax := make([]int, n)
	sufMax := make([]int, n)
	var temp int
	for i := 0; i < n; i++ {
		temp = max(temp, height[i])
		preMax[i] = temp
	}
	temp = 0
	for i := n - 1; i >= 0; i-- {
		temp = max(temp, height[i])
		sufMax[i] = temp
	}
	ans := 0
	for i := 0; i < n; i++ {
		ans += min(preMax[i], sufMax[i]) - height[i]
	}
	return ans
}

func trapSpaceO1(height []int) int {
	//思路：如果前缀最大值比后缀最大值小了，那么无论后缀继续往前移动多少，后缀最大值一定比前缀最大值还大，
	//当前乘的水就可以计算出来了，无需中间的数组，那么可以双指针相向移动
	n := len(height)
	preMax := height[0]
	sufMax := height[n-1]
	left := 0
	right := n - 1
	ans := 0
	//临界条件，相等的时候计算对应那块木板接的单位
	for left <= right {
		preMax = max(preMax, height[left])
		sufMax = max(sufMax, height[right])
		if preMax < sufMax {
			ans += preMax - height[left]
			left++
		} else {
			ans += sufMax - height[right]
			right--
		}
	}
	return ans
}

//😭😭😭😭😭😭😭😭😭😭😭😭😭😭😭😭😭😭😭😭😭二分查找#旋转数组😭😭😭😭😭😭😭😭😭😭😭😭😭😭😭😭😭😭😭😭//

// 峰值元素是指其值严格大于左右相邻值的元素。
// 给你一个整数数组 nums，找到峰值元素并返回其索引。数组可能包含多个峰值，在这种情况下，返回 任何一个峰值 所在位置即可。
func findPeakElement(nums []int) int {
	l := len(nums)
	//[0,n-2]->(-1,n-1)
	left := -1
	right := l - 1
	for left+1 < right {
		mid := (left + right) / 2
		if nums[mid] > nums[mid+1] {
			//蓝色，代表峰顶或者峰顶右边的元素
			right = mid
		} else {
			//红色，代表峰顶左边
			left = mid
		}
	}
	//返回最左边的蓝色
	return right
}

// 已知一个长度为 n 的数组，预先按照升序排列，经由 1 到 n 次 旋转 后，得到输入数组。例如，原数组 nums = [0,1,2,4,5,6,7] 在变化后可能得到：
// 若旋转 4 次，则可以得到 [4,5,6,7,0,1,2]
// 若旋转 7 次，则可以得到 [0,1,2,4,5,6,7]
// 注意，数组 [a[0], a[1], a[2], ..., a[n-1]] 旋转一次 的结果为数组 [a[n-1], a[0], a[1], a[2], ..., a[n-2]] 。
//
// 给你一个元素值 互不相同 的数组 nums ，它原来是一个升序排列的数组，并按上述情形进行了多次旋转。请你找出并返回数组中的 最小元素 。
func findMin(nums []int) int {
	l := len(nums)
	//[0,n-2]->(-1,n-1)
	left := -1
	right := l - 1
	for left+1 < right {
		mid := (left + right) / 2
		if nums[mid] < nums[l-1] {
			//蓝色，(mid,l-1)有序，mid是右区间最小的
			right = mid
		} else {
			//红色，在右边V字的区间
			left = mid
		}
	}
	//返回最左边的蓝色
	return nums[right]
}

// 整数数组 nums 按升序排列，数组中的值 互不相同 。
//
// 在传递给函数之前，nums 在预先未知的某个下标 k（0 <= k < nums.length）上进行了 旋转，使数组变为 [nums[k], nums[k+1], ..., nums[n-1], nums[0], nums[1], ..., nums[k-1]]（下标 从 0 开始 计数）。例如， [0,1,2,4,5,6,7] 在下标 3 处经旋转后可能变为 [4,5,6,7,0,1,2] 。
//
// 给你 旋转后 的数组 nums 和一个整数 target ，如果 nums 中存在这个目标值 target ，则返回它的下标，否则返回 -1 。
func search(nums []int, target int) int {
	l := len(nums)
	//[0,n-2]->(-1,n-1)
	left := -1
	right := l - 1
	//蓝色代表可以确定往i的左边找数，不考虑右边的了
	isBlue := func(i int) bool {
		end := nums[len(nums)-1]
		if nums[i] > end {
			//旋转后右边分成2段，往左的条件是target在(left(>end),nums[i]之间
			return target > end && target <= nums[i]
		} else {
			//左边分成2段，那么往左找的条件就是target在这2段上
			return target > end || target <= nums[i]
		}
	}
	for left+1 < right {
		mid := (left + right) / 2
		if isBlue(mid) {
			//蓝色
			right = mid
		} else {
			//红色
			left = mid
		}
	}
	if nums[right] != target {
		return -1
	}
	//返回最左边的蓝色
	return right
}

// 一个 2D 网格中的 峰值 是指那些 严格大于 其相邻格子(上、下、左、右)的元素。
// 给你一个 从 0 开始编号 的 m x n 矩阵 mat ，其中任意两个相邻格子的值都 不相同 。找出 任意一个 峰值 mat[i][j] 并 返回其位置 [i,j] 。
func findPeakGrid(mat [][]int) []int {
	left := -1
	right := len(mat) - 1
	findMaxIndex := func(arr []int) int {
		res := 0
		for i, num := range arr {
			if arr[res] < num {
				res = i
			}
		}
		return res
	}
	for left+1 < right {
		mid := (left + right) / 2
		//找到一行的最大值，峰值要么是这个点，要么在上下，满足单调性
		maxIndex := findMaxIndex(mat[mid])
		if mat[mid][maxIndex] < mat[mid+1][maxIndex] {
			//在峰值左边
			left = mid
		} else {
			right = mid
		}
	}
	return []int{right, findMaxIndex(mat[right])}

}
