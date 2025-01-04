package golang

//😭😭😭😭😭😭😭😭😭😭😭😭😭😭😭😭😭😭😭😭滑动窗口😭😭😭😭😭😭😭😭😭😭😭😭😭😭😭😭😭😭😭😭//

// 给定一个含有 n 个正整数的数组和一个正整数 target 。
// https://leetcode.cn/problems/minimum-size-subarray-sum/description/
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

// 给你一个整数数组 nums 和一个整数 k 。
// 一个元素 x 在数组中的 频率 指的是它在数组中的出现次数。
// 如果一个数组中所有元素的频率都 小于等于 k ，那么我们称这个数组是 好 数组。
// 请你返回 nums 中 最长好 子数组的长度。
// 子数组 指的是一个数组中一段连续非空的元素序列
func maxSubarrayLength(nums []int, k int) int {
	ans := 0
	cntMap := map[int]int{}
	left := 0
	for right, num := range nums {
		cntMap[num]++
		for cntMap[num] > k {
			cntMap[nums[left]]--
			left++
		}
		ans = max(ans, right-left+1)
	}
	return ans
}

// 给你一个下标从 0 开始的字符串 s ，这个字符串只包含 0 到 9 的数字字符。
// 如果一个字符串 t 中至多有一对相邻字符是相等的，那么称这个字符串 t 是 半重复的 。
// 例如，"0010" 、"002020" 、"0123" 、"2002" 和 "54944" 是半重复字符串，而 "00101022" （相邻的相同数字对是 00 和 22）和 "1101234883" （相邻的相同数字对是 11 和 88）不是半重复字符串。
// 请你返回 s 中最长 半重复 子字符串 的长度。
func longestSemiRepetitiveSubstring(s string) int {
	ans := 0
	left := 0
	repeatCnt := 0
	for right, char := range s {
		if right > 0 && uint8(char) == s[right-1] {
			repeatCnt++
		}
		for repeatCnt > 1 {
			if s[left] == s[left+1] {
				repeatCnt--
			}
			left++
		}
		ans = max(ans, right-left+1)
	}
	return ans
}
