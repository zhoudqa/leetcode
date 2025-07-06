package golang

import "math"

// 给你一个二进制字符串 s ，现需要将其转化为一个 交替字符串 。请你计算并返回转化所需的 最小 字符交换次数，如果无法完成转化，返回 -1 。
// 交替字符串 是指：相邻字符之间不存在相等情况的字符串。例如，字符串 "010" 和 "1010" 属于交替字符串，但 "0100" 不是。
// 任意两个字符都可以进行交换，不必相邻 。
// https://leetcode.cn/problems/minimum-number-of-swaps-to-make-the-binary-string-alternating/description/
func minSwaps(s string) int {
	var zeros, ones int
	for _, c := range s {
		if c == '0' {
			zeros++
		} else {
			ones++
		}
	}
	if math.Abs(float64(zeros-ones)) > 1 {
		//数量差大于1，不能交换成功
		return -1
	}
	countSwaps := func(str string, head int32) int {
		res := 0
		for i, c := range str {
			//c和head一样且是偶数位 或者 c和head不一样且是奇数位 不需要交换
			if !(i%2 == 0 && c == head) && !(i%2 == 1 && c != head) {
				res++
			}
		}
		return res / 2
	}
	if zeros == ones {
		//需要考虑0开头和1开头 哪种方案优
		return min(countSwaps(s, '0'), countSwaps(s, '1'))
	} else {
		//取多的符号开头的肯定交换次数少
		start := '1'
		if zeros > ones {
			start = '0'
		}
		return countSwaps(s, start)
	}
}
