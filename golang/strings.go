package golang

import "math"

// 给你两个下标从 0 开始的字符串 s 和 target 。你可以从 s 取出一些字符并将其重排，得到若干新的字符串。
//
// 从 s 中取出字符并重新排列，返回可以形成 target 的 最大 副本数。
func rearrangeCharacters(s string, target string) int {
	m := map[int32]int{}
	for _, c := range target {
		m[c]++
	}
	sm := map[int32]int{}
	for _, c := range s {
		if _, ok := m[c]; ok {
			sm[c]++
		}
	}
	res := math.MaxInt
	for c, l := range m {
		times := sm[c] / l
		res = min(res, times)
	}
	return res
}
