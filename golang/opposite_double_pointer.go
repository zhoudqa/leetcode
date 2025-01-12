package golang

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

// 给定一个整数数组 temperatures ，表示每天的温度，返回一个数组 answer ，
// 其中 answer[i] 是指对于第 i 天，下一个更高温度出现在几天后。如果气温在这之后都不会升高，请在该位置用 0 来代替。
func dailyTemperatures(temperatures []int) []int {
	n := len(temperatures)
	res := make([]int, n)
	stack := make([]int, 0)
	for i := n - 1; i >= 0; i-- {
		cur := temperatures[i]
		for len(stack) != 0 && cur >= temperatures[stack[len(stack)-1]] {
			//删除后续的小的元素，下一次迭代最多只关心到cur
			stack = stack[:len(stack)-1]
		}
		if len(stack) != 0 {
			res[i] = stack[len(stack)-1] - i
		}
		stack = append(stack, i)
	}
	return res
}
