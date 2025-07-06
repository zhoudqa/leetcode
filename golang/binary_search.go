package golang

import (
	"math"
	"slices"
)

// isBlue代表可能有更好的解，这个解也符合条件，不断循环直到不满足则可以得到一个符合条件的最好的解，在right指针上

// 😭😭😭😭😭😭😭😭😭😭😭😭😭😭😭😭😭😭😭😭二分查找😭😭😭😭😭😭😭😭😭😭😭😭😭😭😭😭😭😭😭😭😭//

// 给你一个按照非递减顺序排列的整数数组 nums，和一个目标值 target。请你找出给定目标值在数组中的开始位置和结束位置。
//
// 如果数组中不存在目标值 target，返回 [-1, -1]。
func searchRange(nums []int, target int) []int {
	blueIndex := minBlueIndex(nums, func(num int) bool {
		return num >= target
	})
	if blueIndex == len(nums) || nums[blueIndex] != target {
		return []int{-1, -1}
	}
	return []int{blueIndex, minBlueIndex(nums, func(num int) bool {
		return num > target+1
	}) - 1}
}

// 返回的index代表最小的满足条件的坐标，index-1则是最大的满足<target的坐标
func minBlueIndex(nums []int, isBlue func(num int) bool) int {
	n := len(nums)
	//开区间(-1,n)->[0,n-1]
	left := -1
	right := n //没有满足条件的时候 返回数组长度
	for left+1 < right {
		mid := (left + right) / 2
		if isBlue(nums[mid]) {
			//继续往左找更小的满足条件的
			right = mid
		} else {
			left = mid
		}
	}
	// 循环结束后 left+1 = right
	// 此时 nums[left] < target 而 nums[right] >= target
	// 所以 right 就是第一个 >= target 的元素下标
	return right
}

// https://leetcode.cn/problems/koko-eating-bananas/description/
func minEatingSpeed(piles []int, h int) int {
	//找到一个最小的满足可以吃完香蕉的速度
	left := 0                  //红色
	right := slices.Max(piles) //最大的蓝色
	isBlue := func(speed int) bool {
		//用speed吃香蕉可以吃完的条件
		curH := 0
		for _, pile := range piles {
			//p/speed向上取整
			curH += (pile-1)/speed + 1
		}
		return curH <= h
	}
	for left+1 < right {
		mid := (left + right) / 2
		if isBlue(mid) {
			right = mid
		} else {
			left = mid
		}
	}
	return right
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
	isBlue := func(idx int) bool {
		return nums[idx] < nums[l-1]
	}
	for left+1 < right {
		mid := (left + right) / 2
		if isBlue(mid) {
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

// 给你一个整数数组 start 和一个整数 d，代表 n 个区间 [start[i], start[i] + d]。
// 你需要选择 n 个整数，其中第 i 个整数必须属于第 i 个区间。所选整数的 得分 定义为所选整数两两之间的 最小 绝对差。
// 返回所选整数的 最大可能得分 。
// https://leetcode.cn/problems/maximize-score-of-numbers-in-ranges/description/
func maxPossibleScore(start []int, d int) int {
	//找到一个得分，为最大的满足条件的得分，如果要满足最小绝对差最大，那么俩俩之间的绝对差都得相等，也即是这里的score
	slices.Sort(start)
	n := len(start)
	isBlue := func(starts []int, score int) bool {
		prex := math.MinInt //保证第一个选的是左区间
		for _, start := range starts {
			//前面的start加上得分比当前的右区间大，不符合条件
			lastX := prex + score
			if lastX > start+d {
				return false
			}
			prex = max(start, lastX)
		}
		return true
	}
	//得分为0肯定是满足条件的，但是不是最大得分，右区间的边界要满足 score <= (s[n-1]+d-s[0])/(n-1)
	//这里取开区间，left+1=right时为空区间
	left, right := -1, (start[n-1]+d-start[0])/(n-1)+1
	for left+1 < right {
		mid := (left + right) / 2
		if isBlue(start, mid) {
			left = mid
		} else {
			right = mid
		}
	}
	return left
}
