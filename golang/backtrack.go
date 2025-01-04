package golang

// 😭😭😭😭😭😭😭😭😭😭😭😭😭😭😭😭😭😭😭😭😭回溯😭😭😭😭😭😭😭😭😭😭😭😭😭😭😭😭😭😭😭😭//

// 给你二叉树的根节点 root 和一个整数目标和 targetSum ，找出所有 从根节点到叶子节点 路径总和等于给定目标和的路径。
func pathSum(root *TreeNode, targetSum int) [][]int {
	var ans [][]int
	if root == nil {
		return ans
	}

	var backtrack func(node *TreeNode, targetSum int, path []int)
	backtrack = func(node *TreeNode, targetSum int, path []int) {
		if node == nil {
			return
		}
		if targetSum == 0 && node.Left == nil && node.Right == nil {
			temp := make([]int, len(path))
			copy(temp, path)
			ans = append(ans, temp)
		}
		if node.Left != nil {
			val := node.Left.Val
			path = append(path, val)
			backtrack(node.Left, targetSum-val, path)
			path = path[:len(path)-1]
		}
		if node.Right != nil {
			val := node.Right.Val
			path = append(path, val)
			backtrack(node.Right, targetSum-val, path)
			path = path[:len(path)-1]
		}
	}
	backtrack(root, targetSum-root.Val, []int{root.Val})
	return ans
}

// 给你一个整数数组 nums ，数组中的元素 互不相同 。返回该数组所有可能的
// 子集
// （幂集）。
//
// 解集 不能 包含重复的子集。你可以按 任意顺序 返回解集。
func subsets(nums []int) [][]int {
	var backtrack func(i int)
	ans := [][]int{}
	path := []int{}
	backtrack = func(idx int) {
		if idx == len(nums) {
			temp := make([]int, len(path))
			copy(temp, path)
			ans = append(ans, temp)
			return
		}
		backtrack(idx + 1)
		path = append(path, nums[idx])
		backtrack(idx + 1)
		path = path[:len(path)-1]
	}
	backtrack(0)
	return ans
}

// 给你一个字符串 s，请你将 s 分割成一些子串，使每个子串都是
// 回文串 。返回 s 所有可能的分割方案。
func partition(s string) [][]string {
	var ans [][]string
	var path []string
	n := len(s)
	var backtrack func(i int)
	isPalindrome := func(str string) bool {
		l := 0
		r := len(str) - 1
		for l <= r {
			if str[l] != str[r] {
				return false
			}
			l++
			r--
		}
		return true
	}
	backtrack = func(i int) {
		//i代表前面路径中所有的字符串都是回文子串了
		if i == n {
			temp := make([]string, len(path))
			copy(temp, path)
			ans = append(ans, temp)
			return
		}
		for j := i; j < n; j++ {
			newCut := s[i : j+1]
			if isPalindrome(newCut) {
				//从i到j的子串是回文，从j+1开始继续判断，直到选完
				path = append(path, newCut)
				backtrack(j + 1)
				path = path[:len(path)-1]
			}
		}
	}
	backtrack(0)
	return ans
}

// 给定两个整数 n 和 k，返回范围 [1, n] 中所有可能的 k 个数的组合。
func combine(n int, k int) [][]int {
	var ans [][]int
	var path []int
	var backtrack func(idx int)
	backtrack = func(i int) {

		left := k - len(path)
		if left == 0 {
			ans = append(ans, append([]int(nil), path...))
			return
		}
		//不选，前提是剩下的数比还需要的数多，边界是i=1 left=1，必须选，这里不能=
		if i > left {
			backtrack(i - 1)
		}
		//选
		path = append(path, i)
		backtrack(i - 1)
		//恢复
		path = path[:len(path)-1]
	}
	backtrack(n)
	return ans
}

// 找出所有相加之和为 n 的 k 个数的组合，且满足下列条件：
// 只使用数字1到9
// 每个数字 最多使用一次
// 返回 所有可能的有效组合的列表 。该列表不能包含相同的组合两次，组合可以以任何顺序返回。
func combinationSum3(k int, n int) [][]int {
	var ans [][]int
	var path []int
	var backtrack func(idx, t int)
	backtrack = func(i, t int) {

		left := k - len(path)
		if left == 0 {
			if t == 0 {
				ans = append(ans, append([]int(nil), path...))
			}
			return
		}
		//不选，前提是剩下的数比还需要的数多，边界是i=1 left=1，必须选，这里不能=
		if i > left {
			backtrack(i-1, t)
		}
		//选
		path = append(path, i)
		backtrack(i-1, t-i)
		//恢复
		path = path[:len(path)-1]
	}
	backtrack(9, n)
	return ans
}

func generateParenthesis(n int) []string {
	var path string
	var ans []string
	var backtrack func(i, open int)
	//open代表左括号个数，i代表当前字符串长度
	backtrack = func(i, open int) {
		if len(path) == n*2 {
			ans = append(ans, path)
			return
		}
		if open < n {
			path += "("
			backtrack(i+1, open+1)
			path = path[:len(path)-1]
		}
		//当前长度-左括号的长度就是右括号的最大长度
		if i-open < open {
			path += ")"
			backtrack(i+1, open)
			path = path[:len(path)-1]
		}

	}
	backtrack(0, 0)
	return ans
}
