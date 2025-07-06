package golang

//😭😭😭😭😭😭😭😭😭😭😭😭😭😭😭😭😭😭😭😭😭二叉树😭😭😭😭😭😭😭😭😭😭😭😭😭😭😭😭😭😭😭😭//

func maxDepth(root *TreeNode) int {
	if root == nil {
		return 0
	}
	return 1 + max(maxDepth(root.Left), maxDepth(root.Right))
}

func isSameTree(p *TreeNode, q *TreeNode) bool {
	if p == nil && q == nil {
		return true
	}
	if p != nil && q != nil {
		return p.Val == q.Val && isSameTree(p.Left, q.Left) && isSameTree(p.Right, q.Right)
	}
	return false
}

func isSymmetricTree(p *TreeNode, q *TreeNode) bool {
	if p == nil && q == nil {
		return true
	}
	if p != nil && q != nil {
		return p.Val == q.Val && isSymmetricTree(p.Left, q.Right) && isSymmetricTree(p.Right, q.Left)
	}
	return false
}

func isSymmetric(root *TreeNode) bool {
	return isSymmetricTree(root.Left, root.Right)
}

func abs(del int) int {
	if del > 0 {
		return del
	} else {
		return -del
	}
}

func isBalanced(root *TreeNode) bool {
	var getHeight func(node *TreeNode) int
	getHeight = func(node *TreeNode) int {
		if node == nil {
			return 0
		}
		if leftHeight := getHeight(node.Left); leftHeight == -1 {
			return -1
		} else if rightHeight := getHeight(node.Right); rightHeight == -1 || abs(leftHeight-rightHeight) > 1 {
			return -1
		} else {
			return 1 + max(leftHeight, rightHeight)
		}
	}
	return getHeight(root) != -1
}

// 给定一个二叉树的 根节点 root，想象自己站在它的右侧，按照从顶部到底部的顺序，返回从右侧所能看到的节点值
func rightSideView(root *TreeNode) []int {
	var ans []int
	maxHeight := -1 //对应0个答案的ans，需要记录第0个节点
	var dfs func(node *TreeNode, height int)
	dfs = func(node *TreeNode, height int) {
		if node == nil {
			return
		}
		//从右往左遍历，如果高度（第一次）大于最大高度，那么就要记录并更新最大高度
		if height > maxHeight {
			ans = append(ans, node.Val)
			maxHeight = height
		}
		dfs(node.Right, height+1)
		dfs(node.Left, height+1)
	}
	dfs(root, 0)
	return ans
}

func lowestCommonAncestor(root, p, q *TreeNode) *TreeNode {
	if root == nil || root == p || root == q {
		return root
	}
	leftLCA := lowestCommonAncestor(root.Left, p, q)
	rightLCA := lowestCommonAncestor(root.Right, p, q)
	if leftLCA != nil && rightLCA != nil {
		return root
	} else if leftLCA != nil {
		return leftLCA
	} else {
		return rightLCA
	}
}

func lowestCommonAncestorForSearchBinaryTree(root, p, q *TreeNode) *TreeNode {
	val := root.Val
	if p.Val < val && q.Val < val {
		return lowestCommonAncestorForSearchBinaryTree(root.Left, p, q)
	} else if p.Val > val && q.Val > val {
		return lowestCommonAncestorForSearchBinaryTree(root.Right, p, q)
	} else {
		return root
	}
}

// 给定一个二叉树的 根节点 root，请找出该二叉树的 最底层 最左边 节点的值。
// 假设二叉树中至少有一个节点。
func findBottomLeftValue(root *TreeNode) int {
	var ans int
	queue := []*TreeNode{root}
	for queue != nil {
		var levelVals []int
		var nextQueue []*TreeNode
		for i := 0; i < len(queue); i++ {
			node := queue[i]
			if node.Right != nil {
				nextQueue = append(nextQueue, node.Right)
			}
			if node.Left != nil {
				nextQueue = append(nextQueue, node.Left)
			}
			levelVals = append(levelVals, node.Val)
		}
		ans = levelVals[len(levelVals)-1]
		queue = nextQueue
	}
	return ans
}
