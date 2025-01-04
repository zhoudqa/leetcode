package golang

import "math"

type ListNode struct {
	Val  int
	Next *ListNode
}

func buildListNode(arr []int) *ListNode {
	head := &ListNode{
		Val:  arr[0],
		Next: nil,
	}
	pre := head
	for i := 1; i < len(arr); i++ {
		pre.Next = &ListNode{Val: arr[i]}
		pre = pre.Next
	}
	return head
}

type TreeNode struct {
	Val   int
	Left  *TreeNode
	Right *TreeNode
}

var null = math.MinInt

func buildTreeNode(nums []int) *TreeNode {
	if len(nums) == 0 || nums[0] == null {
		return nil
	}
	root := &TreeNode{Val: nums[0]}
	queue := []*TreeNode{root}
	i := 1
	for i < len(nums) && len(queue) > 0 {
		node := queue[0]
		queue = queue[1:]

		if nums[i] != null {
			node.Left = &TreeNode{Val: nums[i]}
			queue = append(queue, node.Left)
		}
		i++

		if i < len(nums) && nums[i] != null {
			node.Right = &TreeNode{Val: nums[i]}
			queue = append(queue, node.Right)
		}
		i++
	}
	return root
}
