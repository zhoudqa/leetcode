package golang

import "testing"

//😭😭😭😭😭😭😭😭😭😭😭😭😭😭😭😭😭😭😭😭😭链表反转😭😭😭😭😭😭😭😭😭😭😭😭😭😭😭😭😭😭😭😭//

func reverseList(head *ListNode) *ListNode {
	var pre *ListNode
	//cur表示当前遍历的节点，pre表示上次遍历的节点，当cur为nil时，pre表示最后一个节点，即反转后的头节点
	cur := head
	for cur != nil {
		next := cur.Next
		cur.Next = pre
		pre = cur
		cur = next
	}
	return pre
}

// 给你单链表的头指针 head 和两个整数 left 和 right ，其中 left <= right 。请你反转从位置 left 到位置 right 的链表节点，返回 反转后的链表 。
// https://leetcode.cn/problems/reverse-linked-list-ii/description/
func reverseBetween(head *ListNode, left int, right int) *ListNode {
	dummy := &ListNode{
		Val:  0,
		Next: head,
	}
	//p0代表开始旋转的节点前的节点，需要到left的左边，所以右移left-1
	p0 := dummy
	for i := 0; i < left-1; i++ {
		p0 = p0.Next
	}
	var pre *ListNode
	cur := p0.Next
	for i := left; i <= right; i++ {
		next := cur.Next
		cur.Next = pre
		pre = cur
		cur = next
	}
	//p0.Next还是指向第一个旋转的节点，现在到了最后面，cur到了旋转尾节点后一个节点，把这2个相连
	p0.Next.Next = cur
	//把p0.Next指向最后一个旋转的节点
	p0.Next = pre
	return dummy.Next
}

// 给你链表的头节点 head ，每 k 个节点一组进行翻转，请你返回修改后的链表。
// k 是一个正整数，它的值小于或等于链表的长度。如果节点总数不是 k 的整数倍，那么请将最后剩余的节点保持原有顺序。
// 你不能只是单纯的改变节点内部的值，而是需要实际进行节点交换。
// https://leetcode.cn/problems/reverse-nodes-in-k-group/description/
func reverseKGroup(head *ListNode, k int) *ListNode {
	tmp := head
	n := 0
	for tmp != nil {
		n++
		tmp = tmp.Next
	}

	dummy := &ListNode{
		Val:  0,
		Next: head,
	}
	//p0代表开始旋转的节点前的节点
	p0 := dummy
	for n >= k {
		n -= k
		var pre *ListNode
		cur := p0.Next
		for i := 0; i < k; i++ {
			next := cur.Next
			cur.Next = pre
			pre = cur
			cur = next
		}
		//下一段的p0就是这一段p0的Next=> dummy(p0)->1->2->3->4->5[2] ==> 2->1(p0)->3->4->5[2]
		nextP0 := p0.Next
		//p0.Next还是指向第一个旋转的节点，现在到了最后面，cur到了旋转伟节点后一个节点，把这2个相连
		p0.Next.Next = cur
		//把p0.Next指向最后一个旋转的节点
		p0.Next = pre
		p0 = nextP0
	}
	return dummy.Next
}

//😭😭😭😭😭😭😭😭😭😭😭😭😭😭😭😭😭😭😭😭😭环形链表(快慢指针)😭😭😭😭😭😭😭😭😭😭😭😭😭😭😭😭😭😭😭😭//

// 给你单链表的头结点 head ，请你找出并返回链表的中间结点。
//
// 如果有两个中间结点，则返回第二个中间结点。
func middleNode(head *ListNode) *ListNode {
	slow := head
	fast := head
	for fast != nil && fast.Next != nil {
		slow = slow.Next
		fast = fast.Next.Next
	}
	return slow
}

// 给你一个链表的头节点 head ，判断链表中是否有环。
//
// 如果链表中有某个节点，可以通过连续跟踪 next 指针再次到达，则链表中存在环。 为了表示给定链表中的环，评测系统内部使用整数 pos 来表示链表尾连接到链表中的位置（索引从 0 开始）。注意：pos 不作为参数进行传递 。仅仅是为了标识链表的实际情况。
//
// 如果链表中存在环 ，则返回 true 。 否则，返回 false
func hasCycle(head *ListNode) bool {
	slow := head
	fast := head
	for fast != nil && fast.Next != nil {
		slow = slow.Next
		fast = fast.Next.Next
		if slow == fast {
			return true
		}
	}
	return false
}

// 给定一个链表的头节点  head ，返回链表开始入环的第一个节点。 如果链表无环，则返回 null。
// 如果链表中有某个节点，可以通过连续跟踪 next 指针再次到达，则链表中存在环。 为了表示给定链表中的环，评测系统内部使用整数 pos 来表示链表尾连接到链表中的位置（索引从 0 开始）。如果 pos 是 -1，则在该链表中没有环。注意：pos 不作为参数进行传递，仅仅是为了标识链表的实际情况。
// 不允许修改 链表。
// https://leetcode.cn/problems/linked-list-cycle-ii/description/
func detectCycle(head *ListNode) *ListNode {
	// 设head到入环的第一个节点长度是a a到快慢指针相遇点为b b到a长度为c
	// 相遇时慢指针走过长度为a+b 快指针走过长度为a+b+k(c+b) 且快指针长度是慢指针的2倍可以得到 2(a+b)=a+b+k(c+b)
	// 从而可以推导出 a-c = (k-1)(b+c) 当慢指针从相遇点出发，head从首节点出发时，慢指针到达入环点时，head离相遇点还有a-c的距离
	// 所以慢指针继续转圈k-1圈，一定会和head相遇
	slow := head
	fast := head
	for fast != nil && fast.Next != nil {
		slow = slow.Next
		fast = fast.Next.Next
		if slow == fast {
			for slow != head {
				head = head.Next
				slow = slow.Next
			}
			return head
		}
	}
	return nil
}

// https://leetcode.cn/problems/reorder-list/description/
func reorderList(head *ListNode) {
	//1,2,3,4,5 => 1,2 5,4,3 合并这2个链表即可，循环出口在head2.Next为nil
	//head2代表反转后遍历节点
	head2 := reverseList(middleNode(head))
	for head2.Next != nil {
		nextHead := head.Next
		nextHead2 := head2.Next
		head.Next = head2
		head2.Next = nextHead
		head = nextHead
		head2 = nextHead2
	}
}

//😭😭😭😭😭😭😭😭😭😭😭😭😭😭😭😭😭😭😭😭😭链表(前后指针)😭😭😭😭😭😭😭😭😭😭😭😭😭😭😭😭😭😭😭😭//

func deleteNode(node *ListNode) {
	node.Val = node.Next.Val
	node.Next = node.Next.Next
}

func deleteDuplicates(head *ListNode) *ListNode {
	if head == nil {
		return head
	}
	cur := head
	for cur.Next != nil {
		if cur.Next.Val == cur.Val {
			cur.Next = cur.Next.Next
		} else {
			cur = cur.Next
		}
	}
	return head
}

func TestDeleteDuplicateds2(t *testing.T) {
	head := buildListNode([]int{1, 1})
	deleteDuplicates2(head)
}

// 给定一个已排序的链表的头 head ， 删除原始链表中所有重复数字的节点，只留下不同的数字 。返回 已排序的链表 。
func deleteDuplicates2(head *ListNode) *ListNode {
	if head == nil {
		return head
	}
	dummyHead := &ListNode{Next: head}
	cur := dummyHead
	//只用关心下一个个下下个节点的值就行，因为有dummpy去兜底
	for cur.Next != nil && cur.Next.Next != nil {
		nextVal := cur.Next.Val
		if cur.Next.Next.Val == nextVal {
			//必须判断，只有满足这个条件，才能while循环把cur后面重复元素都删掉，不然先删除后会不重复
			for cur.Next != nil && cur.Next.Val == nextVal {
				cur.Next = cur.Next.Next
			}
		} else {
			cur = cur.Next
		}
	}
	return dummyHead.Next
}

// 给你一个链表，删除链表的倒数第 n 个结点，并且返回链表的头结点。
func removeNthFromEnd(head *ListNode, n int) *ListNode {
	dumpyHead := &ListNode{Next: head}
	end := dumpyHead
	for i := 0; i < n; i++ {
		end = end.Next
	}
	start := dumpyHead
	for end.Next != nil {
		start = start.Next
		end = end.Next
	}
	start.Next = start.Next.Next
	return dumpyHead.Next
}

// 给你一个链表，两两交换其中相邻的节点，并返回交换后链表的头节点。你必须在不修改节点内部的值的情况下完成本题（即，只能进行节点交换）。
func swapPairs(head *ListNode) *ListNode {
	dummyHead := &ListNode{Next: head}
	p0 := dummyHead
	for p0.Next != nil && p0.Next.Next != nil {
		cur1 := p0.Next
		cur2 := p0.Next.Next
		nextP0 := cur1
		p0.Next = cur2
		cur1.Next = cur2.Next
		cur2.Next = cur1
		p0 = nextP0
	}
	return dummyHead.Next
}
