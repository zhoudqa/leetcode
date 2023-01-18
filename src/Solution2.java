import annotations.algorithm.DynamicPrograming;
import annotations.algorithm.PrefixSum;
import annotations.algorithm.SlidingWindow;
import annotations.level.Hard;
import java.util.*;

public class Solution2 extends SolutionBase {

    //给定一个含有 n 个正整数的数组和一个正整数 target 。
    //找出该数组中满足其和 ≥ target 的长度最小的 连续子数组，并返回其长度。如果不存在符合条件的子数组，返回 0 。
    @SlidingWindow
    public int minSubArrayLen(int target, int[] nums) {
        int start = 0;
        int end = 0;
        int sum = 0;
        int res = Integer.MAX_VALUE;
        while (end < nums.length) {
            sum += nums[end];
            while (sum >= target) {
                //记录满足条件的result
                res = Math.min(res, end - start + 1);
                sum -= nums[start];
                start++;
            }
            end++;
        }
        return res == Integer.MAX_VALUE ? 0 : res;

    }

    //给定一个正整数数组 nums和整数 k ，请找出该数组内乘积小于 k 的连续子数组的个数。
    @SlidingWindow
    public int numSubarrayProductLessThanK(int[] nums, int k) {
        int start = 0;
        int end = 0;
        int prod = 1;
        int res = 0;
        while (end < nums.length) {
            prod *= nums[end];
            //边界start=end，prod=1
            while (start <= end && prod >= k) {
                prod /= nums[start];
                start++;
            }
            res += end - start + 1;
            end++;
        }
        return res;

    }

    //给定一个字符串 s ，请你找出其中不含有重复字符的 最长连续子字符串 的长度。
    @SlidingWindow
    public int lengthOfLongestSubstring(String s) {
        int res = 0;
        if (s == null || s.isEmpty()) {
            return 0;
        }
        Set<Character> set = new HashSet<>();
        int start = 0, end = 0;
        final int len = s.length();
        while (start < len) {
            if (start != 0) {
                //新一轮遍历，去掉之前的字符
                set.remove(s.charAt(start - 1));
            }
            while (end < len && !set.contains(s.charAt(end))) {
                set.add(s.charAt(end));
                end++;
            }
            //end后++了，因此为实际最后一个不重复的字符索引(end')+1，长度为end'- start + 1 = end - start
            res = Math.max(res, end - start++);
        }
        return res;

    }

    //将insertVal插入循环链表中
    public Node insert(Node head, int insertVal) {
        final Node insertNode = new Node(insertVal);
        if (head == null) {
            insertNode.next = insertNode;
            return insertNode;
        }
        if (head.next == head) {
            head.next = insertNode;
            insertNode.next = head;
            return head;
        }
        Node cur = head, next = head.next;
        while (next != head) {
            // 3->5->1   4
            if (cur.val <= insertVal && next.val >= insertVal) {
                break;
            }
            //1->2->3 4 || 1->3->5 0
            //反例 3->4->1 2
            if (cur.val > next.val && (insertVal > cur.val || insertVal < next.val)) {
                break;
            }
            cur = next;
            next = next.next;
        }
        cur.next = insertNode;
        insertNode.next = next;
        return head;
    }

    //将2个链表按低位相加
    public ListNode addTwoNumbers(ListNode l1, ListNode l2) {
        Deque<Integer> stack1 = new LinkedList<>();
        Deque<Integer> stack2 = new LinkedList<>();
        while (l1 != null) {
            stack1.push(l1.val);
            l1 = l1.next;
        }
        while (l2 != null) {
            stack2.push(l2.val);
            l2 = l2.next;
        }
        //保留进位
        int jin = 0;
        ListNode head = null;
        while (!stack1.isEmpty() || !stack2.isEmpty() || jin != 0) {
            int a = stack1.isEmpty() ? 0 : stack1.pop();
            int b = stack2.isEmpty() ? 0 : stack2.pop();
            int sum = a + b + jin;
            jin = sum / 10;
            final ListNode cur = new ListNode(sum % 10);
            cur.next = head;
            head = cur;
        }
        return head;

    }

    /**
     * 给定一个整数数组和一个整数 k ，请找到该数组中和为 k 的连续子数组的个数。<br/>
     * <a href='https://leetcode.cn/problems/QTMn0o/'>剑指 Offer II 010. 和为 k 的子数组</a>
     */
    @PrefixSum
    public int subarraySum(int[] nums, int k) {
        int res = 0, sum = 0;
        //key 前缀和,value为该前缀和的个数
        Map<Integer, Integer> preMap = new HashMap<>();
        preMap.put(0, 1);//初始化0前缀和为1种
        //问题转换为num中有多少个连续子数组[i...j]满足pre(i)+k=pre(j) 0<=i<=j<n
        for (final int num : nums) {
            sum += num;
            if (preMap.containsKey(sum - k)) {
                res += preMap.get(sum - k);
            }
            //保存前缀和
            preMap.merge(sum, 1, Integer::sum);
        }
        return res;
    }

    /**
     * 给定两个字符串 s 和 t 。返回 s 中包含 t 的所有字符的最短子字符串。如果 s 中不存在符合条件的子字符串，则返回空字符串 "" 。
     * 如果 s 中存在多个符合条件的子字符串，返回任意一个。
     * 注意： 对于 t 中重复字符，我们寻找的子字符串中该字符数量必须不少于 t 中该字符数量。<br/>
     * <a href='https://leetcode.cn/problems/M1oyTv/'>剑指 Offer II 017. 含有所有字符的最短字符串</a>
     */
    @Hard
    public String minWindow(String s, String t) {
        return null;
    }

    /**
     * 给定一个字符串 s ，请计算这个字符串中有多少个回文子字符串。
     * 具有不同开始位置或结束位置的子串，即使是由相同的字符组成，也会被视作不同的子串。<br/>
     * <a href='https://leetcode.cn/problems/a7VOhD/'>剑指 Offer II 020. 回文子字符串的个数</a>
     */
    @DynamicPrograming
    public int countSubstrings(String s) {
        if (s == null || s.isEmpty()) {
            return 0;
        }
        final int len = s.length();
        boolean[][] dp = new boolean[len][len];
        int res = 0;
        for (int i = len - 1; i >= 0; i--) {
            for (int j = i; j < len; j++) {
                if (i == j) {
                    dp[i][j] = true;
                } else if (s.charAt(i) == s.charAt(j)) {
                    //长度为2则直接为true
                    dp[i][j] = j - i + 1 < 3 || dp[i + 1][j - 1];
                }
                if (dp[i][j]) {
                    res++;
                }
            }
        }

        return res;
    }


    public static void main(String[] args) {
        final Solution2 solution = new Solution2();
//        solution.numSubarrayProductLessThanK(new int[]{10}, 2);
//        solution.addTwoNumbers(stringToListNode("[7,2,4,3]"), stringToListNode("[5,6,4]"));
//        solution.subarraySum(stringToArray("[1,1,1]"), 2);
        solution.countSubstrings("aba");
    }

}
