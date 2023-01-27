import annotations.algorithm.DFS;
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

    /**
     * 给你一个 无重复元素 的整数数组 candidates 和一个目标整数 target ，找出 candidates 中可以使数字和为目标数 target 的所有不同组合，
     * 并以列表形式返回。你可以按 任意顺序返回这些组合。
     * candidates 中的 同一个 数字可以 无限制重复被选取 。如果至少一个数字的被选数量不同，则两种组合是不同的。 <br/>
     * <a href='https://leetcode.cn/problems/combination-sum/'>39. 组合总和</a>
     */
    @DFS
    public List<List<Integer>> combinationSum(int[] candidates, int target) {
        List<List<Integer>> res = new ArrayList<>();
        if (candidates.length == 0) {
            return res;
        }
        dfs(candidates, 0, target, new LinkedList<>(), res);
        return res;
    }

    /**
     * @param candidates 树
     * @param beginIndex 开始回溯的起点
     * @param target 目标和
     * @param path 当前路径
     * @param res 结果集指针
     */
    private void dfs(int[] candidates, int beginIndex, int target, Deque<Integer> path, List<List<Integer>> res) {
        if (target == 0) {
            res.add(new ArrayList<>(path));
            return;
        }
        for (int i = beginIndex; i < candidates.length; i++) {
            final int candidate = candidates[i];
            if (target < candidate) {
                continue;
            }
            path.addLast(candidate);
            dfs(candidates, i, target - candidate, path, res);
            path.removeLast();
        }
    }

    /**
     * 给定一个候选人编号的集合 candidates 和一个目标数 target ，找出 candidates 中所有可以使数字和为 target 的组合。
     * candidates 中的每个数字在每个组合中只能使用 一次 。<br/>
     * <a href='https://leetcode.cn/problems/combination-sum-ii/'>40. 组合总和 II</a>
     */
    @DFS
    public List<List<Integer>> combinationSum2(int[] candidates, int target) {
        List<List<Integer>> res = new ArrayList<>();
        if (candidates.length == 0) {
            return res;
        }
        Arrays.sort(candidates);
        dfs2(candidates, 0, target, new LinkedList<>(), res);
        return res;
    }

    private void dfs2(int[] candidates, int beginIndex, int target, Deque<Integer> path,
            List<List<Integer>> res) {

        if (target == 0) {
            res.add(new ArrayList<>(path));
            return;
        }
        for (int i = beginIndex; i < candidates.length && target >= candidates[i]; i++) {
            final int candidate = candidates[i];
            if (i > beginIndex && candidates[i - 1] == candidates[i]) {
                continue;
            }
            path.addLast(candidate);
            dfs2(candidates, i + 1, target - candidate, path, res);
            path.removeLast();
        }
    }

    /**
     * 给定一个二进制数组 nums , 找到含有相同数量的 0 和 1 的最长连续子数组，并返回该子数组的长度。 <br/>
     * <a href='https://leetcode.cn/problems/A1NYOS/'>剑指 Offer II 011. 0 和 1 个数相同的子数组</a>
     */
    public int findMaxLength(int[] nums) {
        for (int i = 0; i < nums.length; i++) {
            if (nums[i] == 0) {
                nums[i] = -1;
            }
        }
        int res = 0;
        //问题转换为前缀和之差为0的两个坐标间隔最大的值，sum(i)==sum(j)
        //key为前缀和，value为第一个出现该和的index
        Map<Integer, Integer> sumMap = new HashMap<>();
        //初始下标为-1，和为0，保证数组长度为1时解为0-(-1)=1
        sumMap.put(0, -1);
        int sum = 0;
        for (int i = 0; i < nums.length; i++) {
            sum += nums[i];
            if (sumMap.containsKey(sum)) {
                res = Math.max(res, i - sumMap.get(sum));
            } else {
                sumMap.put(sum, i);
            }
        }
        return res;
    }


    /**
     * 给定一个链表，删除链表的倒数第 n 个结点，并且返回链表的头结点。 <br/>
     * <a href='https://leetcode.cn/problems/SLwz0R/'>剑指 Offer II 021. 删除链表的倒数第 n 个结点</a>
     */
    public ListNode removeNthFromEnd(ListNode head, int n) {
        ListNode subHead = head;
        ListNode subTail = head;
        for (int i = 0; i < n; i++) {
            subTail = subTail.next;
        }
        if (subTail == null) {
            //删除头结点的情况
            final ListNode next = head.next;
            head.next = null;
            return next;
        }
        while (subTail.next != null) {
            subHead = subHead.next;
            subTail = subTail.next;
        }
        final ListNode next = subHead.next;
        final ListNode nextnext = next.next;
        next.next = null;
        subHead.next = nextnext;
        return head;
    }


    public static void main(String[] args) {
        final Solution2 solution = new Solution2();
//        solution.numSubarrayProductLessThanK(new int[]{10}, 2);
//        solution.addTwoNumbers(stringToListNode("[7,2,4,3]"), stringToListNode("[5,6,4]"));
//        solution.subarraySum(stringToArray("[1,1,1]"), 2);
//        solution.countSubstrings("aba");
//        solution.combinationSum2(stringToArray("[10,1,2,7,6,1,5]"), 8);
        solution.findMaxLength(stringToArray("[0,1,1,0,0,0,0,1,1,1,1]"));
    }

}
