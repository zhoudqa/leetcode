import annotations.algorithm.*;
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
    @Backtrack
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
    @Backtrack
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

    /**
     * 运用所掌握的数据结构，设计和实现一个  LRU (Least Recently Used，最近最少使用) 缓存机制 。 <br/>
     * <a href='https://leetcode.cn/problems/OrIXps/'>剑指 Offer II 031. 最近最少使用缓存</a>
     */
    class LRUCache {


        class LRUNode {

            int key;
            int value;
            LRUNode pre;
            LRUNode next;

            public LRUNode(int key, int value) {
                this.key = key;
                this.value = value;
            }

            public LRUNode() {
            }
        }

        private final int capacity;
        private int size;
        private final LRUNode dummyHead = new LRUNode();
        private final LRUNode dummyTail = new LRUNode();
        private final Map<Integer, LRUNode> map = new HashMap<>();

        public LRUCache(int capacity) {
            this.capacity = capacity;
            this.size = 0;
            dummyHead.next = dummyTail;
            dummyTail.pre = dummyHead;
        }

        public int get(int key) {
            if (!map.containsKey(key)) {
                return -1;
            }
            final LRUNode node = map.get(key);
            //删除节点
            removeNode(node);
            //添加到头结点
            addToHead(node);
            return node.value;
        }

        private void addToHead(LRUNode node) {
            final LRUNode preHead = dummyHead.next;
            dummyHead.next = node;
            node.pre = dummyHead;
            node.next = preHead;
            preHead.pre = node;
        }

        private LRUNode removeNode(LRUNode node) {
            node.pre.next = node.next;
            node.next.pre = node.pre;
            node.next = null;
            node.pre = null;
            return node;
        }

        private LRUNode removeTail() {
            return removeNode(dummyTail.pre);
        }

        public void put(int key, int value) {
            if (!map.containsKey(key)) {
                //不存在
                final LRUNode node = new LRUNode(key, value);
                map.put(key, node);
                addToHead(node);
                size++;
                if (size > capacity) {
                    final LRUNode tail = removeTail();
                    map.remove(tail.key);
                    size--;
                }
            } else {
                final LRUNode node = map.get(key);
                node.value = value;
                get(key);
            }
        }
    }

    /**
     * 给定一个字符串数组 strs ，将 变位词 组合在一起。 可以按任意顺序返回结果列表。
     * 注意：若两个字符串中每个字符出现的次数都相同，则称它们互为变位词。 <br/>
     * <a href='https://leetcode.cn/problems/sfvd7V/'>剑指 Offer II 033. 变位词组</a>
     */
    public List<List<String>> groupAnagrams(String[] strs) {
        Map<String, List<String>> map = new HashMap<>();
        for (final String str : strs) {
            int[] ints = new int[26];
            for (int i = 0; i < str.length(); i++) {
                ints[str.charAt(i) - 'a']++;
            }
            StringBuilder sb = new StringBuilder();
            for (int i = 0; i < 26; i++) {
                if (ints[i] != 0) {
                    //字母加个数
                    sb.append((char) (i + 'a'));
                    sb.append(ints[i]);
                }
            }
            final String key = sb.toString();
            map.putIfAbsent(key, new ArrayList<>());
            map.get(key).add(str);
        }
        return new ArrayList<>(map.values());
    }

    /**
     * 完全二叉树插入<br/>
     * <a href='https://leetcode.cn/problems/NaqhDT/'>剑指 Offer II 043. 往完全二叉树添加节点</a>
     */
    @BFS
    static class CBTInserter {

        private final TreeNode root;
        private int size;
        private int level;

        public CBTInserter(TreeNode root) {
            this.root = root;
            bfs(root);
        }

        private void bfs(TreeNode root) {
            //统计个数和层数
            Deque<TreeNode> deque = new LinkedList<>();
            deque.add(root);
            while (!deque.isEmpty()) {
                level++;
                final int n = deque.size();
                size += n;
                for (int i = 0; i < n; i++) {
                    final TreeNode poll = deque.poll();
                    if (poll.left != null) {
                        deque.addLast(poll.left);
                    }
                    if (poll.right != null) {
                        deque.addLast(poll.right);
                    }
                }
            }
        }

        public int insert(int v) {
            final TreeNode newNode = new TreeNode(v);
            if (size == ((int) (Math.pow(2, level) - 1))) {
                //长高
                TreeNode parent = root;
                while (parent.left != null) {
                    parent = parent.left;
                }
                parent.left = newNode;
                size++;
                level++;
                return parent.val;
            } else {
                size++;
                return insert(root, 1, newNode);
            }
        }

        private Integer insert(TreeNode node, int nodeLevel, TreeNode newNode) {
            if (nodeLevel == level) {
                //边叶子节点
                return null;
            }
            if (node.left == null) {
                node.left = newNode;
                return node.val;
            } else if (node.right == null) {
                node.right = newNode;
                return node.val;
            } else {
                final Integer leftInsert = insert(node.left, nodeLevel + 1, newNode);
                if (leftInsert == null) {
                    return insert(node.right, nodeLevel + 1, newNode);
                } else {
                    return leftInsert;
                }
            }
        }


        public TreeNode get_root() {
            return root;
        }
    }


    /**
     * 给定两个字符串 text1 和 text2，返回这两个字符串的最长 公共子序列 的长度。如果不存在 公共子序列 ，返回 0 。<br/>
     * 一个字符串的 子序列 是指这样一个新的字符串：它是由原字符串在不改变字符的相对顺序的情况下删除某些字符（也可以不删除任何字符）后组成的新字符串。<br/>
     * 例如，"ace" 是 "abcde" 的子序列，但 "aec" 不是 "abcde" 的子序列。<br/>
     * 两个字符串的 公共子序列 是这两个字符串所共同拥有的子序列。 <br/>
     * <a href='https://leetcode.cn/problems/qJnOS7/description/'>剑指 Offer II 095. 最长公共子序列</a>
     */
    @DynamicPrograming
    public int longestCommonSubsequence(String text1, String text2) {
        //dp(i,j)标识text1前i位和text2前j位的最长公共子序列
        //dp(0,0)=0；当text[j-1]!=text[i-1]时，dp(i,j)为dp[i-1][j]和的dp[i][j-1]中较大的值，否则为dp[i-1][j-1]+1
        int[][] dp = new int[text1.length() + 1][text2.length() + 1];
        for (int i = 1; i <= text1.length(); i++) {
            for (int j = 1; j <= text2.length(); j++) {
                dp[i][j] = text1.charAt(i - 1) == text2.charAt(j - 1) ? dp[i - 1][j - 1] + 1
                        : Math.max(dp[i][j - 1], dp[i - 1][j]);
            }
        }
        return dp[text1.length()][text2.length()];
    }

    /**
     * 正整数 n 代表生成括号的对数，请设计一个函数，用于能够生成所有可能的并且 有效的 括号组合。<br/>
     * <a href='https://leetcode.cn/problems/IDBivT/'>剑指 Offer II 085. 生成匹配的括号</a>
     */
    @Backtrack
    public List<String> generateParenthesis(int n) {
        List<String> res = new ArrayList<>();
        backtrack(res, new StringBuilder(), 0, 0, n);
        return res;
    }

    @DynamicPrograming
    public List<String> generateParenthesisDP(int n) {
        if (n == 0) {
            return Collections.emptyList();
        }
        Map<Integer, List<String>> dp = new HashMap<>();
        final List<String> noneQuote = Collections.singletonList("");
        dp.put(1, Collections.singletonList("()"));
        for (int i = 2; i <= n; i++) {
            dp.put(i, new ArrayList<>());
            for (int j = 0; j < i; j++) {
                final List<String> inners = dp.getOrDefault(j, noneQuote);
                final List<String> outers = dp.getOrDefault(i - 1 - j, noneQuote);
                for (final String inner : inners) {
                    for (final String outer : outers) {
                        dp.get(i).add("(" + inner + ")" + outer);
                    }
                }
            }
        }
        return dp.get(n);
    }

    private void backtrack(List<String> res, StringBuilder sb, int left, int right, int n) {
        if (sb.length() == n << 1) {
            res.add(sb.toString());
            return;
        }
        if (left < n) {
            sb.append('(');
            backtrack(res, sb, left + 1, right, n);
            sb.deleteCharAt(sb.length() - 1);
        }
        if (right < left) {
            sb.append(')');
            backtrack(res, sb, left, right + 1, n);
            sb.deleteCharAt(sb.length() - 1);
        }
    }

    /**
     * 给定一个由 0 和 1 组成的非空二维数组 grid ，用来表示海洋岛屿地图。<br/>
     * 一个 岛屿 是由一些相邻的 1 (代表土地) 构成的组合，这里的「相邻」要求两个 1 必须在水平或者竖直方向上相邻。<br/>
     * 你可以假设 grid 的四个边缘都被 0（代表水）包围着。<br/>
     * 找到给定的二维数组中最大的岛屿面积。如果没有岛屿，则返回面积为 0 。<br/>
     * <a href='https://leetcode.cn/problems/ZL6zAn/'>剑指 Offer II 105. 岛屿的最大面积</a>
     */
    public int maxAreaOfIsland(int[][] grid) {
        int res = 0;
        for (int i = 0; i < grid.length; i++) {
            for (int j = 0; j < grid[0].length; j++) {
                if (grid[i][j] == 1) {
                    res = Math.max(dfs(i, j, grid), res);
                }
            }
        }
        return res;
    }

    private int dfs(int i, int j, int[][] grid) {
        if (i < 0 || j < 0 || i >= grid.length || j >= grid[0].length || grid[i][j] != 1) {
            return 0;
        }
        int area = 1;
        grid[i][j] = 0;//visited标记
        area += dfs(i + 1, j, grid);
        area += dfs(i - 1, j, grid);
        area += dfs(i, j + 1, grid);
        area += dfs(i, j - 1, grid);
        return area;
    }

    /**
     * 给定一个由 不同 正整数组成的数组 nums ，和一个目标整数 target 。请从 nums 中找出并返回总和为 target 的元素组合的个数。
     * 数组中的数字可以在一次排列中出现任意次，但是顺序不同的序列被视作不同的组合。<br/>
     * 题目数据保证答案符合 32 位整数范围。<br/>
     * <a href='https://leetcode.cn/problems/D0F0SV/'>剑指 Offer II 104. 排列的数目</a>
     */
    @DynamicPrograming
    public int combinationSum4(int[] nums, int target) {
        //dp(i)为和为i的组合总数 dp(i)=∑dp(i-num)
        int[] dp = new int[target + 1];
        dp[0] = 1;
        for (int i = 1; i < target + 1; i++) {
            for (final int num : nums) {
                if (num <= i) {
                    //加上选取num的情况
                    dp[i] += dp[i - num];
                }
            }
        }
        return dp[target];
    }

    /**
     * 解码方法总数<br/>
     * <a href='https://leetcode.cn/problems/decode-ways/'>91. 解码方法</a>
     */
    @DynamicPrograming
    public int numDecodings(String s) {
        final int len = s.length();
        //dp(i)表示s的前i个字符的解码方法数，dp(0)=1，即空字符串可以有1种解码方法，解码出一个空字符串。
        int[] dp = new int[len + 1];
        dp[0] = 1;
        for (int i = 1; i <= len; i++) {
            if (s.charAt(i - 1) != '0') {
                dp[i] += dp[i - 1];
            }
            if (i > 1 && s.charAt(i - 2) != '0' && Integer.parseInt(s.substring(i - 2, i)) <= 26) {
                dp[i] += dp[i - 2];
            }
        }
        return dp[len];
    }

    /**
     * 给定一个数组 prices ，它的第 i 个元素 prices[i] 表示一支给定股票第 i 天的价格。<br/>
     * 你只能选择 某一天 买入这只股票，并选择在 未来的某一个不同的日子 卖出该股票。设计一个算法来计算你所能获取的最大利润。<br/>
     * 返回你可以从这笔交易中获取的最大利润。如果你不能获取任何利润，返回 0 。<br/>
     * <a href='https://leetcode.cn/problems/best-time-to-buy-and-sell-stock/description/'>121. 买卖股票的最佳时机</a>
     */
    @DynamicPrograming
    public int maxProfit(int[] prices) {
        //dp[day][max_hand][has_stock]表示在第day天，至今交易hand手，手上(has_stock)有股票的情况获得的最大利润
        //dp[day][max_hand][0]=max(dp[day-1][hand][0], dp[day-1][hand][1] + prices[day]),手上有没有股票可能是前一天没有今天没买卖或者前一天有今天卖出了，取较大值
        //dp[day][max_hand][1]=max(dp[day-1][hand][1], dp[day-1][hand-1][0] - prices[day])，手上有股票可能是前一天有今天没有买卖，或者前一天没有今天买入了
        //dp[0][max_hand][0]=0,dp[0][max_hand][1]=-price[0]
        int dpi0 = 0, dpi1 = Integer.MIN_VALUE;
        for (int i = 0; i < prices.length; i++) {
            dpi0 = Math.max(dpi0, dpi1 + prices[i]);
            //max_hand==0时，dp[day-1][0][0]一定是0，代表昨天交易0手不能交易
            dpi1 = Math.max(dpi1, -prices[i]);
        }
        //最后一天手里不持有股票的利润最高
        return dpi0;
    }

    /**
     * 给定两个以升序排列的整数数组 nums1 和 nums2 , 以及一个整数 k 。<br/>
     * 定义一对值 (u,v)，其中第一个元素来自 nums1，第二个元素来自 nums2 。<br/>
     * 请找到和最小的 k 个数对 (u1,v1),  (u2,v2)  ...  (uk,vk) 。<br/>
     * <a href='https://leetcode.cn/problems/qn8gGX/'>剑指 Offer II 061. 和最小的 k 个数对</a>
     */
    public List<List<Integer>> kSmallestPairs(int[] nums1, int[] nums2, int k) {
        PriorityQueue<List<Integer>> pq = new PriorityQueue<>(Comparator.comparingInt(o -> o.get(0) + o.get(1)));
        for (final int num1 : nums1) {
            for (final int num2 : nums2) {
                pq.add(Arrays.asList(num1, num2));
            }
        }
        List<List<Integer>> res = new ArrayList<>();
        while (!pq.isEmpty() && --k != 0) {
            res.add(pq.poll());
        }
        return res;

    }

    /**
     * 剑指 Offer II 073. 狒狒吃香蕉<br/>
     * <a href='https://leetcode.cn/problems/nZZqjQ/description/'>剑指 Offer II 073. 狒狒吃香蕉</a>
     */
    public int minEatingSpeed(int[] piles, int h) {
        int low = 1, high = 1;
        for (final int pile : piles) {
            //一个小时最多吃掉的香蕉
            high = Math.max(high, pile);
        }
        int res = low;
        while (low < high) {
            int mid = (low + high) / 2;
            final int time = eatTime(piles, mid);
            if (time <= h) {
                //吃快了，结果只能在这里面，先记录再继续二分
                res = mid;
                high = mid;
            } else {
                low = mid + 1;
            }
        }
        return res;
    }

    private int eatTime(int[] piles, int speed) {
        int time = 0;
        for (final int pile : piles) {
            time += (pile + speed - 1) / speed;
        }
        return time;
    }


    public static void main(String[] args) {
        final Solution2 solution = new Solution2();
//        solution.numSubarrayProductLessThanK(new int[]{10}, 2);
//        solution.addTwoNumbers(stringToListNode("[7,2,4,3]"), stringToListNode("[5,6,4]"));
//        solution.subarraySum(stringToArray("[1,1,1]"), 2);
//        solution.countSubstrings("aba");
//        solution.combinationSum2(stringToArray("[10,1,2,7,6,1,5]"), 8);
//        solution.findMaxLength(stringToArray("[0,1,1,0,0,0,0,1,1,1,1]"));
//        solution.groupAnagrams(new String[]{"eat", "tea", "tan", "ate", "nat", "bat"});
//        final CBTInserter insert = new CBTInserter(bfsBuild("[1,2,3,4,5,6]"));
//        insert.insert(7);
//        insert.insert(8);
//        solution.generateParenthesisDP(3);
//        solution.maxAreaOfIsland(stringToMatrix("[[1,1,0,0,0],[1,1,0,0,0],[0,0,0,1,1],[0,0,0,1,1]]"));
//        solution.combinationSum4(stringToArray("[1,2,3]"), 4);
        solution.minEatingSpeed(stringToArray("[3,6,7,11]"), 8);

    }

}
