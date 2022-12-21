import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.PriorityQueue;
import java.util.Set;
import java.util.stream.Collectors;

public class Solution1 {


    public int numDifferentIntegers(String word) {
        Set<String> integers = new HashSet<>();
        StringBuilder sb = new StringBuilder();
        for (final char c : word.toCharArray()) {
            if (c >= 48 && c <= 57) {
                sb.append(c);
            } else if (sb.length() != 0) {
                integers.add(parseNumber(sb.toString()));
                sb = new StringBuilder();
            }
        }
        if (sb.length() != 0) {
            integers.add(parseNumber(sb.toString()));
        }
        return integers.size();
    }

    private String parseNumber(String str) {
        int index = 0;
        while (index < str.length() && str.charAt(index) == 48) {
            index++;
        }
        return index == str.length() ? "0" : str.substring(index);
    }

    //给你一个n x n矩阵matrix ，其中每行和每列元素均按升序排序，找到矩阵中第 k 小的元素。
    //请注意，它是 排序后 的第 k 小元素，而不是第 k 个 不同 的元素。
    //你必须找到一个内存复杂度优于O(n2) 的解决方案。
    //n == matrix.length
    //n == matrix[i].length
    //1 <= n <= 300
    //-109 <= matrix[i][j] <= 109
    //题目数据 保证 matrix 中的所有行和列都按 非递减顺序 排列
    //1 <= k <= n^2
    public int kthSmallest(int[][] matrix, int k) {
        int n = matrix.length;
        int left = matrix[0][0];
        int right = matrix[n - 1][n - 1];
        //找到left到right之间的数，这个数为第N小，如果N<=k,则向左继续二分，否则向右继续二分
        while (left < right) {
            //中间数，不一定在数组中
            int mid = left + (right - left) / 2;
            if (inLeft(matrix, k, n, mid)) {
                //mid指针往左
                right = mid;
            } else {
                //mid指针往右
                left = mid + 1;
            }
        }
        return left;

    }

    private boolean inLeft(int[][] matrix, int k, int n, int mid) {
        //从左下角开始遍历
        int i = n - 1;
        int j = 0;
        int smaller = 0;
        while (i >= 0 && j <= n - 1) {
            if (matrix[i][j] <= mid) {
                //比mid小的在他上面，个数要+1，然后往右移动
                smaller += i + 1;
                j++;
            } else {
                //减小matrix[i][j]看看
                i--;
            }
        }
        return smaller >= k;
    }

    static class ListNode {

        int val;
        ListNode next;

        ListNode() {
        }

        ListNode(int val) {
            this.val = val;
        }

        ListNode(int val, ListNode next) {
            this.val = val;
            this.next = next;
        }
    }

    public ListNode mergeKLists(ListNode[] lists) {
        if (lists == null || lists.length == 0) {
            return null;
        } else if (lists.length == 1) {
            return lists[0];
        } else {
            return mergeLists(lists, 0, lists.length - 1);

        }

    }

    //二分合并
    private ListNode mergeLists(ListNode[] lists, int left, int right) {
        if (left == right) {
            return lists[left];
        }
        int mid = left + (right - left) / 2;
        ListNode leftNode = mergeLists(lists, left, mid);
        ListNode rightNode = mergeLists(lists, mid + 1, right);

        return mergeListNodes(leftNode, rightNode);
    }

    private ListNode mergeListNodes(ListNode list, ListNode list1) {
        if (list == null || list1 == null) {
            return list != null ? list : list1;
        }
        ListNode head = new ListNode(0);
        ListNode tail = head;
        ListNode left = list;
        ListNode right = list1;
        while (left != null && right != null) {
            if (left.val < right.val) {
                //复用以节省空间
                tail.next = left;
                left = left.next;
            } else {
                tail.next = right;
                right = right.next;
            }
            tail = tail.next;
        }
        tail.next = left == null ? right : left;
        return head.next;
    }

    //优先队列(堆)方式
    public ListNode mergeKListsWithHeap(ListNode[] lists) {
        if (lists == null || lists.length == 0) {
            return null;
        }
        PriorityQueue<ListNode> pq = new PriorityQueue<>(Comparator.comparingInt(n -> n.val));
        pq.addAll(Arrays.stream(lists).filter(Objects::nonNull).collect(Collectors.toList()));
        ListNode head = new ListNode(0);
        ListNode tail = head;
        while (!pq.isEmpty()) {
            tail.next = pq.poll();
            tail = tail.next;
            //剩下的一截入队列
            if (tail.next != null) {
                pq.add(tail.next);
            }
        }
        return head.next;
    }

    //给你一个整数数组 nums 和一个整数 k ，请你返回其中出现频率前 k 高的元素。你可以按 任意顺序 返回答案。
    public int[] topKFrequent(int[] nums, int k) {
        Map<Integer, Integer> frequencyMap = new HashMap<>();
        for (int num : nums) {
            frequencyMap.put(num, frequencyMap.getOrDefault(num, 0) + 1);
        }
        //维护大小为k的最小堆
        PriorityQueue<int[]> pq = new PriorityQueue<>(Comparator.comparingInt(o -> o[1]));
        frequencyMap.forEach((val, frequency) -> {
            //频率比最小的还低，就不需要入堆了，保证堆的大小为k
            if (pq.size() == k) {
                if (pq.peek()[1] < frequency) {
                    pq.poll();
                    pq.add(new int[]{val, frequency});
                }
            } else {
                pq.add(new int[]{val, frequency});
            }
        });
        int[] res = new int[k];
        for (int i = 0; i < k; i++) {
            res[i] = pq.poll()[0];
        }
        return res;
    }

    //给你一个整数数组 nums，有一个大小为k的滑动窗口从数组的最左侧移动到数组的最右侧。你只可以看到在滑动窗口内的 k个数字。滑动窗口每次只向右移动一位。
    //
    //返回 滑动窗口中的最大值 。
    public int[] maxSlidingWindow(int[] nums, int k) {
        int resLen = nums.length - (k - 1);
        int[] res = new int[resLen];
        PriorityQueue<int[]> pq = new PriorityQueue<>((o1, o2) -> o2[1] - o1[1]);
        for (int i = 0; i < k; i++) {
            pq.add(new int[]{i, nums[i]});
        }
        int idx = 0;
        res[idx++] = pq.peek()[1];
        for (int i = k; i < nums.length; i++) {
            pq.add(new int[]{i, nums[i]});
            int pre = i - k;
            //低效移除O(N) pq.removeIf(o -> o[0] == pre);
            while (pq.peek()[0] <= pre) {
                //高效移除去掉错误答案 i-k坐标的元素
                pq.poll();
            }
            res[idx++] = pq.peek()[1];
        }
        return res;

    }

    static class TreeNode {

        int val;
        TreeNode left;
        TreeNode right;

        TreeNode(int x) {
            val = x;
        }
    }

    static class Codec {

        // Encodes a tree to a single string.
        public String serialize(TreeNode root) {
            if (root == null) {
                return "X";
            }
            String left = "(" + serialize(root.left) + ")";
            String right = "(" + serialize(root.right) + ")";
            return left + root.val + right;
        }

        public TreeNode deserialize(String data) {
            int[] ptr = {0};
            return parse(data, ptr);
        }

        public TreeNode parse(String data, int[] ptr) {
            if (data.charAt(ptr[0]) == 'X') {
                ++ptr[0];
                return null;
            }
            TreeNode cur = new TreeNode(0);
            cur.left = parseSubtree(data, ptr);
            cur.val = parseInt(data, ptr);
            cur.right = parseSubtree(data, ptr);
            return cur;
        }

        public TreeNode parseSubtree(String data, int[] ptr) {
            ++ptr[0]; // 跳过左括号
            TreeNode subtree = parse(data, ptr);
            ++ptr[0]; // 跳过右括号
            return subtree;
        }

        public int parseInt(String data, int[] ptr) {
            int x = 0, sgn = 1;
            if (!Character.isDigit(data.charAt(ptr[0]))) {
                sgn = -1;
                ++ptr[0];
            }
            while (Character.isDigit(data.charAt(ptr[0]))) {
                x = x * 10 + data.charAt(ptr[0]++) - '0';
            }
            return x * sgn;
        }

    }

    //给你一个字符串 s 和一个整数 k ，请你找出 s 中的最长子串， 要求该子串中的每一字符出现次数都不少于 k 。返回这一子串的长度。
    public int longestSubstring(String s, int k) {
        return longestSubstring(s, 0, s.length() - 1, k);
    }


    //s的[left,right]子串中最长的每个字符出现次数都不小于k的子串
    private int longestSubstring(String s, int left, int right, int k) {
        int[] count = new int[26];
        for (int i = left; i <= right; i++) {
            count[s.charAt(i) - 'a']++;
        }
        int split = -1;
        for (int i = 0; i < 26; i++) {
            if (count[i] > 0 && count[i] < k) {
                //找到第一个不满足k个条件的char
                split = i;
            }
        }
        //找不到不满足条件的为退出条件
        if (split == -1) {
            return right - left + 1;
        }
        int i = left;
        int ret = Integer.MIN_VALUE;
        //从left到right比较每一个使用split分隔的子串
        while (i <= right) {
            while (i <= right && s.charAt(i) - 'a' == split) {
                i++;
            }
            //第一个不为split的index
            int newLeft = i;
            while (i <= right && s.charAt(i) - 'a' != split) {
                i++;
            }
            //第一个split的index
            int newRight = i - 1;
            ret = Math.max(ret, longestSubstring(s, newLeft, newRight, k));
        }
        return ret;
    }

    //给定一个包含 [0, n] 中 n 个数的数组 nums ，找出 [0, n] 这个范围内没有出现在数组中的那个数。
    public int missingNumber(int[] nums) {
        int sum = 0;
        int n = nums.length;
        for (int i = 0; i < n; i++) {
            sum += nums[i];
        }
        return n * (n + 1) / 2 - sum;
    }

    //给定一个未排序的整数数组 nums ，找出数字连续的最长序列（不要求序列元素在原数组中连续）的长度。
    //请你设计并实现时间复杂度为O(n) 的算法解决此问题。
    public int longestConsecutive(int[] nums) {
        Set<Integer> set = new HashSet<>();
        for (int num : nums) {
            set.add(num);
        }
        int ret = 0;
        for (int num : set) {
            //避免无用的循环
            if (!set.contains(num - 1)) {
                int j = 0;
                while (set.contains(num + (++j))) {
                }
                ret = Math.max(ret, j);
            }
        }
        return ret;

    }


    int maxPathSum = Integer.MIN_VALUE;

    //路径 被定义为一条从树中任意节点出发，沿父节点-子节点连接，达到任意节点的序列。同一个节点在一条路径序列中 至多出现一次 。该路径 至少包含一个 节点，且不一定经过根节点。
    //
    //路径和 是路径中各节点值的总和。
    //
    //给你一个二叉树的根节点 root ，返回其 最大路径和 。
    //递归
    public int maxPathSum(TreeNode root) {
        maxSingleSide(root);
        return maxPathSum;
    }

    //该函数返回包含root节点的最大路径和
    private int maxSingleSide(TreeNode root) {
        if (root == null) {
            return 0;
        }
        final int leftMax = Math.max(0, maxSingleSide(root.left));
        final int maxRight = Math.max(0, maxSingleSide(root.right));
        //不断更新路径和
        maxPathSum = Math.max(maxPathSum, root.val + leftMax + maxRight);
        return root.val + Math.max(leftMax, maxRight);

    }

    //给你一个整数 n ，返回 和为 n 的完全平方数的最少数量 。
    //
    //完全平方数 是一个整数，其值等于另一个整数的平方；换句话说，其值等于一个整数自乘的积。例如，1、4、9 和 16 都是完全平方数，而 3 和 11 不是。
    public int numSquares(int n) {
        //f[i] 表示最少需要多少个数的平方来表示整数i
        //f[0]=0
        //f[i]=1+ min(f[i-j^2),j^2<=j
        int[] f = new int[n + 1];
        for (int i = 1; i <= n; i++) {
            int minn = Integer.MAX_VALUE;
            for (int j = 1; j * j <= i; j++) {
                minn = Math.min(minn, f[i - j * j]);
            }
            f[i] = minn + 1;
        }
        return f[n];
    }

    public static void main(String[] args) {
        final Solution1 solution = new Solution1();
//        solution.numDifferentIntegers("0a0");
//        solution.kthSmallest(stringToMatrix("[[1,3,5],[6,7,12],[11,14,14]]"), 3);
//        solution.topKFrequent(stringToArray("[1,1,1,2,2,3]"), 2);
//        solution.maxSlidingWindow(stringToArray("[1,-1]"), 1);
//        final Codec codec = new Codec();
//        final TreeNode node = codec.deserialize("(1,2,3,X,X,4,5)");
//        Codec ser = new Codec();
//        Codec deser = new Codec();
//        TreeNode ans = deser.deserialize(ser.serialize(node));
        solution.longestConsecutive(stringToArray("[100,4,200,1,3,2]"));
        solution.numSquares(12);
        solution.numSquares(13);
    }

    public static int[] stringToArray(String s) {
        String[] split = s.substring(1, s.length() - 1).split(",");
        int[] res = new int[split.length];
        for (int i = 0; i < split.length; i++) {
            res[i] = Integer.parseInt(split[i]);
        }
        return res;
    }

    public static int[][] stringToMatrix(String s) {
        final String[] elements = s.substring(2, s.length() - 2).split("],\\[");
        final int h = elements.length;
        final int w = elements[0].split(",").length;
        int[][] res = new int[h][w];
        for (int i = 0; i < elements.length; i++) {
            final String[] split = elements[i].split(",");
            for (int j = 0; j < split.length; j++) {
                res[i][j] = Integer.parseInt(split[j]);
            }
        }
        return res;
    }

}
