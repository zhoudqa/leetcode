import annotations.algorithm.DynamicPrograming;
import annotations.level.Hard;
import annotations.level.Medium;

import java.util.Arrays;
import java.util.Comparator;

public class Solution2024 {

    //整数转换。编写一个函数，确定需要改变几个位才能将整数A转成整数B。
    public int convertInteger(int A, int B) {
        if (A == B) {
            return 0;
        }
//        0b11101
//        0b01111
        int xor = A ^ B;
        return Integer.bitCount(xor);
    }

    @Medium
    //教学过程中，教练示范一次，学员跟做三次。该过程被混乱剪辑后，记录于数组 actions，其中 actions[i] 表示做出该动作的人员编号。请返回教练的编号。
    public int trainingPlan(int[] nums) {
        if (nums.length == 0) return -1;//输入数组长度不符合要求，返回-1;
        int[] bitSum = new int[32];//java int类型有32位，其中首位为符号位
        int res = 0;
        for (int num : nums) {
            int bitMask = 1;//需要在这里初始化，不能和res一起初始化
            for (int i = 31; i >= 0; i--) {//bitSum[0]为符号位
                //这里同样可以通过num的无符号右移>>>来实现，否则带符号右移(>>)左侧会补符号位，对于负数会出错。
                //但是不推荐这样做，最好不要修改原数组nums的数据
                if ((num & bitMask) != 0) bitSum[i]++;//这里判断条件也可以写为(num&bitMask)==bitMask,而不是==1
                bitMask = bitMask << 1;//左移没有无符号、带符号的区别，都是在右侧补0
            }
        }
        for (int i = 0; i < 32; i++) {//这种做法使得本算法同样适用于负数的情况
            res = res << 1;
            res += bitSum[i] % 3;//这两步顺序不能变，否则最后一步会多左移一次
        }
        return res;

    }

    @DynamicPrograming
    @Hard
    //堆箱子。给你一堆n个箱子，箱子宽 wi、深 di、高 hi。箱子不能翻转，将箱子堆起来时，下面箱子的宽度、高度和深度必须大于上面的箱子。实现一种方法，搭出最高的一堆箱子。箱堆的高度为每个箱子高度的总和。
    public int pileBox(int[][] box) {
        if (box.length == 0) {
            return 0;
        }
        //先排序 保证一定程度上的有序
        Arrays.sort(box, Comparator.comparingInt(e -> e[0]));
        //dp[i]表示以第i个箱子结尾的最大高度
        int[] dp = new int[box.length];
        int ret = 0;
        for (int i = 0; i < box.length; i++) {
            //初始值只有第i个box本身
            int currentHi = box[i][2];
            dp[i] = currentHi;
            for (int j = 0; j < i; j++) {
                if (box[i][0] > box[j][0] && box[i][1] > box[j][1] && currentHi > box[j][2]) {
                    //满足条件后，当前箱子加入仅仅加入前j个会不会更高
                    dp[i] = Math.max(dp[i], dp[j] + currentHi);
                }
            }
            //使用或者不使用第i个选更大的结果
            ret = Math.max(ret, dp[i]);
        }
        return ret;
    }

    @Medium
    @DynamicPrograming
    //你选择掷出 num 个色子，请返回所有点数总和的概率。
    //你需要用一个浮点数数组返回答案，其中第 i 个元素代表这 num 个骰子所能掷出的点数集合中第 i 小的那个的概率。
    public double[] statisticsProbability(int num) {
        //dp[i][j]表示使用i个色子的时候，总和为j的组合数
        int[][] dp = new int[num + 1][6 * num + 1];
        double[] ans = new double[5 * num + 1];
        for (int i = 1; i <= 6; i++) {
            //一个色子每个总和只有一个可能
            dp[1][i] = 1;
        }
        double all = Math.pow(6, num);
        for (int i = 1; i <= num; i++) {
            for (int j = i; j <= 6 * num; j++) {
                //状态转移方程，dp[i][j]=dp[i-1][j-1]+...dp[i-1][j-6]，每个色子摇出来的可能性（需要j比要到的色子数大，不然不能选）
                for (int k = 1; k <= 6; k++) {
                    //总和大于点数才能算上这个色子的可能
                    dp[i][j] += j >= k ? dp[i - 1][j - k] : 0;
                    if (i == num)
                        ans[j - i] = dp[i][j] / all;
                }
            }
        }
        return ans;
    }
}
