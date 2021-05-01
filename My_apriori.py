'''
作者：胡亚洲（2021.01.19）
用Apriori和FP-Growth算法实现频繁项集与关联规则的挖掘

命令行调用方法：$python My_apriori.py -f CsvFilePath -s SupportRatio  -n FrequentNum
例子：$python My_apriori.py -f ../GroceryStore/Groceries.csv -s 0.01  -n 3
'''

import os
import sys
import csv
import time
import copy
import psutil
import pyfpgrowth
import argparse
import matplotlib.pyplot as plt
import numpy as np


# 数据读取：
def read_data(filepath):
    Data = []
    with open(filepath, "r", encoding='UTF-8') as f:
        reader = csv.reader(f)
        start_read = 0   # 不读第一行
        for row in reader:
            if start_read == 0:
                start_read = 1
                continue   # 跳过第一行数据的读取
            str = row[1].replace('{', '').replace('}', '')   # 获取购物记录，并去除两边的'{'和'}'
            str = str.replace('/', ' ')   # 将商品名中的'/'替换成' '
            str_list = str.split(',')   # 划分数据
            str_list = list(set(str_list))   # 去重
            str_list.sort()   # 排序
            Data.append(str_list)   # 将数据添加到Data列表中
    print('共读取{}条交易记录'.format(len(Data)))
    return Data


# 生成候选项集C1
def create_c1(Data):
    C1_set = set()   # 集合：元素不重复，无序，可变
    C1_list = []   # 将候选集C1存在一个二维列表里
    for items in Data:
        for item in items:
            C1_set.add(item)
    print('一共包含{}个商品'.format(len(C1_set)))
    for item in C1_set:
        c1 = []
        c1.append(item)
        C1_list.append(c1)
    C1_list.sort()   # 对列表进行排序
    return C1_list


# 由频繁项集生成新的候选项集 -- 例如L1创建C2，L2创建C3
def create_ck(lk, Advanced=0):
    Ck = []
    lk_len = len(lk)   # lk的长度
    for i in range(lk_len):
        for j in range(i + 1, lk_len):   # 两次遍历Lk-1，找出前n-1个元素相同的项
            l1 = lk[i]
            l2 = lk[j]
            l1.sort()   # 对两个列表排序
            l2.sort()
            if l1[:-1] == l2[:-1]:   # 只有最后一项不同时，生成下一候选项
                # 将两个表合在一起：
                c = list(set(l1).union(set(l2)))   # 将两个列表合在一起，生成新的候选集
                c.sort()   # 对列表排序
                if Advanced == 0:
                    Ck.append(c)
                else:
                    add_c = 1
                    for item in c:
                        c_subset = c.copy()
                        c_subset.remove(item)
                        if c_subset not in lk:
                            add_c = 0   # 不添加这一候选集
                            break
                    if add_c:
                        Ck.append(c)
    Ck.sort()   # 对列表进行排序--按首字母
    return Ck


# 通过候选项ck生成频繁项集lk，并将各频繁项的支持度保存到support_data字典中
def generate_lk(Data, ck, support_num, support_dict, Advanced=0):
    c_count = {}   # 用于标记各候选项在数据集出现的次数
    if Advanced >= 2:
        Data_count = {}   # 记录每个事务表出现的次数
    if Advanced >= 3:
        item_count = {}   # 记录事务表中每个购物记录中每一项被匹配的次数
    Lk = []
    k = len(ck[0])

    for items in Data:  # 遍历数据集
        items_set = frozenset(items)
        if Advanced >= 2:
            Data_count[items_set] = 0  # 对事务列表的统计个数进行初始化--避免统计个数为零而不被剔除的情况：
        if Advanced >= 3:
            item_count[items_set] = {}   # 每个记录对应一个字典，字典里是每个项的匹配次数，匹配次数为零的项不出现在字典里
        for c in ck:
            c_set = frozenset(c)   # 生成不可变集合--dict用hash值进行索引，对要存储的元素有可哈希要求
            if c_set.issubset(items):   # ck中的项是否都出现在购物记录中，items可以不是set类型
                if c_set not in c_count:
                    c_count[c_set] = 1   # 如果候选项不包含在字典中，添加进去
                else:
                    c_count[c_set] += 1  # python不支持字典的key为list和set，只支持不可变set
                if Advanced >= 2:
                    Data_count[items_set] += 1   # 事务项的被匹配次数加一
                if Advanced >= 3:
                    for item in c:
                        if item not in item_count[items_set]:
                            item_count[items_set][item] = 1   # item是str类型
                        else:
                            item_count[items_set][item] += 1
    # 将满足支持度的候选项添加到频繁项集中
    for c_set in c_count:
        if c_count[c_set] >= support_num:
            Lk.append(list(c_set))
            support_dict[c_set] = c_count[c_set]  # 将频繁项集中的项对应的支持度信息保留到字典中
    Lk.sort()   # 对列表进行排序--按首字母
    # 打印各频繁项集的数量：
    if Lk:
        print('频繁{}项集的数目为：{}'.format(k, len(Lk)))
    # 剪枝策略2：减少事务表的规模--在匹配K-项候选集时，如果一个记录被匹配到的次数少于(K+1)此，则将该事务表移除：
    if Advanced >= 2:
        for items_set in Data_count:
            if Data_count[items_set] < (k+1):
                Data.remove(sorted(list(items_set)))   # 要对列表排序，否则可能匹配不上
    # 剪枝策略3：减少事务表中元组的项--如果事务表中购物记录中的某一项被匹配到的次数少于K次，则将该项从购物记录中删除
    if Advanced >= 3:
        Data_new = []   # 保存满足支持度计数项的新的列表
        for items in Data:
            items_set = frozenset(items)
            item_dict = item_count[items_set]   # 对应购物记录各项的记数统计字典
            for item in items:
                if (item not in item_dict) or (item_dict[item] < k):
                    items.remove(item)   # 在购物记录中去掉不满足支持度计数的项
            if items:
                Data_new.append(items)
        Data = copy.deepcopy(Data_new)
        Data.sort()
    return Lk


# Apriori函数，用标记变量Advances确定剪枝方法的使用:
# Advanced=0:不适用剪枝；=1：使用剪枝方法1；=2：使用剪枝方法1和2；=3：使用剪枝方法1,2,3.
def my_apriori(data_raw, support_num, Advanced=0):
    Data = copy.deepcopy(data_raw)  # 防止原列表被改变，用copy()在Advanced=3时不行
    # 计算内存使用：
    pid = os.getpid()
    p = psutil.Process(pid)
    mem_start = p.memory_info().rss

    L_time = []   # 记录不同频繁项集数的总时间变化
    time_start = time.time()
    support_dict = {}   # 频繁项集及对应的支持度--注意字典是无序的
    C1 = create_c1(Data)
    L1 = generate_lk(Data, C1, support_num, support_dict, Advanced)    # 生成频繁1项集
    L_time.append(time.time() - time_start)
    Li = L1.copy()
    while True:
        Ci = create_ck(Li, Advanced)   # 生成候选项集
        Li = generate_lk(Data, Ci, support_num, support_dict, Advanced)   # 生成频繁项集
        L_time.append(time.time() - time_start)
        # 如果频繁项集为空，停止循环：
        if len(Li) == 0:
            break
    mem_use = p.memory_info().rss - mem_start
    return support_dict, L_time, mem_use


# 关联规则挖掘：
def my_rule(support_dict, Confidence_ratio):
    rule_list = []   # 存储关联规则，格式：项A，项B，置信度
    for lk in support_dict:   # lk是不可变集合
        if len(lk) > 1:   # 只有长度大于1的频繁项才可能产生关联规则
            lk_list = list(lk)   # 由不可变集合变成列表
            for item_B in lk_list:   # 寻找lk的子集，计算置信度：
                item_A = lk_list.copy()
                item_A.remove(item_B)   # item_B的子集，去一个项
                item_A = frozenset(item_A)   # 把列表变为不可变集合
                if item_A in support_dict:
                    support_value = support_dict[lk] / support_dict[item_A]    # 计算当前的置信度
                    if support_value >= Confidence_ratio:
                        rule = []   # A,B,对应的置信度
                        rule.append(sorted(list(item_A)))   # item_A是列表格式的
                        rule.append(item_B)   # item_B是str格式的
                        rule.append(support_value)
                        rule_list.append(rule)
    rule_list.sort(key=lambda x: x[-1], reverse=True)  # 对列表按最后一列（置信度）进行排序，并将排序结果翻转至置信度降序排列
    # 打印关联规则：
    print('\n*************** Apriori算法挖掘的关联规则(前10项) ***************')
    with open('../Output/Rules_by_Apriori.txt', 'w') as f:
        f.write('序号\t\t项\t\t置信度\n')
        for i, rule in enumerate(rule_list):
            f.write('({}) {} ==> <{}>,\t{:.4f}\n'.format(i+1, rule[0], rule[1], rule[2]))
            if i <= 9:
                print('({}) {} ==> <{}>,\t{:.4f}'.format(i+1, rule[0], rule[1], rule[2]))
    return rule_list


# 将Apriori算法生成的频繁三项集输出到文件，并打印10个到屏幕上：
def L_output(support_dict, Fre_num):
    if not os.path.exists('../Output'):
        os.makedirs('../Output')
    with open('../Output/Frequent_{}_by_Apriori.txt'.format(Fre_num), 'w') as f:
        f.write('序号\t\t项\t\t出现频次\n')
        print('\n*************** Apriori算法挖掘的频繁{}项集(前10项) ***************'.format(Fre_num))
        index = 1   # 用于统计输出3-项集的个数
        for L_set in support_dict:
            L_list = list(L_set)
            if len(L_list) == Fre_num:
                f.write('({}) {}\t{}\n'.format(index, L_list, support_dict[L_set]))
                if index <= 10:
                    print('({}) {}\t{}'.format(index, L_list, support_dict[L_set]))
                index += 1


if __name__ == '__main__':
    # 用命令行接收参数：
    parser = argparse.ArgumentParser()
    # 文件名：
    parser.add_argument('-f', help='CsvFilePath', type=str, dest='filepath', default='../GroceryStore/Groceries.csv')
    # 支持率：
    parser.add_argument('-s', help='Support Ratio', type=float, dest='Support_ratio', default=0.01)
    # 挖掘频繁项集的大小：
    parser.add_argument('-n', help='Frequnet num', type=int, dest='Fre_num', default=3)

    # 获取参数
    args = parser.parse_args()
    filepath = args.filepath
    Support_ratio = args.Support_ratio
    Fre_num = args.Fre_num
    Confidence_ratio = 0.5

    # 数据读取：
    Data_raw = read_data(filepath)   # 文件的读取结果是一个二维列表，每一行对应一个购物记录。
    support_num = Support_ratio * len(Data_raw)   # 没有对支持度进行取整

    # # 生成频繁项集：
    # # 返回的字典包含所有频繁项集及对应的支持度--字典的key()为不可变集合：
    print('(1)Dummy Apriori:')
    [support_dict, time_dummy, mem_dummy] = my_apriori(Data_raw, support_num, Advanced=0)
    print('\n(2)Advanced Apriori_1:')
    [support_dict, time_one, mem_one] = my_apriori(Data_raw, support_num, Advanced=1)
    print('\n(3)Advanced Apriori_12:')
    [support_dict, time_two, mem_two] = my_apriori(Data_raw, support_num, Advanced=2)
    print('\n(4)Advanced Apriori_123:')
    [support_dict, time_three, mem_three] = my_apriori(Data_raw, support_num, Advanced=3)
    # 输出及打印Apriori算法得到的指定频繁项集
    L_output(support_dict, Fre_num)

    # 挖掘关联规则：
    # 返回一个列表，列表的每一项是一个关联规则，格式为：列表item_A，字符串item_B，置信度
    rule_list = my_rule(support_dict, Confidence_ratio)

    # 调用pyfpgrowth库生成频繁项集和关联规则：
    pid = os.getpid()
    p = psutil.Process(pid)
    mem_FP_start = p.memory_info().rss
    patterns = pyfpgrowth.find_frequent_patterns(Data_raw, support_num)   # 挖掘频繁项集
    mem_FP = p.memory_info().rss - mem_FP_start   # FP方法的内存消耗情况
    rules = pyfpgrowth.generate_association_rules(patterns, Confidence_ratio)   # 返回的结果是字典类型
    print('\n*************** FP-Growth算法挖掘的关联规则 ***************')
    with open('../Output/Rules_by_FP-Growth.txt', 'w') as f:
        f.write('序号\t\t项\t\t置信度\n')
        for i, rule in enumerate(rules):
            f.write('({}) {} ==> {},\t{:.4f}\n'.format(i+1, rule, list(rules[rule][0]), rules[rule][1]))
            print('({}) {} ==> {},\t{:.4f}'.format(i+1, rule, list(rules[rule][0]), rules[rule][1]))

# 画图：
if not os.path.exists('../Fig'):
    os.makedirs('../Fig')

# （1）总耗时随频繁项集数的变化：
plt.figure()
x_axis = np.linspace(1, 4, 4)
plt.plot(x_axis, time_dummy, color='blue', label='Dummy Apriori')
plt.plot(x_axis, time_one, color='green', label='Advanced Apriori_1')
plt.plot(x_axis, time_two, color='red', label='Advanced Apriori_12')
plt.plot(x_axis, time_three, color='black', label='Advanced Apriori_123')
plt.legend()
plt.xlabel('Frequent items')
plt.ylabel('Total time/s')
plt.title('Time via items')
plt.savefig('../Fig/Time via items.png')

# （2）每一项用时的对比：
method_name = ['Dummy', 'Adv_1', 'Adv_12', 'Adv_123']
plt.figure()
for i in range(4):
    plt.subplot(2, 2, i+1)
    if i:
        plt.bar(method_name, [time_dummy[i]-time_dummy[i-1], time_one[i]-time_one[i-1], time_two[i]-time_two[i-1],
                              time_three[i]-time_three[i-1]], color=['blue', 'green', 'red', 'black'])
        # 每一项记录的是总时间，从第二项开始需要减去前一项获取该项所消耗的时间。
    else:
        plt.bar(method_name, [time_dummy[i], time_one[i], time_two[i], time_three[i]],
                color=['blue', 'green', 'red', 'black'])
    plt.xlabel('Freq_{}'.format(i+1))
    plt.ylabel('Time/s')
plt.tight_layout()   # 避免子图间发生重叠
plt.savefig('../Fig/Time of each item.png')

# （3）内存使用情况对比：
method_name.append('FP_Growth')   # 将FP-Growth方法加进去
plt.figure()
plt.bar(method_name, [mem_dummy, mem_one, mem_two, mem_three, mem_FP],
        color=['blue', 'green', 'red', 'black', 'cyan'])
plt.xlabel('Method')
plt.ylabel('Memory/B')
plt.savefig('../Fig/Memory usage.png')
plt.show()
