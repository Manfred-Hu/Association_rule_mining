# Association_rule_mining
1.Content: Application of apriori algorithms with different pruning tricks and mining association rules in the given dataset(../GroceryStore/Groceries.csv).
-- Dummy Apriori;
-- Trick one: Reduce the size of candidate itemsets;
-- Trick two: Reduce the size of the table;
-- Trick three: Reduce the entries of tuples in the table.
-- Perfoemance analysis among differrnt apriori algorithms and FP-Growth algorithm.

2.Ways to run:
-- Run My_apriori.py directly(set the path of 'Groceries.csv' correctly).
-- Command line: $python My_apriori.py -f CsvFilePath -s SupportRatio -n FrequentNum
example: $python My_apriori.py -f ../GroceryStore/Groceries.csv -s 0.01 -n 3

3.Result:
3.1 Frequency three itemsets(FrequentNum=3, SupportRatio=0.01):
![image](https://user-images.githubusercontent.com/68360191/116813061-81675e00-ab84-11eb-96ff-25e167f0c767.png)
3.2 Association rules(SupportRatio=0.01, ConfidenceRatio=0.5):
![image](https://user-images.githubusercontent.com/68360191/116813211-4ade1300-ab85-11eb-9b4f-f7bf379fcb2d.png)
Differnt apriori algorithms has the same result. However, the results of FP-Growth is a bit different, and its association rules is less.
3.3 Comparison of running time:
![image](https://user-images.githubusercontent.com/68360191/116813237-73fea380-ab85-11eb-9412-78892c29b57a.png)
![image](https://user-images.githubusercontent.com/68360191/116813240-782ac100-ab85-11eb-8a80-583fdda04ff2.png)
3.4 Memory usage:
![image](https://user-images.githubusercontent.com/68360191/116813260-8d9feb00-ab85-11eb-885d-4dae5c7d12c9.png)
