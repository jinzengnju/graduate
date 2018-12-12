
import numpy as np

class Judger:
    # Initialize Judger, with the path of accusation list and law articles list
    def __init__(self):
        self.law_dic = {}
        f = open("law.txt", "r")
        self.task2_cnt = 0
        for line in f:
            self.law_dic[int(line[:-1])] = self.task2_cnt
            self.task2_cnt+=1

    # Format the result generated by the Predictor class
    @staticmethod
    def format_result(result):
        rex = {"articles": []}
        res_art = []
        for x in result["articles"]:
            if not (x is None):
                res_art.append(int(x))
        rex["articles"] = res_art
        return rex

    # Gen new results according to the truth and users output
    #这个是可以处理多标签的问题
    #下面这个函数是针对一个样本的预测结果
    def gen_new_result(self, result, truth, label):
        s1 = set(label)
        s2 = set(truth)

        for a in range(0, self.task2_cnt):
            in1 = a in s1
            in2 = a in s2
            if in1:
                if in2:
                    result[0][a]["TP"] += 1
                else:
                    result[0][a]["FP"] += 1
            else:
                if in2:
                    result[0][a]["FN"] += 1
                else:
                    result[0][a]["TN"] += 1
        return result

    # Calculate precision, recall and f1 value
    # According to https://github.com/dice-group/gerbil/wiki/Precision,-Recall-and-F1-measure
    @staticmethod
    def get_value(res):
        if res["TP"] == 0:
            if res["FP"] == 0 and res["FN"] == 0:
                precision = 1.0
                recall = 1.0
                f1 = 1.0
            else:
                precision = 0.0
                recall = 0.0
                f1 = 0.0
        else:
            precision = 1.0 * res["TP"] / (res["TP"] + res["FP"])
            recall = 1.0 * res["TP"] / (res["TP"] + res["FN"])
            f1 = 2 * precision * recall / (precision + recall)

        return precision, recall, f1

    # Generate score for the first two subtasks
    def gen_score(self, arr):
        sumf = 0
        sump=0
        sumr=0
        res=[]
        y = {"TP": 0, "FP": 0, "FN": 0, "TN": 0}
        for x in arr:
            p, r, f = self.get_value(x)
            sump+=p
            sumr+=r
            sumf += f#用于计算宏平均
            for z in x.keys():
                y[z] += x[z]#用于计算微平均
        mi_p, mi_r, mi_f = self.get_value(y)
        ma_p=sump*1.0/len(arr)
        ma_r=sumr*1.0/len(arr)
        ma_f=sumf*1.0/len(arr)
        res=(mi_f+ma_f)/2.0
        res=[mi_p,mi_r,mi_f,ma_p,ma_r,ma_f,res]
        return res

    # Generatue all scores
    def get_score(self, result):
        s1 = self.gen_score(result[0])
        return s1

    def getAccuracy(self,predict,truth,sig_value):
        result = [[]]
        sample_num=len(predict)
        for a in range(0, self.task2_cnt):
            result[0].append({"TP": 0, "FP": 0, "TN": 0, "FN": 0})
        for index in range(sample_num):
            predict_sample=np.where(predict[index]>=sig_value)[0]
            truth_sample=np.where(truth[index]==1)[0]
            result=self.gen_new_result(result,truth_sample,predict_sample)
        res=self.get_score(result)
        return res




if __name__=='__main__':
    judge=Judger()
    predict=np.array([[1],
                      [2],
                      [0]])
    truth = np.array([[1],
                      [1],
                      [0]])
    print(judge.getAccuracy(predict=predict,truth=truth))
