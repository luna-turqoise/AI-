import torch
import torch.nn as nn
import json

# 构建函数评估模型的准确率和召回率, F1值
def evaluate(sentence_list, gold_tags, predict_tags, id2char, id2tag):
    # sentence_list: 代表待评估 的原始样本
    # gold_tags: 真实的标签序列
    # predict_tags: 模型预测出来的标签序列
    # id2char: 代表数字化映射字典
    # id2tag: 代表标签的数字换映射字典
    # 初始化真实实体的集合以及每隔实体的字+标签列表
    gold_entities, gold_entity = [], []
    # 初始化模型预测实体的几个以及每个实体的字+标签列表
    predict_entities, predict_entity = [], []
    # 遍历当前待评估的每个句子
    for line_no, sentence in enumerate(sentence_list):
        # 迭代句子的每个字符
        for char_no in range(len(sentence)):
            # 判断: 弱句子的某一个id值等于0, 意味着是<PAD>, 后面将全部是0, 直接跳出内循环
            if sentence[char_no] == 0:
                break
            # 获取当前句子中的每一个文字字符
            char_text = id2char[sentence[char_no]]
            # 获取当前字符的真实实体标签类型
            gold_tag_type = id2tag[gold_tags[line_no][char_no]]
            # 获取当前字符的预测实体标签类型
            predict_tag_type = id2tag[predict_tags[line_no][char_no]]
            # 首先判断真实实体是否可以加入列表中
            # 首先判断id2tag的第一个字符是否为B, 表示一个实体的开始
            if gold_tag_type[0] == "B":
                # 将实体字符和类型加入实体列表中
                gold_entity = [char_text + "/" + gold_tag_type]
            # 判断id2tag第一个字符是否为I, 表示一个实体的中间到结尾
            # 总体的判断条件:1.类型要以I开始 2.entity不为空 3.实体类型相同
            elif gold_tag_type[0] == "I" and len(gold_entity) != 0 \
                  and gold_entity[-1].split("/")[1][1:] == gold_tag_type[1:]:
                # 满足条件的部分追加进实体列表中
                gold_entity.append(char_text + "/" + gold_tag_type)
            # 判断id2tag的第一个字符是否为O, 并且entity非空, 实体已经完成了全部的判断
            elif gold_tag_type[0] == "O" and len(gold_entity) != 0:
                # 增加一个唯一标识
                gold_entity.append(str(line_no) + "_" + str(char_no))
                # 将一个完整的命名实体追加进最后的列表中
                gold_entities.append(gold_entity)
                # 将gold_eneity清空, 以便判断下一个命名实体
                gold_entity = []
            else:
                gold_entity = []

            # 接下来判断预测出来的命名实体
            # 第一步首先判断id2tag的第一个字符是否为B, 表示实体的开始
            if predict_tag_type[0] == "B":
                predict_entity = [char_text + "/" + predict_tag_type]
            # 判断第一个字符是否是I, 并且entity非空, 并且实体类型相同
            elif predict_tag_type[0] == "I" and len(predict_entity) != 0 \
                 and predict_entity[-1].split("/")[1][1:] == predict_tag_type[1:]:
                predict_entity.append(char_text + "/" + predict_tag_type)
            # 判断第一个字符是否为O, 并且entity非空, 代表一个完整的实体已经识别完毕, 可以追加进列表中
            elif predict_tag_type[0] == "O" and len(predict_entity) != 0:
                # 增加一个唯一标识
                predict_entity.append(str(line_no) + "_" + str(char_no))
                # 将识别出来的完整实体追加进最终的列表中
                predict_entities.append(predict_entity)
                # 将predict_entity清空, 以便判断下一个命名实体
                predict_entity = []
            else:
                predict_entity = []

    # 当外层for循环结束后, 整个判断流程结束, 需要进行指标计算
    acc_entities = [entity for entity in predict_entities if entity in gold_entities]
    # 计算正确预测出来的实体个数
    acc_entities_length = len(acc_entities)
    # 计算预测了多少个实体
    predict_entities_length = len(predict_entities)
    # 计算真实实体的个数
    gold_entities_length = len(gold_entities)

    # 如果准确实体的个数大于0, 则计算准确率,召回率, F1值
    if acc_entities_length > 0:
        step_acc = float(acc_entities_length / predict_entities_length)
        step_recall = float(acc_entities_length / gold_entities_length)
        f1_score = 2.0 * step_acc * step_recall / (step_acc + step_recall)
        return step_acc, step_recall, f1_score, acc_entities_length, predict_entities_length, gold_entities_length
    else:
        return 0, 0, 0, acc_entities_length, predict_entities_length, gold_entities_length

    

# 真实标签数据
tag_list = [
    [0, 0, 3, 4, 0, 3, 4, 0, 0, 0, 0, 0, 0, 0, 3, 4, 0, 0, 0, 0],
    [0, 0, 3, 4, 0, 3, 4, 0, 0, 0, 0, 0, 0, 0, 3, 4, 0, 0, 0, 0],
    [0, 0, 3, 4, 0, 3, 4, 0, 3, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [3, 4, 4, 4, 4, 0, 0, 0, 0, 0, 0, 0, 1, 2, 2, 2, 0, 0, 0, 0],
    [0, 0, 1, 2, 2, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [3, 4, 0, 3, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 1, 2, 2, 2, 2, 2, 2, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 3, 4, 4, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
]

# 预测标签数据
predict_tag_list = [
    [0, 0, 0, 0, 0, 3, 4, 0, 0, 0, 0, 0, 0, 0, 3, 4, 0, 0, 0, 0],
    [0, 0, 3, 4, 0, 3, 4, 0, 0, 0, 0, 0, 0, 0, 3, 4, 0, 0, 0, 0],
    [0, 0, 3, 4, 0, 3, 4, 0, 3, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [3, 4, 4, 4, 4, 0, 0, 0, 0, 0, 0, 0, 1, 2, 2, 2, 0, 0, 0, 0],
    [0, 0, 1, 2, 2, 2, 2, 0, 0, 0, 0, 0, 0, 3, 4, 0, 0, 0, 0, 0],
    [3, 4, 0, 3, 4, 0, 0, 1, 2, 0, 0, 3, 4, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 1, 2, 2, 2, 2, 2, 2, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 3, 4, 4, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
]

# 编码与字符对照字典
id2char = {0: '<PAD>', 1: '确', 2: '诊', 3: '弥', 4: '漫', 5: '大', 6: 'b', 7: '细', 8: '胞', 9: '淋', 10: '巴', 11: '瘤', 12: '1', 13: '年', 14: '反', 15: '复', 16: '咳', 17: '嗽', 18: '、', 19: '痰', 20: '4', 21: '0', 22: ',', 23: '再', 24: '发', 25: '伴', 26: '气', 27: '促', 28: '5', 29: '天', 30: '。', 31: '生', 32: '长', 33: '育', 34: '迟', 35: '缓', 36: '9', 37: '右', 38: '侧', 39: '小', 40: '肺', 41: '癌', 42: '第', 43: '三', 44: '次', 45: '化', 46: '疗', 47: '入', 48: '院', 49: '心', 50: '悸', 51: '加', 52: '重', 53: '胸', 54: '痛', 55: '3', 56: '闷', 57: '2', 58: '多', 59: '月', 60: '余', 61: ' ', 62: '周', 63: '上', 64: '肢', 65: '无', 66: '力', 67: '肌', 68: '肉', 69: '萎', 70: '缩', 71: '半'}

# 编码与标签对照字典
id2tag = {0: 'O', 1: 'B-dis', 2: 'I-dis', 3: 'B-sym', 4: 'I-sym'}

# 输入的数字化sentences_sequence, 由下面的sentence_list经过映射函数sentence_map()转化后得到
sentence_list = [
    "确诊弥漫大b细胞淋巴瘤1年",
    "反复咳嗽、咳痰40年,再发伴气促5天。",
    "生长发育迟缓9年。",
    "右侧小细胞肺癌第三次化疗入院",
    "反复气促、心悸10年,加重伴胸痛3天。",
    "反复胸闷、心悸、气促2多月,加重3天",
    "咳嗽、胸闷1月余, 加重1周",
    "右上肢无力3年, 加重伴肌肉萎缩半年"
]


def sentence_map(sentence_list, char_to_id, max_length):
    sentence_list.sort(key=lambda c:len(c), reverse=True)
    sentence_map_list = []
    for sentence in sentence_list:
        sentence_id_list = [char_to_id[c] for c in sentence]
        padding_list = [0] * (max_length-len(sentence))
        sentence_id_list.extend(padding_list)
        sentence_map_list.append(sentence_id_list)
    return torch.tensor(sentence_map_list, dtype=torch.long)

char_to_id = {"<PAD>":0}

SENTENCE_LENGTH = 20

for sentence in sentence_list:
    for _char in sentence:
        if _char not in char_to_id:
            char_to_id[_char] = len(char_to_id)

sentences_sequence = sentence_map(sentence_list, char_to_id, SENTENCE_LENGTH)


if __name__ == '__main__':
    accuracy, recall, f1_score, acc_entities_length, predict_entities_length, true_entities_length = evaluate(sentences_sequence.tolist(), tag_list, predict_tag_list, id2char, id2tag)

    print("accuracy:",                  accuracy,
          "\nrecall:",                  recall,
          "\nf1_score:",                f1_score,
          "\nacc_entities_length:",     acc_entities_length,
          "\npredict_entities_length:", predict_entities_length,
          "\ntrue_entities_length:",    true_entities_length)


