# 导入关键的工具包
import torch
import json
import torch.nn as nn
import torch.optim as optim
# 导入自定义的类和评估函数
from bilstm_crf import BiLSTM_CRF
from evaluate import evaluate
# 导入工具包
from tqdm import tqdm
import json
import random
import time
import matplotlib.pyplot as plt


# 构建文本到id的映射函数
def prepare_sequence(seq, char_to_id):
    # 初始化空列表
    char_ids = []
    # 遍历中文字符串, 完成数字化的映射
    for i, ch in enumerate(seq):
        # 判断当前字符是都在映射字典中, 如果不在取UNK字符的编码来代替, 否则取对应的字符编码数字
        if char_to_id.get(ch):
            char_ids.append(char_to_id[ch])
        else:
            char_ids.append(char_to_id["UNK"])
    # 将列表封装成Tensor类型返回
    return torch.tensor(char_ids, dtype=torch.long)

# 指定字符编码表的文件路径
json_file = './data/char_to_id.json'

# 添加获取训练数据和验证数据的函数
def get_data():
    # 设置训练数据的路径和验证数据的路径
    train_data_file_path = "data/train_data.txt"
    validate_data_file_path = "data/validate_data.txt"
    # 初始化数据的空列表
    train_data_list = []
    valid_data_list = []
    # 因为每一行都是一个样本, 所以按行遍历文件即可
    for line in open(train_data_file_path, mode='r', encoding='utf-8'):
        # 每一行数据都是json字符串, 直接loads进来即可
        data = json.loads(line)
        train_data_list.append(data)
    # 同理处理验证数据集
    for line in open(validate_data_file_path, mode='r', encoding='utf-8'):
        data = json.loads(line)
        valid_data_list.append(data)
    # 最后以列表的形式返回训练数据集和验证数据集
    return train_data_list, valid_data_list


# 添加绘制损失曲线和评估曲线的函数
def save_train_history_image(train_history_list, validate_history_list,
                             history_image_path, data_type):
    # train_history_list: 训练历史结果数据
    # validate_history_list: 验证历史结果数据
    # history_image_path: 历史数据生成图像的保存路径
    # data_type: 数据类型
    # 直接开始画图
    plt.plot(train_history_list, label="Train %s History" % (data_type))
    plt.plot(validate_history_list, label="Validate %s History" % (data_type))
    plt.legend(loc="best")
    plt.xlabel("Epochs")
    plt.ylabel(data_type)
    plt.savefig(history_image_path.replace("plot", data_type))
    plt.close()


# if __name__ == '__main__':
    # char_to_id = json.load(open(json_file, mode='r', encoding='utf-8'))
    # 遍历样本示例数据, 进行编码映射
    # for line in train_data_list:
        # text, tag = line.get('text'), line.get('label')
        # res = prepare_sequence(text, char_to_id)
        # print(res)
        # print('******')

# if __name__ == '__main__':
#     train_data_list, valid_data_list = get_data()
#     print(train_data_list[:5])
#     print(len(train_data_list))
#     print('******')
#     print(valid_data_list[:5])
#     print(len(valid_data_list))

# 训练函数的主要代码部分
if __name__ == '__main__':
    # 首先确定超参数和训练数据的导入
    EMBEDDING_DIM = 200
    HIDDEN_DIM = 100
    train_data_list, validate_data_list = get_data()
    # 字符到id的映射已经提前准备好了, 直接读取文件即可
    char_to_id = json.load(open('./data/char_to_id.json', mode='r', encoding='utf-8'))
    # 直接将标签到id的映射字典作为超参数写定, 因为这个字典只和特定的任务有关系
    tag_to_ix = {"O": 0, "B-dis":1, "I-dis":2, "B-sym":3, "I-sym":4, "<START>":5, "<STOP>":6}

    # 直接构建模型
    model = BiLSTM_CRF(len(char_to_id), tag_to_ix, EMBEDDING_DIM, HIDDEN_DIM)
    # 直接选定优化器
    optimizer = optim.SGD(model.parameters(), lr=0.5, momentum=0.85, weight_decay=1e-4)

    # 调转字符标签和id值
    id_to_tag = {v:k for k, v in tag_to_ix.items()}
    # 调转字符编码中的字符和Id
    id_to_char = {v:k for k, v in char_to_id.items()}

    # 获取时间戳, 用于模型，图片，日志文件的名称
    time_str = time.strftime("%Y%m%d_%H%M%S", time.localtime(time.time()))
    model_saved_path = "model/bilstm_crf_state_dict_%s.pt" % (time_str)
    train_history_image_path = "log/bilstm_crf_train_plot_%s.png" % (time_str)
    log_file = open("log/train_%s.log"%(time_str), mode='w', encoding='utf-8')

    # 设定一个重要的超参数, 训练轮次
    epochs = 10
    # 将几个重要的统计量做初始化
    train_loss_history, train_acc_history, train_recall_history, train_f1_history = [], [], [], []
    validate_loss_history, validate_acc_history, validate_recall_history, validate_f1_history = [], [], [], []

    # 按照epochs进行轮次的训练和验证
    for epoch in range(epochs):
        tqdm.write("Epoch {}/{}".format(epoch+1, epochs))
        total_acc_length, total_prediction_length, total_gold_length, total_loss = 0, 0, 0, 0
        # 对于任意一轮epoch训练, 我们先进行训练集的训练
        for train_data in tqdm(train_data_list):
            model.zero_grad()
            # 取出训练样本
            sentence, tags = train_data.get("text"), train_data.get("label")
            # 完成数字化编码
            sentence_in = prepare_sequence(sentence, char_to_id)
            targets = torch.tensor([tag_to_ix[t] for t in tags], dtype=torch.long)
            # 计算损失值, model中计算损失值是通过neg_log_likelihood()函数完成的
            loss = model.neg_log_likelihood(sentence_in, targets) 
            loss.backward()
            optimizer.step()
            # 累加每一个样本的损失值
            step_loss = loss.data.numpy()
            total_loss += step_loss
            # 评估一下模型的表现
            score, best_path_list = model(sentence_in)
            # 得到降维的张量和最佳路径
            sentence_in_unq = sentence_in.unsqueeze(0)
            targets_unq = targets.unsqueeze(0)
            best_path_list_up = [best_path_list]
            step_acc, step_recall, f1_score, \
            acc_entities_length, predict_entities_length, gold_entities_length = \
            evaluate(sentence_in_unq.tolist(), targets_unq.tolist(), best_path_list_up, id_to_char, id_to_tag)

            # 累加三个重要的统计量, 预测正确的实体数, 模型预测出的总实体数, 真实标签的实体数
            total_acc_length += acc_entities_length
            total_prediction_length += predict_entities_length
            total_gold_length += gold_entities_length

        # 所有训练集数据在一个epoch遍历结束后, 打印阶段性的成果
        print("train:", total_acc_length, total_prediction_length, total_gold_length)
        # 如果真实预测出来的实体数大于0, 则计算准确率, 召回率, F1分值
        if total_prediction_length > 0:
            train_mean_loss = total_loss / len(train_data_list)
            train_epoch_acc = total_acc_length / total_prediction_length
            train_epoch_recall = total_acc_length / total_gold_length
            train_epoch_f1 = 2.0 * train_epoch_acc * train_epoch_recall / (train_epoch_acc + train_epoch_recall)
        else:
            log_file.write("train_total_prediction_length is zero!!!" + "\n")

                # 每一个epoch, 训练结束后直接在验证集上进行验证
        total_acc_length, total_prediction_length, total_gold_length, total_loss = 0, 0, 0, 0
        # 进入验证阶段最重要的一步就是保持模型参数不变, 不参与反向传播和参数更新
        with torch.no_grad():
            for validate_data in tqdm(validate_data_list):
                # 直接提取验证集的特征和标签, 并进行数字化映射
                sentence, tags = validate_data.get("text"), validate_data.get("label")
                sentence_in = prepare_sequence(sentence, char_to_id)
                targets = torch.tensor([tag_to_ix[t] for t in tags], dtype=torch.long)
                # 验证阶段的损失值依然是通过neg_log_likelihood函数进行计算
                loss = model.neg_log_likelihood(sentence_in, targets)
                # 验证阶段的解码依然是通过直接调用model中的forward函数进行
                score, best_path_list = model(sentence_in)
                # 累加每一个样本的损失值
                step_loss = loss.data.numpy()
                total_loss += step_loss
                sentence_in_unq = sentence_in.unsqueeze(0)
                targets_unq = targets.unsqueeze(0)
                best_path_list_up = [best_path_list]
                step_acc, step_recall,  f1_score,  acc_entities_length,  predict_entities_length, gold_entities_length = evaluate(sentence_in_unq.tolist(), targets_unq.tolist(), best_path_list_up, id_to_char, id_to_tag)
                # 累加三个重要的统计量, 预测正确的实体数, 预测的总实体数, 真实标签的实体数 
                total_acc_length += acc_entities_length
                total_prediction_length += predict_entities_length
                total_gold_length += gold_entities_length

        print("validate:", total_acc_length, total_prediction_length, total_gold_length)
        # 当准确预测的数量大于0, 并且总的预测标签量大于0, 计算验证集上的准确率, 召回率, F1值
        if total_acc_length > 0 and total_prediction_length > 0:
            validate_mean_loss = total_loss / len(validate_data_list)
            validate_epoch_acc = total_acc_length / total_prediction_length
            validate_epoch_recall = total_acc_length / total_gold_length
            validate_epoch_f1 = 2 * validate_epoch_acc * validate_epoch_recall / (validate_epoch_acc + validate_epoch_recall)
            log_text = "Epoch: %s | train loss: %.5f |train acc: %.3f |train recall: %.3f | train f1: %.3f" \
                       " | validate loss: %.5f | validate acc: %.3f | validate recall: %.3f | validate f1: %.3f" % \
                       (epoch, train_mean_loss, train_epoch_acc, train_epoch_recall, train_epoch_f1,
                        validate_mean_loss, validate_epoch_acc, validate_epoch_recall, validate_epoch_f1)
            log_file.write(log_text + "\n")

            # 将当前epoch的重要统计量添加进画图列表中
            train_loss_history.append(train_mean_loss)
            train_acc_history.append(train_epoch_acc)
            train_recall_history.append(train_epoch_recall)
            train_f1_history.append(train_epoch_f1)
            validate_loss_history.append(validate_mean_loss)
            validate_acc_history.append(validate_epoch_acc)
            validate_recall_history.append(validate_epoch_recall)
            validate_f1_history.append(validate_epoch_f1)
        else:
            log_file.write("validate_total_prediction_length is zero!" + "\n")

    # 当整个所有的轮次结束后, 说明模型训练完毕, 直接保存模型
    torch.save(model.state_dict(), model_saved_path)

    # 完成画图的功能代码
    # 将loss的历史数据画成图片
    save_train_history_image(train_loss_history, validate_loss_history, train_history_image_path, "Loss")

    # 将准确率的历史数据画成图片
    save_train_history_image(train_acc_history, validate_acc_history, train_history_image_path, "Acc")

    # 将召回率的历史数据画成图片
    save_train_history_image(train_recall_history, validate_recall_history, train_history_image_path, "Recall")

    # 将F1值的历史数据画成图片
    save_train_history_image(train_f1_history, validate_f1_history, train_history_image_path, "F1")

    print("Train Finished".center(100, "-"))

    # 训练结束后, 最后打印一下训练好的模型中的各个组成模块的参数详情
    for name, parameters in model.named_parameters():
        print(name, ":", parameters.size())




