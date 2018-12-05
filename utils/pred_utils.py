# coding=utf-8
import csv
import cv2
import numpy as np
import pandas as pd

from Constants import ModelType, Rank
from utils import labels, img_utils


# Update complete...
# Generating Rank Statistics...
def process_api_predictions(rank_pred, stat_pred, rank_stats):
    rank_args = pd.DataFrame(rank_pred, columns=labels.get_rank_labels()).to_dict()
    stat_args = pd.DataFrame(stat_pred, columns=labels.get_stat_labels()).to_dict()

    output = {}
    r_details = []

    # Add any additional details for image
    for index in rank_args:
        index_val = rank_args.get(index).get(0)
        if index_val >= .75:
            if index not in labels.get_all_rank_list():
                r_details.append(index)

    # Append additional image details if present
    if len(r_details) > 0:
        output["img_details"] = r_details
    else:
        output["img_details"] = "None"

    print(rank_args)
    # Rank DataFrame
    r_args = get_rank_df(rank_pred)

    # Calculate Image Rank
    pred_rank = "No data..."
    rank_val = np.nanmax(r_args.values)
    output["confidence"] = np.multiply(rank_val, 100.0)

    for key in r_args.keys():
        if r_args.get(key=str(key)).get(0) == rank_val:
            pred_rank = str(key)

    output["pred_rank"] = pred_rank
    output["rank_val"] = rank_val

    # Process the Stat args
    sn = np.argmax(stat_args.values())
    # Stat final predictions
    sfp = stat_args.keys()[sn]

    # Ranking stats for Stat Model final prediction
    r_stats = rank_stats.loc[rank_stats['rank_pt_index'] == int(sfp)]

    if not pred_rank == "Rank_NG":
        output["num_clicks"] = float(r_stats["click_median"].values)
        output["num_likes"] = float(r_stats["like_median"].values)
        output["popularity"] = float(r_stats["popularity_avg"].values)
    else:
        output["num_clicks"] = 0
        output["num_likes"] = 0
        output["popularity"] = 0

    # # Print Stats
    # print("\n-----------------------------------\nprocess_api_predictions()\n-----------------------------------")
    # print("Prediction: " + str(output.get("rank")))
    # print("Rank: " + str(pred_rank))
    # print("Confidence: " + str(output.get("confidence")))
    # print("num_clicks: " + str(output.get("num_clicks")))
    # print("num_likes: " + str(output.get("num_likes")))
    # print("popularity: " + str(output.get("popularity")))
    # print("Details: " + str(output.get("img_details")))
    # print("\n-----------------------------------\n")

    return output


def get_rank_df(pred):
    rank_args = pd.DataFrame(pred, columns=labels.get_rank_labels()).to_dict()
    data = [{"Rank_1": rank_args.get("Rank_1").get(0),
             "Rank_2": rank_args.get("Rank_2").get(0),
             "Rank_3": rank_args.get("Rank_3").get(0),
             "Rank_4": rank_args.get("Rank_4").get(0),
             "Rank_5": rank_args.get("Rank_5").get(0),
             "Rank_NG": rank_args.get("Rank_NG").get(0)}]
    return pd.DataFrame(data=data, columns=labels.get_all_rank_list())


def get_jpn_rank(pred_rank, rank_val):
    """
    Append Rank prediction
    """

    if pred_rank == "Rank_1":
        if rank_val >= .75:
            output = u"ギリギリセーフ。".encode("utf-8")
        elif .75 > rank_val > .25:
            output = u"NGにはしないけど、いいかな〜".encode("utf-8")
        else:
            output = u"決められないけど、NGにはしない・・・いいかな・・・".encode("utf-8")

    elif pred_rank == "Rank_2":
        if rank_val >= .75:
            output = u"ギリギリではないけど、いい感じでもない。".encode("utf-8")
        elif .75 > rank_val > .25:
            output = u"ちょっと駄目っぽいけど、セーフ〜".encode("utf-8")
        else:
            output = u"決められないけど、セーフにする・・・いいかな・・・".encode("utf-8")

    elif pred_rank == "Rank_3":
        if rank_val >= .75:
            output = u"いい感じ。".encode("utf-8")
        elif .75 > rank_val > .25:
            output = u"いい感じかも〜".encode("utf-8")
        else:
            output = u"決められないけど、いいかも・・・".encode("utf-8")

    elif pred_rank == "Rank_4":
        if rank_val >= .75:
            output = u"とてもいい感じ！".encode("utf-8")
        elif .75 > rank_val > .25:
            output = u"めっちゃいい感じかも〜".encode("utf-8")
        else:
            output = u"決められないけど、とてもいい感じかも・・・".encode("utf-8")

    else:
        if rank_val >= .75:
            output = u"これはNGだ。".encode("utf-8")
        elif .75 > rank_val > .25:
            output = u"これはNGだと思う〜".encode("utf-8")
        else:
            output = u"決められないけど、これはNGかも・・・".encode("utf-8")
    return output


def process_rank_prediction(pred_args):
    pred_args = pd.DataFrame(data=pred_args, columns=labels.get_rank_labels()).to_dict()

    output = {}
    img_details = []

    # Add any additional details for image
    for index in pred_args:
        index_val = pred_args.get(index).get(0)
        if index_val >= .75:
            if index not in labels.get_all_rank_list():
                img_details.append(index)

    # Append additional image details
    if len(img_details) > 0:
        output["img_details"] = img_details
    else:
        output["img_details"] = ""

    rank_data = {"Rank_1": pred_args.get("Rank_1").get(0),
                 "Rank_2": pred_args.get("Rank_2").get(0),
                 "Rank_3": pred_args.get("Rank_3").get(0),
                 "Rank_4": pred_args.get("Rank_4").get(0),
                 "Rank_5": pred_args.get("Rank_5").get(0),
                 "Rank_NG": pred_args.get("Rank_NG").get(0)}

    rank_args = pd.DataFrame(data=[rank_data], columns=labels.get_all_rank_list())

    # Calculate Image Rank
    rank_val = np.nanmax(rank_args.values)
    output["confidence"] = np.multiply(rank_val, 100.0)

    for key in rank_args.keys():
        if rank_args.get(key=str(key)).get(0) == rank_val:
            output["rank"] = str(key)

    return rank_data, output


def process_stat_prediction(pred_args):
    pred_args = pd.DataFrame(data=pred_args, columns=labels.get_stat_labels()).to_dict()
    prediction = []

    for loc in pred_args:
        # Confidence
        conf = pred_args.get(loc).get(0)
        #
        if conf >= 0.75:
            prediction.append(loc)

    fp = np.argmax(pred_args.values())
    final_pred = pred_args.keys()

    return prediction, final_pred[fp]


def get_model_accuracy(prerequisite, pred_batch, lab, model_type):
    # Encoded label  format_example = ([0,0,0,0,0,1,0,...])

    print("=================================")
    print("get_model_accuracy()")
    print("=================================\n")

    # Base confidence for model predictions
    req_conf = float(prerequisite) / float(100)

    print("Batch size: " + str(pred_batch.__len__()))
    print("Prerequisite: " + str(prerequisite))
    print("Max Confidence: " + str(np.nanmax([pred_batch[i].max() for i in range(pred_batch.__len__())])) + "\n")
    print("Average Max Confidence: " + str(
        np.nanmean([pred_batch[i].max() for i in range(pred_batch.__len__())])) + "\n")

    TP = 0
    FP = 0
    TN = 0
    FN = 0

    for i in range(pred_batch.__len__()):
        # Results for rank_model
        if model_type == ModelType.Rank:
            # Raw Ranking Predictions
            r_data = pd.DataFrame(data=[pred_batch[i]], columns=labels.get_rank_labels()).to_dict()

            # True label for image
            l_args = [i for i, idx in enumerate(np.array(lab[i])) if idx == 1]

            # Actual Labels for current image
            true_labels = [labels.get_rank_labels()[index] for index in l_args]

            # Model Confidence levels for Ranks
            r_dict = pd.DataFrame(data=[{"Rank_1": r_data.get("Rank_1").get(0),
                                         "Rank_2": r_data.get("Rank_2").get(0),
                                         "Rank_3": r_data.get("Rank_3").get(0),
                                         "Rank_4": r_data.get("Rank_4").get(0),
                                         "Rank_5": r_data.get("Rank_5").get(0),
                                         "Rank_NG": r_data.get("Rank_NG").get(0)}])

            # Model Final Rank Prediction
            model_pred = r_dict.keys()[np.argmax(r_dict.values)]

            # If the model has high confidence
            if r_data.get(model_pred).get(0) >= req_conf:
                # High confident correct prediction
                if str(model_pred) in true_labels:
                    TP += 1
                # High confident incorrect prediction
                else:
                    FP += 1

            # If the model has low confidence
            else:
                # Low confident correct prediction
                if str(model_pred) in true_labels:
                    TN += 1
                # Low confident incorrect prediction
                else:
                    FN += 1

        # Results for stat_model
        else:
            # Max prediction confidence index
            pred = pred_batch[i]
            img_label = lab[i]

            # Model prediction
            model_pred = list(pred).index(pred.max())

            # True label for image
            img_label = [i for i, idx in enumerate(np.array(img_label)) if idx == 1]

            # Model prediction (single prediction)
            pred_confidence = pred_batch[i].max()

            # If the model has high confidence
            if pred_confidence >= req_conf:
                # High confident correct prediction
                if model_pred == img_label:
                    TP += 1
                # High confident incorrect prediction
                else:
                    FP += 1

            # If the model has low confidence
            else:
                # Low confident correct prediction
                if np.array_equal(pred_batch[i], np.array(lab[i])):
                    TN += 1
                # Low confident incorrect prediction
                else:
                    FN += 1

    print_performance(TP, FP, FN, TN)
    accuracy, precision, recall, f1 = get_performance(TP, FP, FN, TN)

    col = ["Prerequisite", "Accuracy", "Precision", "Recall", "F1_Score"]
    df_data = {col[0]: prerequisite, col[1]: accuracy, col[2]: precision, col[3]: recall, col[4]: f1}
    return accuracy, precision, recall, f1, df_data


def get_performance(TP, FP, FN, TN):
    """
    モデルのパフォーマンスを計算する

    :param TP: 自身あり正解に答えた数
    :param FP: 自身あり不正解に答えた数
    :param FN: 自身なし正解に答えた数
    :param TN: 自身なし不正解に答えた数
    :return:
        accuracy: モデルの予測精度
        precision: 思い出す能力の割合
        recall: 関係ある事の思い出す能力の割合
        f1: パフォーマンスのスコア
    """

    if float(TP + FN) == 0:
        recall = 0
    else:
        recall = float(TP / float(TP + FN))

    if float(TP + FP) == 0:
        precision = 0
    else:
        precision = float(TP / float(TP + FP))

    if float(TP + TN + FP + FN) == 0:
        accuracy = 0
    else:
        accuracy = float((TP + TN) / float(TP + TN + FP + FN))

    if float(recall + precision) <= 0:
        f1 = 0
    else:
        f1 = 2 * recall * precision / (recall + precision)

    return accuracy, precision, recall, f1


def print_performance(TP, FP, FN, TN):
    """
    モデルのパフォーマンスをアウトプットする

    :param TP: 自身あり正解に答えた数
    :param FP: 自身あり不正解に答えた数
    :param FN: 自身なし正解に答えた数
    :param TN: 自身なし不正解に答えた数
    :return:
            :return:
        accuracy: モデルの予測精度
        precision: 思い出す能力の割合
        recall: 関係ある事の思い出す能力の割合
        f1: パフォーマンスのスコア
    """

    accuracy, precision, recall, f1 = get_performance(TP, FP, FN, TN)
    print("TP: " + str(TP) + "\nFP: " + str(FP) + "\nTN: " + str(TN) + "\nFN: " + str(FN))
    print("Acc " + str(accuracy) + " -- Prec: " + str(precision) + " -- Recall: " + str(recall) + " -- F1: " + str(f1))


def make_batch_stat_predictions(model, img_cache, getCSV, csv_name, preview_img):
    valid_img_data, valid_path_data = img_utils.load_cache_data(img_cache, file_no=0)

    csv_data = []
    field_names = ["File_Name", "Predictions", "Validity", "Raw_Output"]

    prediction_results = []

    if model is not None and valid_img_data is not None:
        for i in range(len(valid_img_data)):
            img_index = valid_img_data[i]
            orig_label = valid_path_data[i].split("/")[-2]
            img_name = valid_path_data[i].split("/")[-1]
            valid_labels = orig_label.split(".")

            pImg = np.reshape(a=img_index, newshape=(-1, 224, 224, 3))
            p_data = model.predict(pImg, batch_size=1, verbose=1)
            pr, fr = process_stat_prediction(p_data)

            prediction_results.append([pr, fr])

            if getCSV:
                validity = False
                for v in valid_labels:
                    if v in pr:
                        validity = True

                csv_data.append({field_names[0]: str(img_name),
                                 field_names[1]: str(fr),
                                 field_names[2]: str(validity),
                                 field_names[3]: str(pr).strip("[]")})

                print("Stat predictions -- Index[" + str(i) + "] -- Prediction: "
                      + str(fr) + " -- Validity: " + str(validity) + " -- Raw_Output: "
                      + str(pr).strip("[]"))

            if len(valid_img_data) == 1 and preview_img:
                cv2.imshow("img[" + str(i) + "]", cv2.imread(valid_path_data[i]))
                while True:
                    k = cv2.waitKeyEx(30) & 0xFF
                    if k == 27:
                        break
                cv2.destroyAllWindows()
    if getCSV:
        with open('logs/' + csv_name + '.csv', "w+") as csvfile:
            np.sort(csv_data)
            writer = csv.DictWriter(csvfile, fieldnames=field_names)
            writer.writeheader()
            writer.writerows(csv_data)
            print("Completed saving stat predictions to csv...")

    return prediction_results


def get_simple_stat_prediction(model, img_args, batch_size):
    pimg = np.reshape(a=img_args, newshape=(-1, 224, 224, 3))
    pdata = model.predict(pimg, batch_size=batch_size, verbose=1)
    pred_args = pd.DataFrame(pdata, columns=labels.get_stat_labels_str()).to_dict()
    fp = np.argmax(pred_args.values())
    return pred_args.keys()[fp]


#
def get_rank_predictions(model, valid_img_data, valid_path_data, getCSV, csv_name, img_range):
    csv_data = []
    field_names = ["File", "Prediction", "Validity", "Rank_1", "Rank_2", "Rank_3", "Rank_4", "Rank_5", "Rank_NG", "Details"]


    prediction_results = []

    if model is not None and valid_img_data is not None:
        print("get_prediction -- img_data.shape: " + str(np.shape(valid_img_data)))

        for i in img_range:
            img_index = valid_img_data[i]

            pImg = np.reshape(a=img_index, newshape=(-1, 224, 224, 3))
            p_data = model.predict(pImg, batch_size=1, verbose=1)
            rank_args, pred_args = process_rank_prediction(p_data)

            prediction_results.append([rank_args, pred_args])

            if getCSV:
                orig_label = valid_path_data[i].split("/")[-2]
                img_name = valid_path_data[i].split("/")[-1]
                valid_labels = orig_label.split(".")

                validity = False
                if pred_args.get("rank") in valid_labels:
                    validity = True

                csv_data.append({field_names[0]: str(img_name),
                                 field_names[1]: str(pred_args.get("rank")),
                                 field_names[2]: str(validity),
                                 field_names[9]: str(pred_args.get("img_details")).strip("[]"),
                                 field_names[3]: ("%.4f" % (rank_args.get(field_names[3]))),
                                 field_names[4]: ("%.4f" % (rank_args.get(field_names[4]))),
                                 field_names[5]: ("%.4f" % (rank_args.get(field_names[5]))),
                                 field_names[6]: ("%.4f" % (rank_args.get(field_names[6]))),
                                 field_names[7]: ("%.4f" % (rank_args.get(field_names[7]))),
                                 field_names[8]: ("%.4f" % (rank_args.get(field_names[8])))})

                print("Index[" + str(i) + "] -- Prediction: "
                      + str(pred_args.get("rank"))) + " -- Validity: " + str(validity)
    if getCSV:
        with open('logs/predictions/' + csv_name + '.csv', "w+") as csvfile:
            np.sort(csv_data)
            writer = csv.DictWriter(csvfile, fieldnames=field_names)
            writer.writeheader()
            writer.writerows(csv_data)
            print("Completed saving predictions to csv...")

    return prediction_results
