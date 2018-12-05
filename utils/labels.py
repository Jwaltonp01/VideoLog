# coding=utf-8
from sklearn.preprocessing import MultiLabelBinarizer

# TODO Make variables private

# img_labelsがまだ利用されていない！
# 女性画像最適化のラベル
from Constants import ModelType, Rank

# TODO Possible labels for the version 2-3 of the rank AI
# TODO These labels will allow specific traits within a image to be processed
# img_labels = [
#     "ADULT_COMIC",
#     "CHILD",
#     "NO_INFO",
#     "PUPIL_INFORMATION",
#     "ADULT_IMAGE",
#     "MULTIPLE_PEOPLE",
#     "PRIVATE_INFORMATION",
#     "FEMALE_PRIVATE_PART", "MALE_PRIVATE_PART",
#     "Animal",
#     "Comic",
#     "Landscape",
#     "TEST",
#     "RANK_1", "RANK_2", "RANK_3", "RANK_4", "RANK_5", "RANK_NG",
#     "OBJECT",
#     "MALE", "FEMALE",
#     "SNOW",
#     "TOO_MUCH_SNOW",
#     "BAD_LIGHTING",
#     "BAD_IMG_QUALITY",
#     "FULL_FACE", "PARTIAL_FACE", "NO_FACE",
#     "MAKEUP", "NOMAKEUP",
#     "EYES_OPEN", "EYES_CLOSED",
#     "FACE_MASK",
#     "NOT_FACING_FORWARD"
# ]

__ng__ = [
    "AdultComic", "Children", "NoInformation", "PupilUniform", "AdultImage",
    "ManyPeople", "PrivateInformation", "WomanPrivatePart", "ManPrivatePart", "Animal",
    "Comic", "Landscape", "PersonNoFace", "PersonWithFace", "Text",
    "Rank_NG", "Object", "Male"
    ]

# ランクAIのラベル （古い）
__ranking_labels_v1__ = [
    "AdultImage", "PrivateInformation", "Animal", "Comic", "Lanscape", "Text",
    "Rank_1", "Rank_2", "Rank_3", "Rank_4", "Rank_5", "Rank_NG", "Object", "Male", "Female"
    ]

# ランクAIのラベル (現在）
__ranking_labels_v2__ = [
    "AdultImage", "PrivateInformation", "Animal", "Comic", "Landscape", "Text",
    "Rank_1", "Rank_2", "Rank_3", "Rank_4", "Rank_5", "Rank_NG", "Object", "Male", "Female",
    "TooMuchSnow", "BadQuality", "BadAngle", "PartFace", "NoFace", "ManyPeople"
    ]

# ランクAIのラベル
__ranking_labels_v3__ = [
    "Rank_1", "Rank_2", "Rank_3", "Rank_4", "Rank_5", "Rank_NG", "Male", "Female",
    "Animal", "NA", "Erotic", "Landscape", "TooMuchSnow", "NoEye", "TooClose",
    "Snow", "BadQuality", "BadAngle", "PartFace", "NoFace", "ManyPeople"
    ]

# NG Labels
__vidlog_labels_v1__ = ["Boob", "Vagina", "Penis", "Anus"]

__all_ranks__ = ["Rank_1", "Rank_2", "Rank_3", "Rank_4", "Rank_5", "Rank_NG"]
__ok_ranks__ = ["Rank_3", "Rank_4", "Rank_5"]
__img_traits__ = ["Male", "Female", "Erotic", "Animal", "Landscape", "Text",
                  "Snow", "BadQuality", "BadAngle", "PartFace", "NoFace", "ManyPeople"]

__stat_labels_str__ = [str(i) for i in range(0, 20)]
__stat_labels__ = [float(i) for i in range(0, 20)]


def get_jpn_label(label):
    if label == "AdultImage" or label == "Erotic":
        val = u"エロい画像".encode("UTF-8")
    elif label == "PrivateInformation":
        val = u"個人情報".encode("UTF-8")
    elif label == "Animal":
        val = u"動物".encode("UTF-8")
    elif label == "Comic":
        val = u"漫画".encode("UTF-8")
    elif label == "Landscape":
        val = u"背景".encode("UTF-8")
    elif label == "Text":
        val = u"テキスト".encode("UTF-8")
    elif label == "Text":
        val = u"テキスト".encode("UTF-8")
    elif label == "Rank_1":
        val = u"ランク１".encode("UTF-8")
    elif label == "Rank_2":
        val = u"ランク２".encode("UTF-8")
    elif label == "Rank_3":
        val = u"ランク３".encode("UTF-8")
    elif label == "Rank_4":
        val = u"ランク４".encode("UTF-8")
    elif label == "Rank_5":
        val = u"ランク５".encode("UTF-8")
    elif label == "Rank_NG":
        val = u"ランクNG".encode("UTF-8")
    elif label == "Object":
        val = u"オブジェクト".encode("UTF-8")
    elif label == "NA":
        val = u"プロフィール画像として使えない".encode("UTF-8")
    elif label == "Male":
        val = u"男性".encode("UTF-8")
    elif label == "Female":
        val = u"女性".encode("UTF-8")
    elif label == "TooMuchSnow":
        val = u"SNOWを使いすぎ".encode("UTF-8")
    elif label == "BadQuality":
        val = u"画質がよくない".encode("UTF-8")
    elif label == "BadAngle":
        val = u"角度の角度がよくない".encode("UTF-8")
    elif label == "PartFace":
        val = u"顔の一部しか見えない".encode("UTF-8")
    elif label == "NoFace":
        val = u"顔が写っていない".encode("UTF-8")
    elif label == "NoEye":
        val = u"目が見えない".encode("UTF-8")
    elif label == "ManyPeople":
        val = u"第三者の顔が見える".encode("UTF-8")
    elif label == "Snow":
        val = u"SNOW使用".encode("UTF-8")
    else:
        val = get_jpn_default_label()

    print("Labels.get_jpn_label -- Return value: " + str(val))
    return val


def get_jpn_default_label():
    return u"無し".encode("UTF-8")


def get_ng_list():
    return __ng__


def get_all_rank_list():
    return __all_ranks__


def get_ok_rank_list():
    return __ok_ranks__


def get_rank_label(arg):
    return __ranking_labels_v3__[arg]


def get_vidlog_label(index):
    return __vidlog_labels_v1__[index]


def get_rank_labels():
    return __ranking_labels_v3__


def get_stat_labels():
    return __stat_labels__


def get_vidlog_labels():
    return __vidlog_labels_v1__


def get_stat_labels_str():
    return __stat_labels_str__


def num_rank_labels():
    return len(__ranking_labels_v3__)


def num_stat_labels():
    return len(__stat_labels__)


def num_vidlog_labels():
    return len(__vidlog_labels_v1__)


def get_label_ids(paths, model_type):
    """
    Load numpy array (binary matrix) containing all predetermined labels
    When loading multi labels containing folder for the image in label.label.label format.
    Meaning separate each label by a "."
    """
    ids = []

    # Encode class values as integers
    encoder = get_label_encoder(model_type)

    # ラベル
    for i in paths:
        # List of labels in raw format
        raw_labels = i.split("/")[-2]
        # List of labels for img format
        rl = raw_labels.split(".")
        # Record labels to label database
        ids.append(rl)

    return encoder.fit_transform(ids)


def get_label_encoder(model_type):
    encoder = None

    # Encode class values as integers
    if model_type == ModelType.Rank:
        encoder = MultiLabelBinarizer(classes=__ranking_labels_v3__)
    elif model_type == ModelType.Stat:
        encoder = MultiLabelBinarizer(classes=__stat_labels_str__)
    elif model_type == ModelType.VidLog:
        encoder = MultiLabelBinarizer(classes=__vidlog_labels_v1__)

    return encoder
