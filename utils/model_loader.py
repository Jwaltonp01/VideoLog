# coding=utf-8
from keras.layers.core import Dense, Activation
from keras.layers.pooling import GlobalAveragePooling2D
from keras.models import Model
from keras.optimizers import SGD
from models import densenet169
from utils import img_utils
import os


def ini_model(model):
    """
    DEMOの実行

    注意：
        これを実行する前にRankAIとStatAIの起動を確認してください。
    """

    di = os.path.dirname(os.getcwd()) + "\\VideoLog\\dummy.png"
    print("Dummy image location: " + str(di))

    # DEMOの為にNOISE画像を作成する
    demo_image = img_utils.load_pred_img(di, 224, 224)
    # Rank AI raw prediction
    pred = model.predict(demo_image, batch_size=1, verbose=1)

    if len(pred) > 0:
        print(str("\nAI model successfully initiated...\n"))


def get_base_model(final_layer, final_layer_name, img_input, weights_path, num_classes, lr=1e-03):
    # Truncate and replace softmax layer for transfer learning
    # Cannot use models.layers.pop() since models is not of Sequential() type
    # The method below works since pre-trained weights are stored in layers but not in the models
    x_newfc = GlobalAveragePooling2D(name='pool' + str(final_layer_name))(final_layer)
    x_newfc = Dense(num_classes, name='fc6')(x_newfc)
    x_newfc = Activation('sigmoid', name='prob')(x_newfc)

    model = Model(img_input, x_newfc)

    model.load_weights(weights_path, by_name=True)

    # Learning rate is changed to 0.001
    sgd = SGD(lr=lr, decay=1e-6, momentum=0.9, nesterov=True)

    model.compile(optimizer=sgd, loss='binary_crossentropy', metrics=['accuracy'])

    return model


def get_new_model(weights_path, num_classes, model, img_input, final_layer, lr):
    model.load_weights(weights_path, by_name=True)

    # Truncate and replace softmax layer for transfer learning
    # Cannot use models.layers.pop() since models is not of Sequential() type
    # The method below works since pre-trained weights are stored in layers but not in the models
    x_newfc = GlobalAveragePooling2D(name='pool5')(final_layer)
    x_newfc = Dense(num_classes, name='fc6')(x_newfc)
    x_newfc = Activation('sigmoid', name='prob')(x_newfc)

    model = Model(img_input, x_newfc)

    # Learning rate is changed to 0.001
    sgd = SGD(lr=lr, decay=1e-6, momentum=0.9, nesterov=True)

    model.compile(optimizer=sgd, loss='binary_crossentropy', metrics=['accuracy'])

    return model


def load_model(weight_path, num_classes):
    # Load DenseNet169 base models
    base_model, final_layer, final_layer_name, img_input = densenet169.create_model()

    # AIのモデルを復活する
    model = get_base_model(weights_path=weight_path,
                           num_classes=num_classes,
                           img_input=img_input,
                           final_layer=final_layer,
                           final_layer_name=final_layer_name,
                           lr=1e-03)
    return model


def load_new_model(old_weight_path, old_class_num, new_class_num, lr):
    if lr > 0:
        lr = lr
    else:
        lr = 1e-03

    # Load DenseNet169 base models
    base_model, final_layer, final_layer_name, img_input = densenet169.create_model()

    # Load previously trained models
    prev_model = get_base_model(weights_path=old_weight_path,
                                num_classes=old_class_num,
                                img_input=img_input,
                                final_layer=final_layer,
                                final_layer_name=final_layer_name,
                                lr=1e-03)

    # Load and return new models
    return get_new_model(weights_path=old_weight_path,
                         num_classes=new_class_num,
                         model=prev_model,
                         img_input=img_input,
                         final_layer=final_layer,
                         lr=lr)
