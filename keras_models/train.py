import keras
from sklearn.metrics import f1_score, log_loss
from addict import Dict
import time
import numpy as np

def model_run(model, patience, class_weights, model_name, trainDataGen, validDataGen, num_gpus=8, num_steps_each_eval=256, epochs=1, update_amount=0.9, verbosity=False):
    overall_loss_hist = []
    loss_hist = []
    val_loss_hist = []
    start = time.time()
    if num_steps_each_eval is None:
        num_steps_each_eval = len(trainDataGen) #do eval checks at end of epoch
    class_weight = np.array(class_weights)
    update_amount = 0.9
    best_f1 = -1
    exhausted_patience = 0
    batch_num = 0
    totalEpochs = 0
    validDataGen.batch_size = 256*num_gpus #force batch_size high cuz we want to get good performance and utilization across all
    while totalEpochs <= epochs:
            if batch_num == len(trainDataGen):
                trainDataGen.on_epoch_end() #if shuffle is turned on, the dataGen shuffles the data
                batch_num = 0
                totalEpochs += 1
                overall_loss_hist.append(np.mean(loss_hist))
                loss_hist = []
                print("Epoch {}/{}: loss: {}".format(totalEpochs, epochs, overall_loss_hist[totalEpochs-1]))

            loss = model.train_on_batch(trainDataGen[batch_num][0], trainDataGen[batch_num][1], class_weight=class_weight,)
            loss_hist.append(loss)
            if verbosity:
                print("Progress: {}/{}".format(batch_num, len(trainDataGen)))
                print("Epoch: {}, Step: {}/{}, Loss: {}".format(totalEpochs, batch_num, num_steps_each_eval, loss))
            if batch_num % num_steps_each_eval == 0:
                y_true = []
                for k in range(len(validDataGen)):
                    _, y_true_add = validDataGen[k]
                    y_true.append(y_true_add)
                y_true = np.concatenate(y_true)

                y_pred = model.predict_generator(validDataGen)
                y_pred = y_pred.argmax(1)
                val_loss_hist.append(log_loss(y_true.argmax(1), y_pred))
                if y_pred.mean() == int(y_pred.mean()):
                    problem_class = int(y_pred.mean())
                    class_weight = class_weight / update_amount
                    class_weight[problem_class] = class_weight[problem_class] * update_amount ** 2
                    print("predicted all {}".format(problem_class))
                    print("Updating Class Weight to: {}".format(class_weight))
                valid_f1 = f1_score(y_true.argmax(1), y_pred)
                if valid_f1 > best_f1:
                    print("f1 increased from {} to {}".format(best_f1, valid_f1))
                    best_f1 = valid_f1
                    keras.models.save_model(model, model_name)
                    exhausted_patience = 0
                else:
                    exhausted_patience += 1
                if exhausted_patience > patience:
                    break
            batch_num += 1

    return Dict({
        'history':
        {
            "loss": overall_loss_hist,
            "val_loss": val_loss_hist
         }
    })
