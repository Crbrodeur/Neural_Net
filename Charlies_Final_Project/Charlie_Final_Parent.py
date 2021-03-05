import charlie_model_lib as models
import pandas as pd
import function_sandbox as fs


def final_deployment(list_models, csv_filename='Charlies_Dog_Cat_Classification.csv'):
    predictions = []
    models = list_models
    for model in models:
        predictions.append(model.deploy_model())
        model = 0

    temp = predictions.pop(0)
    for prediction in predictions:
        for temp_entry, row in zip(temp, prediction):
            temp_entry.append(row[2])

    temp[0].append('Majority Voting')
    for row in temp[1:]:
        row.append(fs.most_frequent(row[2:]))

    number_wrong =0

    print(temp[0][1])
    print(temp[0][len(temp[0])-1])

    for row in temp[1:]:

        if row[1] != row[len(row)-1]:
            number_wrong += 1
        else:
            continue
    message = 'FINAL MODEL ACCURACY: ' + str(((1-number_wrong/(len(temp)-1))*100)) + ' %'
    temp.append(message)
    print(message)

    my_df = pd.DataFrame(temp)
    my_df.to_csv(csv_filename, index=False, header=False)

def train_models(model_list):
    models = model_list
    for model in models:
        print(model.get_model_name())
        model.train_model()
        model.summarize_results()
        model.test_model()
        model.save_model_weights()


model_1 = models.VGG16BasedModel()
model_2 = models.VGG19BasedModel()
model_3 = models.ResNet50BasedModel()
model_4 = models.InceptionV3BasedModel()
model_5 = models.ResNetV2BasedModel()
model_6 = models.MobileNetBasedModel()
model_7 = models.XceptionBasedModel()
models = [model_1, model_2, model_3, model_4, model_5, model_6, model_7]
#
# train_models(models)
# final_deployment(models)
#
# train_models(models)
final_deployment(models)




