#pip install tensorflow==2.11.0

#pip install git+https://github.com/VenkateshwaranB/stellargraph.git

import pandas as pd
import numpy as np
import scipy.sparse as sp
import os

import stellargraph as sg
from stellargraph import StellarGraph
from stellargraph.mapper import FullBatchNodeGenerator
from stellargraph.layer import GCN

from tensorflow.keras import layers, optimizers, losses, metrics, Model
from sklearn import preprocessing, model_selection
from IPython.display import display, HTML
import matplotlib.pyplot as plt


well_list = ["16a","4","5a"]
well_name = ''
for j in range(len(well_list)):
  well_name = well_list[j]
  print(well_name)
  if well_name != '':
    for i in range(5, 11):
      cls_name = i
      for k in range(1, 5):
        time = k
        # the base_directory property tells us where it was downloaded to:
        cora_cites_file = os.path.join("/content/drive/MyDrive/NewZealand_data/mckee-"+str(well_name)+"/"+str(cls_name)+"/", "mckee-"+str(well_name)+" BF node for "+str(cls_name)+".csv")
        cora_content_file = os.path.join("/content/drive/MyDrive/NewZealand_data/mckee-"+str(well_name)+"/"+str(cls_name)+"/", "mckee-"+str(well_name)+" BF_full_edge "+str(cls_name)+".txt")

        cora_cites_cont = pd.read_csv(
            cora_cites_file)

        cora_cites = pd.DataFrame()
        #cora_cites['target'] = cora_cites['target'].str.replace('.', '')
        cora_cites['target'] = cora_cites_cont['full'].map(lambda x: str(x).replace('.', ''))
        cora_cites['source'] = cora_cites_cont['full1'].map(lambda x: str(x).replace('.', ''))

        cora_feature_names = [f"w{i}" for i in range(14)]

        cora_raw_content = pd.read_csv(
            cora_content_file,
            sep="\t",  # tab-separated
            header=None,  # no heading row
            names=["DEPTH", *cora_feature_names, "RESULTS"],  # set our own names for the columns
        )
        cora_raw_content['DEPTH'] = cora_raw_content['DEPTH'].map(lambda x: str(x).replace('.', ''))

        cora_content_str_subject = cora_raw_content.set_index("DEPTH")

        cora_content_no_subject = cora_content_str_subject.drop(columns="RESULTS")
        G = StellarGraph({"paper": cora_content_no_subject}, {"cites": cora_cites})

        node_subjects = cora_content_str_subject["RESULTS"]

        trainS = ''
        trainT = ''

        if well_name == "16a":
          trainS = 320
          trainT =143
        elif well_name == "4":
          trainS = 4506
          trainT =520
        elif well_name == "5a":
          trainS = 3553
          trainT =301

        train_subjects, test_subjects = model_selection.train_test_split(
            node_subjects, train_size=trainS, test_size=None, stratify=node_subjects
        )
        val_subjects, test_subjects = model_selection.train_test_split(
            test_subjects, train_size=trainT, test_size=None, stratify=test_subjects
        )

        target_encoding = preprocessing.LabelBinarizer()

        train_targets = target_encoding.fit_transform(train_subjects)
        val_targets = target_encoding.transform(val_subjects)
        test_targets = target_encoding.transform(test_subjects)

        generator = FullBatchNodeGenerator(G, method="gcn")

        train_gen = generator.flow(train_subjects.index, train_targets)

        gcn = GCN(
            layer_sizes=[16, 16], activations=["relu", "relu"], generator=generator, dropout=0.5
        )

        x_inp, x_out = gcn.in_out_tensors()

        predictions = layers.Dense(units=train_targets.shape[1], activation="softmax")(x_out)

        model = Model(inputs=x_inp, outputs=predictions)
        model.compile(
            optimizer=optimizers.Adam(lr=0.01),
            loss=losses.categorical_crossentropy,
            metrics=["acc"],
        )

        from tensorflow.keras.callbacks import EarlyStopping

        es_callback = EarlyStopping(monitor="val_acc", patience=50, restore_best_weights=True)

        val_gen = generator.flow(val_subjects.index, val_targets)

        history = model.fit(
            train_gen,
            epochs=200,
            validation_data=val_gen,
            verbose=2,
            shuffle=False,  # this should be False, since shuffling data means shuffling the whole graph
            callbacks=[es_callback],
        )

        hist_df = pd.DataFrame(history.history)

        if time == 1:
          hist_df.to_excel("/content/drive/MyDrive/NewZealand_data/mckee-"+str(well_name)+"/"+str(cls_name)+"/results/History_"+str(cls_name)+".xlsx")
        else:
          hist_df.to_excel("/content/drive/MyDrive/NewZealand_data/mckee-"+str(well_name)+"/"+str(cls_name)+"/results/History_"+str(cls_name)+"new"+str(time)+".xlsx")

        test_gen = generator.flow(test_subjects.index, test_targets)

        test_metrics = model.evaluate(test_gen)
        print("\nTest Set Metrics:")
        for name, val in zip(model.metrics_names, test_metrics):
            print("\t{}: {:0.4f}".format(name, val))

        all_nodes = node_subjects.index
        all_gen = generator.flow(all_nodes)
        all_predictions = model.predict(all_gen)

        node_predictions = target_encoding.inverse_transform(all_predictions.squeeze())

        df = pd.DataFrame({"Predicted": node_predictions, "True": node_subjects})

        if time == 1:
          df.to_excel("/content/drive/MyDrive/NewZealand_data/mckee-"+str(well_name)+"/"+str(cls_name)+"/results/Results_"+str(cls_name)+".xlsx")
        else:
          df.to_excel("/content/drive/MyDrive/NewZealand_data/mckee-"+str(well_name)+"/"+str(cls_name)+"/results/Results_"+str(cls_name)+" new"+str(time)+".xlsx")