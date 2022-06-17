#! -*- coding: utf-8 -*-

from keras import optimizers
from Capsule_Keras import *
from keras import utils
from keras import callbacks
from keras.models import Model
from keras.layers import *
from keras import backend as K
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
from pandas import DataFrame
from sklearn.model_selection import train_test_split
from sklearn.manifold import TSNE
import argparse
from scipy import sparse


#####################################################################################################################
#method for plot
def line_plot(num_classes, plot_x,ratio_plot,dotted_line,dotted_line_real):
    color = ['red', 'darksalmon', 'sienna', 'gold', 'olivedrab', 'darkgreen', 'chartreuse', 'darkcyan', 'deepskyblue',
             'blue', 'darkorchid', 'plum', 'm', 'mediumvioletred', 'k', 'palevioletred']
    for i in range(num_classes):
        if i == k:
            plt.plot(plot_x, ratio_plot[i], c=color[i], label=i + 1)
        else:
            plt.plot(plot_x, ratio_plot[i], c=color[i])

    if dotted_line > 0:
        dotted_line = dotted_line_real
        plt.plot([dotted_line, dotted_line], [1.0, 0], 'k--', linewidth=3.0)
    # if k == num_classes-1:
    plt.legend(loc='lower left')
    plt.xlabel('Masking genes along PC1')
    plt.ylabel('Prediction accuracy(%)')
    title = 'Primary capsule ' + str(k + 1)
    plt.title(title)

def scatter_plot(weightpca,select_genes):
    plt.scatter(weightpca[:, 0], weightpca[:, 1], color='r', s=1, alpha=0.3, label='gene')
    plt.scatter(weightpca[select_genes, 0], weightpca[select_genes, 1], color='b', s=1, alpha=0.8,
                label='select_gene')

    plt.ylabel('PC2')
    plt.xlabel('PC1')
    plt.title('Primary capsule ' + str(k + 1))


#####################################################################################################################
# system config
parser = argparse.ArgumentParser(description='scCapsNet-mask')


parser.add_argument('--inputdata', type=str, default='data/retina_data.npz', help='address for input data')
parser.add_argument('--inputcelltype', type=str, default='data/retina_celltype.npy', help='address for celltype label')
parser.add_argument('--num_classes', type=int, default=15, help='number of cell type')
parser.add_argument('--randoms', type=int, default=30, help='random number to split dataset')
parser.add_argument('--dim_capsule', type=int, default=32, help='dimension of the capsule')
parser.add_argument('--activation_F', type=str, default='relu', help='activation function')
parser.add_argument('--batch_size', type=int, default=400, help='training parameters_batch_size')
parser.add_argument('--epochs', type=int, default=15, help='training parameters_epochs')
parser.add_argument('--training', type=str, default='F', help='training model(T) or loading model(F) ')
parser.add_argument('--weights', type=str, default='data/retina_demo.weight', help='trained weights')
parser.add_argument('--plot_direction', type=str, default='one_side', help='display option, both_side or one_side')
parser.add_argument('--pc_slice', type=int, default=20, help='fineness divided along PC direction ')
parser.add_argument('--threshold', type=float, default=0.05, help='threshold for setting dotted line')


args = parser.parse_args()
inputdata = args.inputdata
inputcelltype = args.inputcelltype
num_classes = args.num_classes
randoms = args.randoms
z_dim = args.dim_capsule
activation_F = args.activation_F
epochs = args.epochs
batch_size = args.batch_size
training = args.training
weight = args.weights
plot_direction = args.plot_direction
pc_slice = args.pc_slice
threshold = args.threshold

#####################################################################################################################
#training data and test data
if inputdata[-3:] == 'npz':
    data = sparse.load_npz(inputdata)
    data = data.todense()
    data = np.asarray(data)
else:
    data = np.load(inputdata)

labels = np.load(inputcelltype)

print(type(data))
print(data.shape)

x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size = 0.1, random_state= randoms)
Y_test = y_test
Y_train = y_train

y_train = utils.to_categorical(y_train, num_classes)
y_test = utils.to_categorical(y_test, num_classes)

feature_size =  x_train.shape[1]
input_size = x_train.shape[1]
print(input_size)

#####################################################################################################################
#Model
x_in = Input(shape=(input_size,))
x = x_in
x_all = list(np.zeros((num_classes,1)))
encoders = []
for i in range(num_classes):
    x_all[i] = Dense(z_dim, activation=activation_F)(x_in)
    encoders.append(Model(x_in, x_all[i]))

x = Concatenate()(x_all)
x = Reshape((num_classes, z_dim))(x)
capsule = Capsule(num_classes, z_dim, 3, False)(x)
output = Lambda(lambda x: K.sqrt(K.sum(K.square(x), 2)), output_shape=(num_classes,))(capsule)

model = Model(inputs=x_in, outputs=output)

adam = optimizers.Adam(lr=0.00015,beta_1=0.9,beta_2=0.999,epsilon=1e-08)
model.compile(loss=lambda y_true,y_pred: y_true*K.relu(0.9-y_pred)**2 + 0.25*(1-y_true)*K.relu(y_pred-0.1)**2,
              optimizer=adam,
              metrics=['accuracy'])

model.summary()

if training=='T':
    model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              verbose=1,
              validation_data=(x_test, y_test))

    weight_name = "results/training_"+"n"+str(num_classes)+"_"+"r"+str(randoms)+"_"+"dim"+str(z_dim)+"_"+"e"+str(epochs)+"_"+"b"+str(batch_size)+"_"+".weight"
    model.save_weights(weight_name)
    model.load_weights(weight_name)
else:
    model.load_weights(weight)


#####################################################################################################################
#output Prediction probability for each type
print(data.shape)
result  = model.predict(data)
np.save("results/Prediction_probability.npy", result)

#####################################################################################################################
#Find cell type related genes
primary_capsule_encoder_weights = []
for i in range(num_classes):
    primary_capsule_encoder_weights.append(encoders[i].get_weights()[0])

subplotindex = 0
select_genes = []
dotted_line = -1
Lindex = 0

total_plot_x = []
total_ratio_plot = []
total_dotted_line = []
total_dotted_line_real = []
total_select_genes = []
total_weightpca = []

print()
print('Select cell type related genes ......')
for k in range(num_classes):
    print('primary capsule: ',k+1)
    totalweight = primary_capsule_encoder_weights[k]
    pca = PCA(n_components=16)
    pca.fit(totalweight)
    weightpca = pca.transform(totalweight)

    total_weightpca.append(weightpca)

    difference = (np.max(weightpca[:, 0]) - np.min(weightpca[:, 0])) / pc_slice
    PC_max = np.max(weightpca[:, 0])
    PC_min = np.min(weightpca[:, 0])

    ############################################################################################################
    #backward
    ratio_plot = np.zeros((num_classes, pc_slice + 1))
    x_new_test = copy.deepcopy(x_test)
    dotted_line = -1
    select_genes = []
    plot_x = []

    for j in range(pc_slice + 1):
        plot_x.append(PC_max - difference * j)
        # sub A select genes
        gene_count = 0
        for i in range(feature_size):
            if weightpca[i, 0] > (np.max(weightpca[:, 0]) - difference * j):
                gene_count = gene_count + 1
                rownum = x_new_test.shape[0]
                x_new_test[:, i] = np.zeros(rownum)
                if dotted_line < 0 and i not in select_genes:
                    select_genes.append(i)

        # sub B calculate the accuracy
        Y_pred = model.predict(x_new_test)
        Y_pred_order = np.argsort(Y_pred, axis=1)
        Y_pred_1 = Y_pred_order[:, num_classes - 1]
        Y_pred_real_order = np.sort(Y_pred, axis=1)
        Y_pred_real_1 = Y_pred_real_order[:, num_classes - 1]

        current_type = 0
        total = np.zeros((num_classes, 1))
        correct = np.zeros((num_classes, 1))

        for i in range(x_test.shape[0]):
            index_int = int(Y_test[i])
            if Y_test[i] == Y_pred_1[i]:
                correct[index_int] = correct[index_int] + 1
            total[index_int] = total[index_int] + 1

        ratio_drop = np.zeros((num_classes, 1))
        for i in range(len(total)):
            ratio_drop[i] = correct[i] / total[i]

        for i in range(len(total)):
            ratio_plot[i, j] = ratio_drop[i]

        for i in range(len(total)):
            ratio_plot[i, j] = ratio_drop[i]
            # find the position of dotted line
            if dotted_line < 0 and k == i and ratio_plot[i, j] < threshold:
                dotted_line = j

    dotted_line_real = PC_max - difference * dotted_line
    total_plot_x.append(plot_x)
    total_ratio_plot.append(ratio_plot)
    total_dotted_line.append(dotted_line)
    total_dotted_line_real.append(dotted_line_real)
    total_select_genes.append(select_genes)

    ############################################################################################################
    # forward
    ratio_plot = np.zeros((num_classes, pc_slice+1))
    x_new_test = copy.deepcopy(x_test)
    dotted_line = -1
    select_genes = []
    plot_x = []
    for j in range(pc_slice + 1):
        plot_x.append(PC_min + difference * j)
        # sub A select genes
        gene_count = 0
        for i in range(feature_size):
            if weightpca[i, 0] < (np.min(weightpca[:, 0]) + difference * j):
                gene_count = gene_count + 1
                rownum = x_new_test.shape[0]
                x_new_test[:, i] = np.zeros(rownum)
                if dotted_line < 0 and i not in select_genes:
                    select_genes.append(i)

        # sub B calculate the accuracy
        Y_pred = model.predict(x_new_test)
        Y_pred_order = np.argsort(Y_pred, axis=1)
        Y_pred_1 = Y_pred_order[:, num_classes - 1]
        Y_pred_real_order = np.sort(Y_pred, axis=1)
        Y_pred_real_1 = Y_pred_real_order[:, num_classes - 1]

        current_type = 0
        total = np.zeros((num_classes, 1))
        correct = np.zeros((num_classes, 1))

        for i in range(x_test.shape[0]):
            index_int = int(Y_test[i])
            if Y_test[i] == Y_pred_1[i]:
                correct[index_int] = correct[index_int] + 1
            total[index_int] = total[index_int] + 1

        ratio_drop = np.zeros((num_classes, 1))
        for i in range(len(total)):
            ratio_drop[i] = correct[i] / total[i]

        for i in range(len(total)):
            ratio_plot[i, j] = ratio_drop[i]

        for i in range(len(total)):
            ratio_plot[i, j] = ratio_drop[i]
            # find the position of dotted line
            if dotted_line < 0 and k == i and ratio_plot[i, j] < threshold:
                dotted_line = j

    dotted_line_real = PC_min + difference * dotted_line
    total_plot_x.append(plot_x)
    total_ratio_plot.append(ratio_plot)
    total_dotted_line.append(dotted_line)
    total_dotted_line_real.append(dotted_line_real)
    total_select_genes.append(select_genes)


#############################################################################################################
#Plotting
if plot_direction == 'both_side':
    plt.figure(figsize=(20, 4 * np.round(2 * num_classes / 4, 0)))
    plt.subplots_adjust(hspace=0.3, wspace=0.3)
    plt.figure(1)
    plt.figure(figsize=(20, 4 * np.round(2 * num_classes / 4, 0)))
    plt.subplots_adjust(hspace=0.3, wspace=0.3)
    plt.figure(2)

else:
    plt.figure(figsize=(20, 4 * np.round(num_classes / 4, 0)))
    plt.subplots_adjust(hspace=0.3, wspace=0.3)
    plt.figure(1)
    plt.figure(figsize=(20, 4 * np.round(2 * num_classes / 4, 0)))
    plt.subplots_adjust(hspace=0.3, wspace=0.3)
    plt.figure(2)

total_select_genes_one_side = []
for k in range(num_classes):
    weightpca = total_weightpca[k]
    if plot_direction == 'both_side':
        #backward
        plot_x = total_plot_x[2*k]
        ratio_plot = total_ratio_plot[2*k]
        dotted_line = total_dotted_line[2*k]
        dotted_line_real = total_dotted_line_real[2*k]
        select_genes = total_select_genes[2*k]

        plt.figure(1)
        Lindex = 2 * k + 1
        print('subplot: ' + str(Lindex))
        ax = plt.subplot(np.round(2 * num_classes / 4, 0), 4, Lindex)
        ax.invert_xaxis()
        line_plot(num_classes, plot_x, ratio_plot, dotted_line, dotted_line_real) #prediction accuracy

        # scatter plot
        plt.figure(2)
        Lindex = 2 * k + 1
        print('subplot: ' + str(Lindex))
        ax = plt.subplot(np.round(2 * num_classes / 4, 0), 4, Lindex)
        scatter_plot(weightpca, select_genes) #selected gene

        # forward
        plot_x = total_plot_x[2 * k + 1]
        ratio_plot = total_ratio_plot[2 * k + 1]
        dotted_line = total_dotted_line[2 * k + 1]
        dotted_line_real = total_dotted_line_real[2 * k + 1]
        select_genes = total_select_genes[2 * k + 1]

        #line plot
        plt.figure(1)
        Lindex = 2 * k + 2
        print('subplot: ' + str(Lindex))
        ax = plt.subplot(np.round(2 * num_classes / 4, 0), 4, Lindex)
        line_plot(num_classes, plot_x, ratio_plot, dotted_line, dotted_line_real) #prediction accuracy
        # scatter plot
        plt.figure(2)
        Lindex = 2 * k + 2
        print('subplot: ' + str(Lindex))
        ax = plt.subplot(np.round(2 * num_classes / 4, 0), 4, Lindex)
        scatter_plot(weightpca, select_genes) #selected gene

    else:
        dotted_line_B = total_dotted_line[2 * k]
        dotted_line_F = total_dotted_line[2 * k + 1]
        if (dotted_line_B<dotted_line_F and dotted_line_B>0) or dotted_line_F<0:
            direction = 'Backward'
            plot_x = total_plot_x[2 * k]
            ratio_plot = total_ratio_plot[2 * k ]
            dotted_line = total_dotted_line[2 * k]
            dotted_line_real = total_dotted_line_real[2 * k]
            select_genes = total_select_genes[2 * k]
        else:
            direction = 'Forward'
            plot_x = total_plot_x[2 * k + 1]
            ratio_plot = total_ratio_plot[2 * k + 1]
            dotted_line = total_dotted_line[2 * k + 1]
            dotted_line_real = total_dotted_line_real[2 * k + 1]
            select_genes = total_select_genes[2 * k + 1]

        total_select_genes_one_side.append(select_genes)
        plt.figure(1)
        Lindex = k + 1
        print('subplot: ' + str(Lindex))
        ax = plt.subplot(np.round(num_classes / 4, 0), 4, Lindex)
        if direction == 'Backward':
            ax.invert_xaxis()
        line_plot(num_classes, plot_x, ratio_plot, dotted_line, dotted_line_real)

        # scatter plot
        plt.figure(2)
        Lindex = k + 1
        print('subplot: ' + str(Lindex))
        ax = plt.subplot(np.round(num_classes / 4, 0), 4, Lindex)
        scatter_plot(weightpca, select_genes)

print("Plotting and saving......")
plt.figure(1)
plt.savefig("results/Prediction_accuracy_curve.png",dpi=500)
plt.figure(2)
plt.savefig("results/Choosen_genes.png",dpi=500)
plt.show()

if plot_direction == 'both_side':
    total_select_genes_dic = {}
    for i in range(len(total_select_genes)):
        total_select_genes_dic[i] = np.asarray(total_select_genes[i])
    np.save("results/total_select_genes.npy", total_select_genes_dic)
else:
    total_select_genes_one_side_dic = {}
    for i in range(len(total_select_genes_one_side)):
        total_select_genes_one_side_dic[i] = np.asarray(total_select_genes_one_side[i])
    np.save("results/total_select_genes_one_side.npy", total_select_genes_one_side_dic)



